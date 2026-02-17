from pathlib import Path
import argparse
import json
import sys

from PIL import Image

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification


def load_labels_from_file(path: Path):
    try:
        j = json.loads(path.read_text())
        # If dict of id->label
        if isinstance(j, dict):
            # normalize keys to ints
            return {int(k): v for k, v in j.items()}
        # If list of labels
        if isinstance(j, list):
            return {i: label for i, label in enumerate(j)}
    except Exception:
        pass
    return None


def collect_image_paths(src: Path):
    if src.is_file():
        return [src]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    imgs = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in exts]
    return imgs


def predict_batch(model, processor, images, device, topk=3):
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_inds = torch.topk(probs, k=min(topk, probs.shape[-1]), dim=-1)
    return topk_probs.cpu().tolist(), topk_inds.cpu().tolist()


def main():
    p = argparse.ArgumentParser(description="Run local Hugging Face image classifier")
    p.add_argument("image", help="Image file or directory of images")
    p.add_argument("--model-dir", default="yolo/scripts/trash-clasiffier-biodegradable", help="Path to local model folder")
    p.add_argument("--device", default=None, help="cpu or cuda (auto if omitted)")
    p.add_argument("--topk", type=int, default=3, help="Top K predictions to show")
    p.add_argument("--labels-file", default=None, help="Optional JSON file with id->label mapping or list of labels")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Model dir not found: {model_dir}")
        sys.exit(1)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading processor and model from {model_dir} on {device}")
    try:
        processor = AutoImageProcessor.from_pretrained(model_dir)
    except Exception:
        # Backwards compat
        from transformers import AutoFeatureExtractor

        processor = AutoFeatureExtractor.from_pretrained(model_dir)

    model = AutoModelForImageClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Try to obtain labels mapping
    id2label = {}
    if getattr(model.config, "id2label", None):
        id2label = {int(k): v for k, v in model.config.id2label.items()} if isinstance(model.config.id2label, dict) else model.config.id2label

    # fallback: labels file
    if args.labels_file:
        lf = Path(args.labels_file)
        if lf.exists():
            labels = load_labels_from_file(lf)
            if labels:
                id2label = labels

    # fallback: try config.json in model dir
    if not id2label:
        cfg = model_dir / "config.json"
        if cfg.exists():
            labels = load_labels_from_file(cfg)
            if labels:
                id2label = labels

    image_paths = collect_image_paths(Path(args.image))
    if not image_paths:
        print("No images found at provided path")
        sys.exit(1)

    batch_size = 8
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        probs, inds = predict_batch(model, processor, images, device, topk=args.topk)
        for pth, p_list, i_list in zip(batch_paths, probs, inds):
            print(pth)
            for prob, idx in zip(p_list, i_list):
                label = id2label.get(idx, str(idx))
                print(f"  {label}: {prob:.4f}")
            print()


if __name__ == "__main__":
    main()
