"""Run local Hugging Face image classification model.

Supports:
- Classify one image or all images in a folder
- Live camera mode (overlay top-K predictions)

Example:
  python hugging.py --image yolo/scripts/bio/images.webp --model-dir yolo/scripts/trash-clasiffier-biodegradable
  python hugging.py --camera 0 --model-dir yolo/scripts/trash-clasiffier-biodegradable --topk 1
"""
from pathlib import Path
import argparse
import json
import sys
import time

from PIL import Image

import torch
import torch.nn.functional as F


def load_processor_and_model(model_dir, device=None):
    # Import lazily to avoid optional heavy deps at module import time
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        processor = None
        try:
            processor = AutoImageProcessor.from_pretrained(model_dir)
        except Exception:
            # fallback
            try:
                from transformers import AutoFeatureExtractor

                processor = AutoFeatureExtractor.from_pretrained(model_dir)
            except Exception:
                processor = None

        model = AutoModelForImageClassification.from_pretrained(model_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to load processor/model from {model_dir}: {e}")

    use_device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(use_device)
    model.eval()
    return processor, model, use_device


def load_labels(model):
    id2label = {}
    cfg_map = getattr(model.config, "id2label", None)
    if cfg_map:
        # id2label can be dict with str keys or ints
        try:
            id2label = {int(k): v for k, v in cfg_map.items()}
        except Exception:
            # maybe already list-like
            if isinstance(cfg_map, list):
                id2label = {i: lbl for i, lbl in enumerate(cfg_map)}
            else:
                # last resort
                id2label = cfg_map
    return id2label


def collect_image_paths(src: Path):
    if src.is_file():
        return [src]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    imgs = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in exts]
    return imgs


def predict_images(model, processor, device, image_paths, topk=3):
    from transformers import logging

    batch_size = 8
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            probs = F.softmax(out.logits, dim=-1)
            topk_probs, topk_inds = torch.topk(probs, k=min(topk, probs.shape[-1]), dim=-1)

        for pth, p_list, i_list in zip(batch, topk_probs.cpu().tolist(), topk_inds.cpu().tolist()):
            results.append((pth, p_list, i_list))
    return results


def run_image_mode(args):
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Model dir not found: {model_dir}")
        sys.exit(1)

    processor, model, device = load_processor_and_model(str(model_dir), device=args.device)
    id2label = load_labels(model)

    src = Path(args.image)
    images = collect_image_paths(src)
    if not images:
        print("No images found at provided path")
        sys.exit(1)

    res = predict_images(model, processor, device, images, topk=args.topk)
    out = []
    for pth, probs, inds in res:
        print(pth)
        preds = []
        for prob, idx in zip(probs, inds):
            label = id2label.get(idx, str(idx)) if id2label else str(idx)
            print(f"  {label}: {prob:.4f}")
            preds.append({"label": label, "prob": prob})
        out.append({"image": str(pth), "predictions": preds})

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"Saved results to {out_path}")


def run_camera_mode(args):
    import cv2
    
    def _choose_font_scale(text, max_width, font=cv2.FONT_HERSHEY_SIMPLEX, thickness=2):
        scale = 0.8
        while scale >= 0.3:
            (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
            if w <= max_width:
                return scale
            scale -= 0.05
        return 0.3

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Model dir not found: {model_dir}")
        sys.exit(1)

    try:
        processor, model, device = load_processor_and_model(str(model_dir), device=args.device)
    except Exception as e:
        print("Model load failed:", e)
        sys.exit(1)

    id2label = load_labels(model)

    cap = cv2.VideoCapture(int(args.camera))
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        sys.exit(1)

    last_time = 0
    interval = max(0.05, float(args.interval))
    print("Press 'q' to quit, 's' to save snapshot")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        now = time.time()
        label_text = ""
        if now - last_time >= interval:
            last_time = now
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                inputs = processor(images=pil, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model(**inputs)
                    probs = F.softmax(out.logits, dim=-1)
                    topk_probs, topk_inds = torch.topk(probs, k=min(args.topk, probs.shape[-1]), dim=-1)
                    parts = []
                    for p, idx in zip(topk_probs[0].cpu().tolist(), topk_inds[0].cpu().tolist()):
                        lbl = id2label.get(idx, str(idx)) if id2label else str(idx)
                        parts.append(f"{lbl}:{p:.2f}")
                    label_text = " | ".join(parts)
            except Exception as e:
                label_text = f"err"

        # overlay
        display = frame.copy()
        if label_text:
            # compute scale to fit width
            canvas_w = display.shape[1]
            max_w = max(50, canvas_w - 20)
            scale = _choose_font_scale(label_text, max_w)
            thickness = 2 if scale >= 0.6 else 1
            cv2.putText(display, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thickness)

        cv2.imshow("Live Classifier", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            # save snapshot
            outp = Path(args.save_dir) if args.save_dir else Path.cwd()
            outp.mkdir(parents=True, exist_ok=True)
            fname = outp / f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(str(fname), display)
            print(f"Saved snapshot {fname}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Hugging Face trash classifier runner")
    p.add_argument("--model-dir", default="yolo/scripts/trash-clasiffier-biodegradable", help="Local HF model folder")
    p.add_argument("--image", help="Image file or folder to classify")
    p.add_argument("--camera", help="Camera device id (enter for live mode)", default=None)
    p.add_argument("--device", help="cpu or cuda (default auto)", default=None)
    p.add_argument("--topk", type=int, default=1, help="Top K labels to return")
    p.add_argument("--output", help="Optional JSON output file for image mode")
    p.add_argument("--interval", default=0.3, help="Live mode inference interval in seconds")
    p.add_argument("--save-dir", help="Directory to save snapshots (live mode)")
    args = p.parse_args()

    if args.image and args.camera:
        print("Please provide either --image or --camera, not both")
        sys.exit(1)

    if args.image:
        run_image_mode(args)
    elif args.camera is not None:
        # if user passed empty string, default to 0
        args.camera = args.camera if args.camera != "" else "0"
        run_camera_mode(args)
    else:
        print("No mode selected. Use --image <path> or --camera <id>")
        p.print_help()


if __name__ == "__main__":
    main()
