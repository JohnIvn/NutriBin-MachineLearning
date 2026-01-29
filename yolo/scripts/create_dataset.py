"""Convert a simple class-folder dataset into YOLO detection format.

This script takes images organized as:

  yolo/data/images/<class_name>/*.jpg

and creates a YOLO-style dataset with train/val splits where each
image receives a single bounding box that covers the whole image
(useful to reuse classification data for YOLO detection experiments).

Outputs:
- yolo/data/images/train, yolo/data/images/val (flat image files)
- yolo/data/labels/train, yolo/data/labels/val (one .txt per image)
- yolo/data/data.yaml for Ultralytics/YOLO training

Usage: run from repository root or from this folder. No external
dependencies beyond the Python standard library are required.
"""

from pathlib import Path
import shutil
import random
import json


ROOT = Path(__file__).resolve().parents[1]
DATA_IMAGES = ROOT / 'data' / 'images'
OUT = ROOT / 'data'


def make_dirs():
    for d in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        (OUT / d).mkdir(parents=True, exist_ok=True)


def gather_classes():
    # Ignore output subfolders (train/val) if script is re-run from same tree
    ignore = {'train', 'val'}
    classes = [p.name for p in DATA_IMAGES.iterdir() if p.is_dir() and p.name not in ignore]
    classes.sort()
    return classes


def split_and_copy(classes, val_ratio=0.2, seed=42):
    random.seed(seed)
    for cls_idx, cls in enumerate(classes):
        src_dir = DATA_IMAGES / cls
        images = [p for p in src_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        random.shuffle(images)
        n_val = int(len(images) * val_ratio)
        val_imgs = images[:n_val]
        train_imgs = images[n_val:]

        for src in train_imgs:
            dst = OUT / 'images' / 'train' / src.name
            try:
                # avoid copying a file onto itself
                if src.resolve() == dst.resolve():
                    continue
                shutil.copy2(src, dst)
            except PermissionError as e:
                print(f"Warning: could not copy {src} -> {dst}: {e}")
                continue
            write_label(dst.name, cls_idx, 'train')

        for src in val_imgs:
            dst = OUT / 'images' / 'val' / src.name
            try:
                if src.resolve() == dst.resolve():
                    continue
                shutil.copy2(src, dst)
            except PermissionError as e:
                print(f"Warning: could not copy {src} -> {dst}: {e}")
                continue
            write_label(dst.name, cls_idx, 'val')


def write_label(image_name, cls_idx, split):
    # Single box covering the entire image (x_center y_center width height), normalized
    line = f"{cls_idx} 0.5 0.5 1.0 1.0\n"
    lbl_path = OUT / 'labels' / split / (Path(image_name).with_suffix('.txt').name)
    lbl_path.write_text(line)


def write_data_yaml(classes):
    data_yaml = {
        'train': str((OUT / 'images' / 'train').resolve()),
        'val': str((OUT / 'images' / 'val').resolve()),
        'nc': len(classes),
        'names': classes,
    }
    (OUT / 'data.yaml').write_text(json.dumps(data_yaml, indent=2))


def save_class_indices(classes):
    mapping = {i: name for i, name in enumerate(classes)}
    (ROOT / 'outputs').mkdir(exist_ok=True)
    (ROOT / 'outputs' / 'class_indices.json').write_text(json.dumps(mapping, indent=2))


def main():
    make_dirs()
    classes = gather_classes()
    if not classes:
        print('No class subfolders found in', DATA_IMAGES)
        return
    split_and_copy(classes)
    write_data_yaml(classes)
    save_class_indices(classes)
    print('Created YOLO-format dataset under', OUT)


if __name__ == '__main__':
    main()

