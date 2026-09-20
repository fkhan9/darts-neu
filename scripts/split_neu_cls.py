r"""
Splits the class-sorted NEU-CLS folder into train/images and validation/images
subfolders, matching the exact directory shape get_neu_cls_loaders() already
expects (so dataset.py needs zero changes).

train/images/<class>/    -> used by dataset.py's own 50/50 internal split for
                             the bilevel architecture-search train/val sets
validation/images/<class>/ -> kept as the genuine held-out TEST set (dataset.py
                             calls this "validation" but it is your final test set)

Usage:
    python split_neu_cls.py --src "C:\path\to\NEU-CLS-sorted" --dst "C:\path\to\NEU-CLS-final" --test_frac 0.2 --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Class-sorted NEU-CLS folder (6 subfolders)")
    parser.add_argument("--dst", required=True, help="Output root (will contain train/images, validation/images)")
    parser.add_argument("--test_frac", type=float, default=0.2, help="Fraction held out as final test set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--move", action="store_true", help="Move instead of copy")
    args = parser.parse_args()

    random.seed(args.seed)
    src = Path(args.src)
    dst = Path(args.dst)

    class_dirs = [d for d in src.iterdir() if d.is_dir()]
    if not class_dirs:
        raise SystemExit(f"No class subfolders found in {src}")

    train_root = dst / "train" / "images"
    val_root = dst / "validation" / "images"

    total_train, total_val = 0, 0

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        images = sorted(class_dir.glob("*.*"))
        random.shuffle(images)

        n_val = round(len(images) * args.test_frac)
        val_files = images[:n_val]
        train_files = images[n_val:]

        (train_root / class_name).mkdir(parents=True, exist_ok=True)
        (val_root / class_name).mkdir(parents=True, exist_ok=True)

        for f in train_files:
            target = train_root / class_name / f.name
            shutil.move(str(f), str(target)) if args.move else shutil.copy2(str(f), str(target))
        for f in val_files:
            target = val_root / class_name / f.name
            shutil.move(str(f), str(target)) if args.move else shutil.copy2(str(f), str(target))

        print(f"{class_name:<18} train={len(train_files):<4} val/test={len(val_files):<4} total={len(images)}")
        total_train += len(train_files)
        total_val += len(val_files)

    print(f"\n{'TOTAL':<18} train={total_train:<4} val/test={total_val:<4} total={total_train + total_val}")
    print(f"\nOutput written to:\n  {train_root}\n  {val_root}")
    print("\nPoint dataset.py's data_root at:", dst)


if __name__ == "__main__":
    main()