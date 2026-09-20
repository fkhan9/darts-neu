
r'''
Organizes a flat NEU-CLS folder (e.g. Cr_1.bmp, In_23.bmp, ...) into
class-named subfolders so torchvision.datasets.ImageFolder can read it.

Usage:
    python organize.py --src "C:\path\to\flat_neu_cls" --dst "C:\path\to\NEU-CLS-sorted"

By default COPIES files (safe, leaves your original download untouched).
Pass --move to move instead of copy.
'''

import argparse
import shutil
from pathlib import Path

# Official NEU-CLS class prefixes -> full class names
PREFIX_MAP = {
    "Cr": "crazing",
    "In": "inclusion",
    "Pa": "patches",
    "PS": "pitted_surface",
    "RS": "rolled-in_scale",
    "Sc": "scratches",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to flat folder of NEU-CLS images")
    parser.add_argument("--dst", required=True, help="Path to write class-sorted output")
    parser.add_argument("--move", action="store_true", help="Move instead of copy")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        raise SystemExit(f"Source folder does not exist: {src}")

    # Create class subfolders up front
    for class_name in PREFIX_MAP.values():
        (dst / class_name).mkdir(parents=True, exist_ok=True)

    counts = {name: 0 for name in PREFIX_MAP.values()}
    unmatched = []

    files = list(src.glob("*.*"))
    print(f"Found {len(files)} files in {src}")

    for f in files:
        if not f.is_file():
            continue
        prefix = f.stem.split("_")[0]  # "Cr_101.bmp" -> "Cr"
        class_name = PREFIX_MAP.get(prefix)
        if class_name is None:
            unmatched.append(f.name)
            continue
        target = dst / class_name / f.name
        if args.move:
            shutil.move(str(f), str(target))
        else:
            shutil.copy2(str(f), str(target))
        counts[class_name] += 1

    print("\nSorted counts:")
    total = 0
    for class_name, n in counts.items():
        print(f"  {class_name:<16} {n}")
        total += n
    print(f"  {'TOTAL':<16} {total}")

    if unmatched:
        print(f"\nWARNING: {len(unmatched)} files had no matching prefix, left untouched:")
        for name in unmatched[:10]:
            print(f"  {name}")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")

    if total != 1800:
        print(f"\nNOTE: expected 1800 total images, got {total}. Worth double-checking "
              f"the source folder isn't missing files or mixing in something unexpected.")
    else:
        print("\n1800/1800 sorted correctly — matches the expected NEU-CLS count.")


if __name__ == "__main__":
    main()