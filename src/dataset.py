"""
NEU-CLS dataset loading for DARTS search.

Expected folder structure (standard Kaggle NEU-CLS mirror):
    data/NEU-CLS/train/images/<class_name>/*.jpg
    data/NEU-CLS/validation/images/<class_name>/*.jpg

We only use the images/ subfolders -- the annotations/ folders (XML
bounding boxes) are for the detection variant of this dataset and are
not needed for classification.

NEU-CLS images are grayscale (1 channel), but our Network's stem is
built for 3-channel input (following the original DARTS/CIFAR-10
convention). Rather than changing the stem, we convert grayscale ->
3-channel by repeating the single channel, via transforms.Grayscale
(num_output_channels=3). This keeps model_search.py unchanged and is
the standard, low-risk fix for adapting a CIFAR-style stem to
single-channel domains.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


NEU_MEAN = [0.5, 0.5, 0.5]   # placeholder; replace with computed dataset stats if time allows
NEU_STD = [0.5, 0.5, 0.5]


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),   # defects have no canonical "up" on a rolled strip
        transforms.ToTensor(),
        transforms.Normalize(NEU_MEAN, NEU_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(NEU_MEAN, NEU_STD),
    ])

    return train_transform, eval_transform


def get_neu_cls_loaders(data_root, batch_size=32, val_split_for_arch=0.5, num_workers=0):
    """
    data_root: path to the NEU-CLS folder, e.g. 'data/NEU-CLS'
               (expects data_root/train/images/<class>/*.jpg and
                data_root/validation/images/<class>/*.jpg)

    DARTS's bilevel search needs its OWN train/val split (separate from
    any final-evaluation test set). So we split the dataset's train/
    folder in half:
      - one half -> DARTS's "train" set (updates weights w)
      - other half -> DARTS's "val" set (updates architecture alpha)
    The dataset's own validation/ folder is kept completely separate,
    reserved as a genuine held-out test set for final, unbiased
    reporting.
    """
    train_transform, eval_transform = get_transforms()

    full_train = datasets.ImageFolder(root=f'{data_root}/train/images', transform=train_transform)
    test_set = datasets.ImageFolder(root=f'{data_root}/validation/images', transform=eval_transform)

    n_total = len(full_train)
    n_arch_val = int(n_total * val_split_for_arch)
    n_train = n_total - n_arch_val

    generator = torch.Generator().manual_seed(42)  # reproducible split
    train_subset, arch_val_subset = random_split(full_train, [n_train, n_arch_val], generator=generator)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    arch_val_loader = DataLoader(arch_val_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, arch_val_loader, test_loader, full_train.classes