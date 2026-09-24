"""
Discretize the search result into a fixed architecture, retrain it from
scratch (no bilevel split -- just ordinary supervised training on the full
training set), and evaluate on the genuinely held-out test set.

Usage:
    python train_final.py                      # runs a fresh search first, then retrains
    python train_final.py --genotype_file g.txt # retrains from an already-saved genotype

Save a genotype from a completed search with:
    print(model.genotype(), file=open('genotype.txt', 'w'))
in train_search.py, right after the search loop finishes.
"""

import sys
sys.path.insert(0, 'src')

import argparse
import ast
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model_final import NetworkFinal
from genotypes import Genotype
from dataset import get_neu_cls_loaders, get_transforms
from torchvision import datasets


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--genotype_file', type=str, default='genotype.txt',
                         help='Path to a saved genotype (from model.genotype(), printed to a text file)')
    parser.add_argument('--data_root', type=str, default='data/NEU-CLS-final')
    parser.add_argument('--C', type=int, default=16,
                         help='Channels for the final model. Can be larger than the search-phase C, '
                              'since a fixed architecture is far cheaper in memory than a supernet.')
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.025)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    with open(args.genotype_file, 'r') as f:
        genotype_str = f.read().strip()
    genotype = eval(genotype_str, {'Genotype': Genotype})
    print('Loaded genotype:', genotype)

    # Full training set (no bilevel split needed here -- this is ordinary
    # supervised training of one fixed architecture, not a search).
    train_transform, eval_transform = get_transforms()
    train_set = datasets.ImageFolder(root=f'{args.data_root}/train/images', transform=train_transform)
    test_set = datasets.ImageFolder(root=f'{args.data_root}/validation/images', transform=eval_transform)
    classes = train_set.classes
    print('Classes:', classes)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = NetworkFinal(C=args.C, num_classes=len(classes), layers=args.layers, genotype=genotype).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Final model parameters: {n_params:,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        epoch_loss, n_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        train_acc = evaluate(model, train_loader, device)
        print(f'Epoch {epoch + 1}/{args.epochs} | loss: {epoch_loss / n_batches:.4f} | train_acc: {train_acc:.4f}')

    test_acc = evaluate(model, test_loader, device)
    print(f'\nFinal discretized-architecture, held-out test accuracy: {test_acc:.4f}')
    print(f'(compare against supernet accuracy from train_search.py -- these are not the same measurement)')

    torch.save(model.state_dict(), 'checkpoints/final_model.pt')
    print('Saved final model to checkpoints/final_model.pt')


if __name__ == '__main__':
    main()