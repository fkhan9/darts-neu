import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model_search import Network
from architect import Architect
from dataset import get_neu_cls_loaders


def plot_curves(history, out_path='checkpoints/search_curves.png'):
    epochs = range(1, len(history['w_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history['w_loss'], label='weight loss (w)', marker='o')
    axes[0].plot(epochs, history['a_loss'], label='architecture loss (alpha)', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('DARTS Search: Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], label='train accuracy', marker='o')
    axes[1].plot(epochs, history['val_acc'], label='arch-val accuracy', marker='o')
    axes[1].axhline(y=1/6, color='gray', linestyle='--', label='random chance (1/6)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('DARTS Search: Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f'Saved training curves to {out_path}')


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    C = 8
    layers = 4
    epochs = 5
    batch_size = 16

    train_loader, arch_val_loader, test_loader, classes = get_neu_cls_loaders(
        'data/NEU-DET', batch_size=batch_size
    )
    print('Classes:', classes)

    model = Network(C=C, num_classes=len(classes), layers=layers).to(device)
    criterion = nn.CrossEntropyLoss()

    w_optimizer = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
    architect = Architect(model)

    history = {'w_loss': [], 'a_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        arch_val_iter = iter(arch_val_loader)
        epoch_w_loss = 0.0
        epoch_a_loss = 0.0
        n_batches = 0

        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)

            try:
                x_val, y_val = next(arch_val_iter)
            except StopIteration:
                arch_val_iter = iter(arch_val_loader)
                x_val, y_val = next(arch_val_iter)
            x_val, y_val = x_val.to(device), y_val.to(device)

            a_loss = architect.step(x_val, y_val, criterion)

            w_optimizer.zero_grad()
            logits = model(x_train)
            w_loss = criterion(logits, y_train)
            w_loss.backward()
            w_optimizer.step()

            epoch_w_loss += w_loss.item()
            epoch_a_loss += a_loss
            n_batches += 1

        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, arch_val_loader, device)

        print(f'Epoch {epoch+1}/{epochs} | '
              f'w_loss: {epoch_w_loss/n_batches:.4f} | '
              f'a_loss: {epoch_a_loss/n_batches:.4f} | '
              f'train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f}')

        history['w_loss'].append(epoch_w_loss / n_batches)
        history['a_loss'].append(epoch_a_loss / n_batches)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

    test_acc = evaluate(model, test_loader, device)
    print(f'Final held-out test accuracy: {test_acc:.4f}')

    plot_curves(history)


if __name__ == '__main__':
    main()