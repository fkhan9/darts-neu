"""
Discretize the search result into a fixed architecture, retrain it from
scratch (no bilevel split, just ordinary supervised training on the full
training set) and evaluate on the held-out test set.

Usage:
    python train_final.py --name seed1                       # retrains runs/seed1/search/genotype.txt
    python train_final.py --name other --genotype_file path/to/genotype.txt
Settings come from configs/default.yaml (retrain section); override with --set.

The last-epoch model is evaluated; nothing is selected on the test set.
"""

import sys
sys.path.insert(0, 'src')

import argparse
import csv
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn

from model_final import NetworkFinal
from genotypes import Genotype
from dataset import get_retrain_loaders, IMG_SIZE
from config import load_config, save_snapshot
from runtime import (set_seed, phase_dir, get_rng_state, set_rng_state,
                     save_atomic, write_json, rewrite_jsonl)
from evalutil import predict
from metrics import classification_report, softmax
from arch_stats import genotype_to_dict, cell_stats
from plotting import plot_retrain


def format_duration(seconds):
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h >= 1:
        return f'{int(h)}h {int(m)}m {s:.1f}s'
    if m >= 1:
        return f'{int(m)}m {s:.1f}s'
    return f'{s:.1f}s'


def count_flops(model, device):
    """Forward FLOPs for one 96x96 image (torch FlopCounterMode: counts convolutions and linear layers)."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        model.eval()
        x = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(x)
        model.train()
        return int(fc.get_total_flops())
    except Exception as e:  # keep the run going; the field is just empty
        print('FLOP counting failed:', e)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--name', required=True, help='run folder: <out_root>/<name>/retrain/')
    parser.add_argument('--genotype_file', default=None,
                        help='default: <out_root>/<name>/search/genotype.txt')
    parser.add_argument('--seed', type=int, default=None, help='overrides run.seed')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='override any config key, e.g. --set retrain.epochs=30 (repeatable)')
    cli = parser.parse_args()

    overrides = list(cli.set)
    if cli.seed is not None:
        overrides.append(f'run.seed={cli.seed}')
    cfg = load_config(cli.config, overrides)

    out_dir = phase_dir(cfg, cli.name, 'retrain')
    save_snapshot(cfg, out_dir)
    print(f'Run folder: {out_dir}')

    metrics_path = os.path.join(out_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        print('This retrain has already finished (metrics.json exists). Nothing to do.')
        return

    set_seed(cfg.run.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    geno_file = cli.genotype_file or os.path.join(cfg.run.out_root, cli.name, 'search', 'genotype.txt')
    with open(geno_file, 'r') as f:
        genotype_str = f.read().strip()
    genotype = eval(genotype_str, {'Genotype': Genotype, 'range': range})
    shutil.copyfile(geno_file, os.path.join(out_dir, 'genotype_used.txt'))
    print('Loaded genotype:', genotype)

    train_loader, test_loader, classes, test_set, data_info = get_retrain_loaders(
        cfg.data.root, batch_size=cfg.retrain.batch_size, num_workers=cfg.data.num_workers)
    print('Classes:', classes)
    print('Class counts:', data_info)

    model = NetworkFinal(C=cfg.retrain.C, num_classes=len(classes),
                         layers=cfg.retrain.layers, genotype=genotype).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops = count_flops(model, device)
    print(f'Final model parameters: {n_params:,} | forward FLOPs per image: {flops}')

    epochs = cfg.retrain.epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.retrain.lr, momentum=cfg.retrain.momentum,
                                weight_decay=cfg.retrain.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    rows, start_epoch = [], 0
    ckpt_path = os.path.join(out_dir, 'ckpt_last.pt')
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['opt'])
        scheduler.load_state_dict(ck['sched'])
        rows, start_epoch = ck['rows'], ck['epoch']
        set_rng_state(ck['rng'])
        print(f'Resuming from checkpoint after epoch {start_epoch}')

    log_path = os.path.join(out_dir, 'epochs.jsonl')
    rewrite_jsonl(rows, log_path)

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]['lr']
        loss_sum, n_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1
        scheduler.step()

        tr_logits, tr_labels = predict(model, train_loader, device)
        te_logits, te_labels = predict(model, test_loader, device)  # monitor only, never used to select
        row = {
            'epoch': epoch + 1, 'lr': lr, 'loss': loss_sum / n_batches,
            'train_acc': float((tr_logits.argmax(1) == tr_labels).mean()),
            'test_acc': float((te_logits.argmax(1) == te_labels).mean()),
            'epoch_time_s': time.time() - t0,
        }
        rows.append(row)
        with open(log_path, 'a') as f:
            f.write(json.dumps(row) + '\n')
        print(f'Epoch {epoch+1}/{epochs} | loss: {row["loss"]:.4f} | train_acc: {row["train_acc"]:.4f} | '
              f'test_acc: {row["test_acc"]:.4f} | epoch_time: {format_duration(row["epoch_time_s"])} | '
              f'elapsed: {format_duration(sum(r["epoch_time_s"] for r in rows))}')

        if (epoch + 1) % cfg.checkpoint.every == 0 or epoch + 1 == epochs:
            save_atomic({'model': model.state_dict(), 'opt': optimizer.state_dict(),
                         'sched': scheduler.state_dict(), 'rows': rows, 'epoch': epoch + 1,
                         'rng': get_rng_state()}, ckpt_path)

    # ---- final evaluation of the last-epoch model on the held-out test set
    logits, labels = predict(model, test_loader, device)
    rep = classification_report(logits, labels, classes)
    probs = softmax(logits.astype(np.float64))

    with open(os.path.join(out_dir, 'per_image.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'true', 'pred', 'correct'] + [f'p_{c}' for c in classes])
        for (path, _), t, p, pr in zip(test_set.samples, labels, probs.argmax(1), probs):
            w.writerow([path, classes[t], classes[p], int(t == p)] + [f'{v:.6f}' for v in pr])

    retrain_time = sum(r['epoch_time_s'] for r in rows)
    search_summary = os.path.join(cfg.run.out_root, cli.name, 'search', 'summary.json')
    search_time = None
    if os.path.exists(search_summary) and cli.genotype_file is None:
        with open(search_summary) as f:
            search_time = json.load(f)['search_time_s']

    gd = genotype_to_dict(genotype)
    rep.update({
        'name': cli.name, 'seed': cfg.run.seed, 'genotype_file': geno_file,
        'genotype': gd, 'normal': cell_stats(gd['normal']), 'reduce': cell_stats(gd['reduce']),
        'parameters': n_params, 'flops_forward_1img': flops,
        'retrain_epochs': epochs, 'retrain_time_s': retrain_time, 'search_time_s': search_time,
        'data_info': data_info,
    })
    write_json(rep, metrics_path)
    plot_retrain(rows, os.path.join(out_dir, 'retrain_curves.png'))
    torch.save(model.state_dict(), os.path.join(out_dir, 'final_model.pt'))

    print(f'\nTest accuracy {rep["accuracy"]:.4f} ({rep["errors"]} errors of {rep["n"]}) | '
          f'macro-F1 {rep["macro_f1"]:.4f} | NLL {rep["nll"]:.4f}')
    print('Per-class F1:', {c: round(v, 4) for c, v in zip(classes, rep['f1'])})
    print(f'Parameters {n_params:,} | retrain time {format_duration(retrain_time)}')
    print(f'Saved results in {out_dir}')


if __name__ == '__main__':
    main()
