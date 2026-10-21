import sys
sys.path.insert(0, 'src')

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from model_search import Network
from architect import Architect
from dataset import get_neu_cls_loaders
from config import load_config, save_snapshot
from runtime import (set_seed, phase_dir, get_rng_state, set_rng_state,
                     save_atomic, write_json, rewrite_jsonl)
from evalutil import predict
from metrics import classification_report, softmax
from arch_stats import genotype_to_dict, cell_stats, mass_stats
from plotting import plot_search


def format_duration(seconds):
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h >= 1:
        return f'{int(h)}h {int(m)}m {s:.1f}s'
    if m >= 1:
        return f'{int(m)}m {s:.1f}s'
    return f'{s:.1f}s'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--name', required=True, help='run folder: <out_root>/<name>/search/')
    parser.add_argument('--seed', type=int, default=None, help='overrides run.seed')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='override any config key, e.g. --set search.epochs=80 (repeatable)')
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = list(args.set)
    if args.seed is not None:
        overrides.append(f'run.seed={args.seed}')
    cfg = load_config(args.config, overrides)

    out_dir = phase_dir(cfg, args.name, 'search')
    save_snapshot(cfg, out_dir)
    print(f'Run folder: {out_dir}')

    summary_path = os.path.join(out_dir, 'summary.json')
    if os.path.exists(summary_path):
        print('This search has already finished (summary.json exists). Nothing to do.')
        return

    set_seed(cfg.run.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    total_epochs = cfg.search.epochs
    train_loader, arch_val_loader, test_loader, classes, data_info = get_neu_cls_loaders(
        cfg.data.root, batch_size=cfg.data.batch_size,
        val_split_for_arch=cfg.data.arch_val_split,
        num_workers=cfg.data.num_workers,
        split_seed=cfg.data.split_seed,
    )
    print('Classes:', classes)
    print('Class counts:', data_info)

    model = Network(C=cfg.model.C, num_classes=len(classes), layers=cfg.model.layers).to(device)
    criterion = nn.CrossEntropyLoss()

    # Weight optimizer: network weights only. The alphas are updated solely by the
    # architect's own Adam optimizer (see Architect).
    w_optimizer = torch.optim.SGD(model.weight_parameters(), lr=cfg.search.w_lr,
                                  momentum=cfg.search.w_momentum,
                                  weight_decay=cfg.search.w_weight_decay)
    architect = Architect(model,
                          arch_lr=cfg.search.arch_lr,
                          arch_weight_decay=cfg.search.arch_weight_decay,
                          arch_betas=cfg.search.arch_betas,
                          unrolled_momentum=cfg.search.w_momentum,
                          unrolled_weight_decay=cfg.search.w_weight_decay)

    w_ids = {id(p) for g in w_optimizer.param_groups for p in g['params']}
    assert not any(id(a) in w_ids for a in model.arch_parameters()), 'alphas must not be in the weight optimizer'
    assert len(w_ids) == len(model.weight_parameters()) and len(w_ids) + 2 == len(list(model.parameters()))

    write_json({'classes': classes, 'data_info': data_info,
                'supernet_parameters': sum(p.numel() for p in model.parameters())},
               os.path.join(out_dir, 'meta.json'))

    alpha_init = [a.detach().cpu().clone() for a in model.arch_parameters()]
    rows, start_epoch = [], 0

    ckpt_path = os.path.join(out_dir, 'ckpt_last.pt')
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        w_optimizer.load_state_dict(ck['w_opt'])
        architect.optimizer.load_state_dict(ck['a_opt'])
        rows, start_epoch = ck['rows'], ck['epoch']
        alpha_init = ck['alpha_init']
        set_rng_state(ck['rng'])
        print(f'Resuming from checkpoint after epoch {start_epoch}')

    log_path = os.path.join(out_dir, 'epochs.jsonl')
    rewrite_jsonl(rows, log_path)

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        arch_val_iter = iter(arch_val_loader)
        w_loss_sum, a_loss_sum, n_batches, n_arch = 0.0, 0.0, 0, 0

        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)

            try:
                x_val, y_val = next(arch_val_iter)
            except StopIteration:
                arch_val_iter = iter(arch_val_loader)
                x_val, y_val = next(arch_val_iter)
            x_val, y_val = x_val.to(device), y_val.to(device)
            eta = w_optimizer.param_groups[0]['lr']
            a_loss_sum += architect.step(x_train, y_train, x_val, y_val, eta, w_optimizer, criterion)
            n_arch += 1

            w_optimizer.zero_grad()
            w_loss = criterion(model(x_train), y_train)
            w_loss.backward()
            w_optimizer.step()

            w_loss_sum += w_loss.item()
            n_batches += 1

        tr_logits, tr_labels = predict(model, train_loader, device)
        train_acc = float((tr_logits.argmax(1) == tr_labels).mean())
        val = classification_report(*predict(model, arch_val_loader, device), classes)

        an, ar = [a.detach().cpu().numpy() for a in model.arch_parameters()]
        sn, sr = softmax(an), softmax(ar)
        gd = genotype_to_dict(model.genotype())
        normal, reduce_ = cell_stats(gd['normal']), cell_stats(gd['reduce'])
        change = max(float(np.abs(an - alpha_init[0].cpu().numpy()).max()),
                     float(np.abs(ar - alpha_init[1].cpu().numpy()).max()))

        row = {
            'epoch': epoch + 1,
            'arch_updates': n_arch,
            'w_loss': w_loss_sum / n_batches,
            'a_loss': (a_loss_sum / n_arch) if n_arch else None,
            'train_acc': train_acc,
            'val_acc': val['accuracy'],
            'val_nll': val['nll'],
            'val_macro_f1': val['macro_f1'],
            'val_per_class_acc': val['recall'],
            'alpha_max_abs_change': change,
            'alphas_normal': an.tolist(), 'alphas_reduce': ar.tolist(),
            'softmax_normal': sn.tolist(), 'softmax_reduce': sr.tolist(),
            'genotype': gd,
            'normal': normal, 'reduce': reduce_,
            'mass_normal': mass_stats(sn), 'mass_reduce': mass_stats(sr),
        }
        row['epoch_time_s'] = time.time() - t0
        rows.append(row)
        with open(log_path, 'a') as f:
            f.write(json.dumps(row) + '\n')

        elapsed = sum(r['epoch_time_s'] for r in rows)
        print(f'Epoch {epoch+1}/{total_epochs} | w_loss: {row["w_loss"]:.4f} | '
              f'a_loss: {"-" if row["a_loss"] is None else format(row["a_loss"], ".4f")} | '
              f'train_acc: {train_acc:.4f} | val_acc: {val["accuracy"]:.4f} | '
              f'normal param-free: {normal["param_free_count"]}/8 (skip {normal["skip_count"]}) | '
              f'alpha move: {change:.3f} | epoch_time: {format_duration(row["epoch_time_s"])} | '
              f'elapsed: {format_duration(elapsed)}')

        if (epoch + 1) % cfg.checkpoint.every == 0 or epoch + 1 == total_epochs:
            save_atomic({
                'model': model.state_dict(), 'w_opt': w_optimizer.state_dict(),
                'a_opt': architect.optimizer.state_dict(), 'rows': rows, 'epoch': epoch + 1,
                'alpha_init': alpha_init,
                'rng': get_rng_state(),
            }, ckpt_path)

    t_logits, t_labels = predict(model, test_loader, device)
    supernet_test_acc = float((t_logits.argmax(1) == t_labels).mean())
    search_time = sum(r['epoch_time_s'] for r in rows)
    epochs_run = len(rows)

    final_genotype = model.genotype()
    with open(os.path.join(out_dir, 'genotype.txt'), 'w') as f:
        print(final_genotype, file=f)

    plot_search(rows, os.path.join(out_dir, 'search_curves.png'), classes=classes)

    last = rows[-1]
    write_json({
        'name': args.name, 'seed': cfg.run.seed,
        'epochs_planned': total_epochs, 'epochs_run': epochs_run,
        'search_time_s': search_time, 'mean_epoch_time_s': search_time / epochs_run,
        'supernet_test_acc': supernet_test_acc,
        'final_genotype': last['genotype'],
        'final_normal': last['normal'], 'final_reduce': last['reduce'],
        'final_alpha_max_abs_change': last['alpha_max_abs_change'],
        'final_val_acc': last['val_acc'],
        'classes': classes, 'data_info': data_info,
    }, summary_path)

    print(f'Supernet held-out test accuracy: {supernet_test_acc:.4f} (supernet, not the final model)')
    print(f'Total search time: {format_duration(search_time)} over {epochs_run} epochs')
    print(f'Saved genotype and logs in {out_dir}')


if __name__ == '__main__':
    main()
