"""Plots built only from the saved logs, so they can be redrawn offline for any run."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _nan(v):
    return float('nan') if v is None else v


def plot_search(rows, out_path, classes=None):
    ep = [r['epoch'] for r in rows]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))

    a = ax[0][0]
    a.plot(ep, [r['w_loss'] for r in rows], marker='o', ms=3, label='weight loss')
    a.plot(ep, [_nan(r['a_loss']) for r in rows], marker='o', ms=3, label='architecture loss (arch-val)')
    a.set_title('Loss'); a.set_xlabel('Epoch'); a.legend(); a.grid(alpha=.3)

    a = ax[0][1]
    a.plot(ep, [r['train_acc'] for r in rows], marker='o', ms=3, label='train')
    a.plot(ep, [r['val_acc'] for r in rows], marker='o', ms=3, label='arch-val')
    a.axhline(1 / len(rows[0]['val_per_class_acc']), color='gray', ls='--', label='chance')
    a.set_title('Supernet accuracy'); a.set_xlabel('Epoch'); a.legend(); a.grid(alpha=.3)

    a = ax[0][2]
    a.plot(ep, [r['normal']['param_free_share'] for r in rows], marker='o', ms=3, label='parameter-free share (normal cell)')
    a.plot(ep, [r['normal']['skip_count'] / 8 for r in rows], marker='s', ms=3, label='skip share (normal cell)')
    a.plot(ep, [r['mass_normal']['param_free'] for r in rows], ls='--', label='supernet mass on param-free ops')
    a.set_ylim(0, 1); a.set_title('DP1: parameter-free operations'); a.set_xlabel('Epoch'); a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[1][0]
    for key in ('skip', 'pool', 'conv', 'none'):
        a.plot(ep, [r['mass_normal'][key] for r in rows], label=key)
    a.set_title('Mean softmax weight per op group (normal cell)'); a.set_xlabel('Epoch'); a.legend(); a.grid(alpha=.3)

    a = ax[1][1]
    n_cls = len(rows[0]['val_per_class_acc'])
    for c in range(n_cls):
        a.plot(ep, [r['val_per_class_acc'][c] for r in rows],
               label=(classes[c] if classes else f'class {c}'))
    a.set_title('Per-class arch-val accuracy'); a.set_xlabel('Epoch'); a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[1][2]
    a.plot(ep, [r['alpha_max_abs_change'] for r in rows], marker='o', ms=3)
    a.set_title('Largest alpha change since initialisation'); a.set_xlabel('Epoch'); a.grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_retrain(rows, out_path):
    ep = [r['epoch'] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [r['loss'] for r in rows], marker='o', ms=3)
    ax[0].set_title('Training loss'); ax[0].set_xlabel('Epoch'); ax[0].grid(alpha=.3)
    ax[1].plot(ep, [r['train_acc'] for r in rows], marker='o', ms=3, label='train')
    if rows and rows[0].get('test_acc') is not None:
        ax[1].plot(ep, [_nan(r.get('test_acc')) for r in rows], marker='o', ms=3, label='test (monitor only)')
    ax[1].set_title('Accuracy'); ax[1].set_xlabel('Epoch'); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
