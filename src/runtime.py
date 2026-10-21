"""Seeding and run-folder helpers shared by the search and retrain scripts."""

import os
import random


def set_seed(seed):
    """
    Seed Python, NumPy and PyTorch (CPU and all GPUs). GPU nondeterminism is
    deliberately left on (no cudnn.deterministic), so two runs with the same
    seed can still differ slightly; the five seeds are what carry the variation.
    """
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def phase_dir(cfg, name, phase):
    """runs/<name>/<phase>/ (phase is 'search' or 'retrain'); created if missing."""
    d = os.path.join(cfg.run.out_root, name, phase)
    os.makedirs(d, exist_ok=True)
    return d


def get_rng_state():
    import numpy as np
    import torch
    return {
        'py': random.getstate(),
        'np': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state):
    import numpy as np
    import torch
    random.setstate(state['py'])
    np.random.set_state(state['np'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def save_atomic(obj, path):
    """Write to a temp file and rename, so a crash mid-save never leaves a broken checkpoint."""
    import torch
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(obj, path):
    import json
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def rewrite_jsonl(rows, path):
    import json
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    os.replace(tmp, path)
