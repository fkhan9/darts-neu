"""
Run configuration: one YAML file of defaults, command-line overrides, and a
snapshot saved inside every run folder so a run can always be traced back to
the exact settings that produced it.

No torch import here, so the loader can be tested anywhere.
"""

import copy
import os
import yaml


class Config:
    """Read-only attribute view over a nested dict: cfg.search.epochs."""

    def __init__(self, d):
        object.__setattr__(self, '_d', d)

    def __getattr__(self, name):
        try:
            v = self._d[name]
        except KeyError:
            raise AttributeError(name)
        return Config(v) if isinstance(v, dict) else v

    def __setattr__(self, name, value):
        raise AttributeError('Config is read-only; use --set key=value')

    def to_dict(self):
        return copy.deepcopy(self._d)


def _apply_override(d, dotted, raw):
    keys = dotted.split('.')
    node = d
    for k in keys[:-1]:
        if not isinstance(node, dict) or k not in node:
            raise SystemExit(f'Unknown config key: {dotted}')
        node = node[k]
    last = keys[-1]
    if not isinstance(node, dict) or last not in node:
        raise SystemExit(f'Unknown config key: {dotted}')
    if isinstance(node[last], dict):
        raise SystemExit(f'{dotted} is a section, set one of its keys')
    node[last] = yaml.safe_load(raw)  # "80" -> 80, "true" -> True, "[0.9, 0.99]" -> list


def load_config(path, overrides=()):
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    for item in overrides:
        if '=' not in item:
            raise SystemExit(f'--set expects key=value, got: {item}')
        key, raw = item.split('=', 1)
        _apply_override(d, key.strip(), raw)
    return Config(d)


def diff_keys(a, b, prefix=''):
    """Dotted names of every key whose value differs between two config dicts."""
    out = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        name = f'{prefix}{k}'
        if isinstance(va, dict) and isinstance(vb, dict):
            out += diff_keys(va, vb, name + '.')
        elif va != vb:
            out.append(name)
    return out


def save_snapshot(cfg, phase_dir):
    """
    Write phase_dir/config.yaml. If one already exists it must match, so a
    folder never mixes results from two different settings. (Matching is also
    what later lets an interrupted run be resumed in place.)
    """
    os.makedirs(phase_dir, exist_ok=True)
    path = os.path.join(phase_dir, 'config.yaml')
    new = cfg.to_dict()
    if os.path.exists(path):
        with open(path, 'r') as f:
            old = yaml.safe_load(f)
        changed = diff_keys(old, new)
        if changed:
            raise SystemExit(
                f'{phase_dir} already holds a run with different settings '
                f'({", ".join(changed)}). Use a new --name or delete that folder.'
            )
    else:
        with open(path, 'w') as f:
            yaml.safe_dump(new, f, sort_keys=False)
    return path
