"""
Architecture statistics for DP1 (parameter-free operations). Pure Python/NumPy,
no torch, so the offline scripts can use it on a machine without a GPU stack.

Parameter-free here means skip_connect, avg_pool_3x3 and max_pool_3x3. The
'none' operation is not counted: it never appears in a discretized cell.
"""

from collections import Counter

PRIMITIVES = ['none', 'avg_pool_3x3', 'max_pool_3x3', 'skip_connect',
              'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5']
PARAM_FREE = ('avg_pool_3x3', 'max_pool_3x3', 'skip_connect')
POOLS = ('avg_pool_3x3', 'max_pool_3x3')


def genotype_to_dict(g):
    """Genotype namedtuple -> JSON-friendly dict (concat is implied by the search space)."""
    return {'normal': [[op, int(i)] for op, i in g.normal],
            'reduce': [[op, int(i)] for op, i in g.reduce]}


def cell_stats(gene):
    """gene: list of [op, input_index] for one discretized cell (8 entries)."""
    counts = Counter(op for op, _ in gene)
    n = len(gene)
    pf = sum(counts[o] for o in PARAM_FREE)
    return {
        'counts': dict(counts),
        'n_ops': n,
        'skip_count': counts['skip_connect'],
        'pool_count': sum(counts[o] for o in POOLS),
        'param_free_count': pf,
        'param_free_share': pf / n if n else 0.0,
    }


def mass_stats(softmax_w):
    """
    softmax_w: (n_edges, 8) array of softmax(alpha). Mean over edges of the
    weight each op group receives in the supernet (sums to 1 across groups).
    """
    import numpy as np
    W = np.asarray(softmax_w)
    m = W.mean(axis=0)
    idx = {p: i for i, p in enumerate(PRIMITIVES)}
    return {
        'none': float(m[idx['none']]),
        'skip': float(m[idx['skip_connect']]),
        'pool': float(m[idx['avg_pool_3x3']] + m[idx['max_pool_3x3']]),
        'conv': float(sum(m[idx[p]] for p in PRIMITIVES if 'conv' in p)),
        'param_free': float(sum(m[idx[p]] for p in PARAM_FREE)),
    }


def top_op_per_edge(softmax_w):
    """Strongest non-'none' op on each edge, as op names."""
    import numpy as np
    W = np.asarray(softmax_w)
    out = []
    for row in W:
        best = max((k for k in range(len(PRIMITIVES)) if PRIMITIVES[k] != 'none'), key=lambda k: row[k])
        out.append(PRIMITIVES[best])
    return out
