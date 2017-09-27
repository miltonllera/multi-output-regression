import numpy as np
import scipy.sparse as ssp
from itertools import chain

from structure.graphs import DiGraph, MBCGraph
from core.misc import get_rng


def random_dag(variables, fan_in=-1, rng=None):
    if fan_in == -1:
        fan_in = len(variables) - 1

    rng = get_rng(rng)
    d = len(variables)

    # g = ssp.lil_matrix((d, d))
    g = DiGraph((d, d), dtype=bool)

    rng.shuffle(variables)

    for i, node in enumerate(variables[1:]):
        i += 1
        n_parents = rng.choice(min(fan_in, i) + 1)
        parents = rng.choice(variables[:i], size=n_parents, replace=False)
        if len(parents):
            g.add_edges([(p, node) for p in parents])

    return g


def random_mbc(n_features, n_targets, fan_in=-1, rng=None):
    rng = get_rng(rng)
    n_vars = n_features + n_targets

    if fan_in is None:
        fan_in = n_vars - 1

    g = rng.randint(0, 2, size=(n_vars, n_vars))

    for i in range(n_vars):
        g[i, i:] = 0

    g = g.T

    idx = list(range(n_features + n_targets))
    rng.shuffle(idx)
    g = g[np.ix_(idx, idx)]

    g[np.ix_(range(n_features), np.arange(n_targets) + n_features)] = 0

    for j in range(n_vars):
        n_edges = g[:, j].sum()
        if n_edges > fan_in:
            to_remove = rng.choice(g[:, j].nonzero()[0], n_edges - fan_in, replace=False)
            g[to_remove, j] = 0

    return MBCGraph(g, n_features)
