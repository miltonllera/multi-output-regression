import numpy as np
import scipy.sparse as ssp
from itertools import chain

from structure.graphs import DiGraph
from core.misc import get_rng


def random_dag(variables, max_fan_in=-1, rng=None):
    if max_fan_in == -1:
        max_fan_in = len(variables) - 1

    rng = get_rng(rng)
    d = len(variables)

    # g = ssp.lil_matrix((d, d))
    g = DiGraph((d, d), dtype=bool)

    rng.shuffle(variables)

    for i, node in enumerate(variables[1:]):
        i += 1
        n_parents = rng.choice(min(max_fan_in, i) + 1)
        parents = rng.choice(variables[:i], size=n_parents, replace=False)
        if len(parents):
            g.add_edges([(p, node) for p in parents])

    return g
