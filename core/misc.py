import numpy as np
from itertools import chain, combinations


def get_rng(seed=None):
    if isinstance(seed, np.random.RandomState):
        return seed
    return np.random.RandomState(seed)


def power_set(iterable, max_size=-1):
    # list(chain.from_iterable(map(set, combinations(iterable, r)) for r in range(min(len(iterable), max_size) + 1)))
    if max_size == -1:
        max_size = len(iterable)

    sets = chain.from_iterable(combinations(iterable, r) for r in range(max_size + 1))
    sets = map(set, sets)
    return list(sets)


def det_2by2(m):
    return m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]


def logdet_traingular(l):
    return 2 * sum(np.log(np.diag(l)))
