import numpy as np
from scipy.sparse import issparse


def shd(g1, g2):
    """
    Compute the structural Hamming Distance between two graphs, i.e. the minimum number of edge moves (adds, removals
    and reversals) needed to transform g1 into g2.

    Parameters
    ----------
    g1: scipy.sparse.sspmatrix or np.ndarray
        The first graph.
    g2: scipy.sparse.sspmatrix or np.ndarray
    The second graph.

    Returns
    -------
    out: int
    The SHD between g1 and g2

    Notes
    -----
    Adapted from the R code written by Markus Kalisch, Date:  1 Dec 2006, 17:21.

    """
    m1 = g1.A if issparse(g1) else g1
    m2 = g2.A if issparse(g2) else g2

    s1 = m1 + m1.T
    s2 = m2 + m2.T

    s1[s1 == 2] = 1
    s2[s2 == 2] = 1

    ds = s1 - s2

    r, c = np.nonzero(ds > 0)
    m1[r, c] = 0

    d = np.abs(m1 - m2)

    return len(r) + np.sum((d + d.T) > 0) / 2
