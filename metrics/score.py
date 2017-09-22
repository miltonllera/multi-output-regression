import numpy as np
from scipy.linalg import solve_triangular, cholesky
from scipy.special import gammaln

from structure.graphs import DiGraph
from core.misc import det_2by2, logdet_traingular
from core.gaussian import update_normal_wishart_parameters


class BGe:
    """
        The log-score the structure of the distribution of a variable given a set of parents according to the data
        using the BGe metric (equation 17) as found in Learning Gaussian Networks, Heckerman & Geiger, 1994.
        Corrections to the subset calculation as found in Kuipers, 2014 (Equation A.27). Notation as in Murphy, 2009.
        Some implementations details as found in Kuipers & Moffa, 2015.

        Parameters
        ----------
        data: array_like
            Sample data, where columns are variables and rows are sample cases.
        mu0: NDArray
            Vector of shape (n,), the prior means for the distribution.
        t0: NDArray
            Prior precision matrix of shape (n, n)
        k: int
            Equivalent sample size, determines the strength of the prior values.
        v: int
            Degrees of freedom. Must hold that v > d - 1, where n is the dimension of the data.

        Notes
        -----
        The returned instance has a defined __call__ method with the following signature:

        bge_score(structure) -> float

        where structure is a list of pairs (x, p) (x is the child variables and p are its parents and can be None)
        or a networkx.DiGraph, in other words it computes p(D_x| pa(x)) for each variable and sums their log-values.

        This class allows for partial updates to the posterior data through the update_posterior_params method,
        enabling partial fits to data.
        """
    def __init__(self, data, mu0=None, t0=None, k=None, v=None):
        n, d = data.shape

        if mu0 is None:
            mu0 = np.zeros(d)
        elif mu0.shape != (d,):
            raise ValueError('mu0 has shape {0} expected {1]'.format(mu0.shape, (d,)))

        if k is None:
            k = 1
        if k <= 0:
            raise ValueError('The value of k must be strictly greater than 0')

        if v is None:
            v = d + k + 1
        if v < d:
            raise ValueError('The value of dof must be greater than d - 1')

        if t0 is None:
            t0_scale = k * (v - d - 1) / (k + 1)
            t0 = t0_scale * np.eye(d)
        elif t0.shape != (d, d):
            raise ValueError('T0 has shape {0} expected {1]'.format(t0.shape, (d, d)))
        else:
            raise NotImplementedError()

        # Update the parameters of the model
        mu_n, sn, kn, vn = update_normal_wishart_parameters(data, mu0, t0, k, v)

        # Save the log constant and log-gamma terms for efficient reuse. Gamma arguments for subsets as in Press et.al.
        log_const = 1 / 2 * np.log(k / kn) - n/2 * np.log(np.pi)

        vmd = v - d
        dp_values = np.arange(d) + 1
        self.indep_term = log_const + gammaln((dp_values + n + vmd) / 2) - gammaln((dp_values + vmd) / 2) +\
            ((2 * dp_values + vmd - 1) / 2) * np.log(t0_scale)

        self.mu_n, self.sn, self.vn, self.kn = mu_n, sn, vn, kn

    @property
    def dim(self):
        return self.mu_n.shape[0]

    @property
    def params(self):
        return self.mu_n, self.sn, self.kn, self.vn

    def __call__(self, network):
        if isinstance(network, tuple):
            structure = [network]
        elif isinstance(network, DiGraph):
            structure = [(n, network.T[n].nonzero()[1]) for n in network.nodes_iter()]
        else:
            structure = network

        if len(structure) == 1:
            return self._set_score_ratio(*structure[0])

        return np.sum([self._set_score_ratio(x, ps) for x, ps in structure])

    def _set_score_ratio(self, x, parent_set):
        if parent_set is None:
            parent_set = []
        elif len(parent_set):
            parent_set = sorted(parent_set)

        d_p = len(parent_set)
        v_plus_dim = (self.vn - self.dim + d_p + 1) / 2
        a = self.sn[x, x]

        if d_p == 0:
            return self.indep_term[0] - v_plus_dim * np.log(a)

        if d_p == 1:
            sub_sn = self.sn[parent_set[0], parent_set[0]]
            b = self.sn[x, parent_set[0]]

            log_det_p = np.log(sub_sn)
            log_det2 = np.log(a - b ** 2 / sub_sn)

        elif d_p == 2:
            sub_sn = self.sn[np.ix_(parent_set, parent_set)]
            b = self.sn[x, parent_set].reshape(-1, 1)

            log_det_p = np.log(det_2by2(sub_sn))
            log_det2 = np.log(det_2by2(sub_sn - np.dot(b, b.T) / a)) + np.log(a) - log_det_p

        else:
            sub_sn = self.sn[np.ix_(parent_set, parent_set)]
            b = self.sn[x, parent_set]

            l = cholesky(sub_sn, lower=True)
            c = solve_triangular(l, b, lower=True)

            log_det_p = logdet_traingular(l)
            log_det2 = np.log(a - np.sum(c ** 2))

        return self.indep_term[d_p] - v_plus_dim * log_det2 - log_det_p / 2

    def score(self, structure):
        return self(structure)


