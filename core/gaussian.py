import numpy as np
import scipy.linalg as linalg
from scipy import stats

from core.misc import get_rng
from structure.graphs import DiGraph, topsort


def gn_params_mle(network: DiGraph, data):
    """
    Compute the MLE of the parameters for each Gaussian factor in the network. The MLE of a Gaussian Network given some
    data vector D is characterized by the unconditional mean, conditional variance and weights of the influence of the
    parents of a node when computing the conditional mean given some data.



    """
    n, d = data.shape

    # Compute sufficient statistics
    sample_mean = data.mean(axis=0)

    data -= sample_mean
    cov = np.dot(data.T, data) / (n - 1)

    # The parameter matrices
    beta = np.zeros((d, d))

    # Initial variance is the diagonal of the covariance
    var = np.diagonal(cov).copy()

    for node in network.nodes_iter():
        ps = network.parents(node)

        if len(ps):
            sub_idx = np.ix_(ps, ps)

            a = cov[sub_idx]
            b = cov[node, ps]

            # Solving for the coefficients
            beta[node, ps] = linalg.solve(a, b, assume_a='sym')

            beta_node = beta[node, ps]
            var[node] -= np.dot(np.dot(beta_node, cov[sub_idx]), beta_node)

    return sample_mean, var, beta


def mvn_params_mle(data):
    """
    Return de MLE estimate of a Multivariate Gaussian Distribution.

    Parameters
    ----------
    data: numpy.ndarray of size (N, D)

    Returns
    --------
    (mean, cov): 2-tuple
        The MLE mean and covariance obtained by deriving the log-likelihood.
         The parameters have shapes (D,) and (D, D) respectively

    """
    return np.mean(data, axis=0), np.cov(data, rowvar=False)


def conditional_mvn_params(mean, sigma, given_values, return_cov=False):
    """
    Parameters
    ----------
    mean: numpy.ndarray
        The gen_mean of the MVN as a 1-D array.

    sigma: numpy.ndarray
        A 2-D covariance matrix for the MVN of the form:

            sigma = [sigma_11  sigma_12]
                    [sigma_21  sigma_22]

        where the sigmas are the submatrices indexed by the subscripts corresponding
        to the predictors (subscript 1) and targets (subscript 2) by row and column respectively.

    given_values: numpy.ndarray
        The value of the predictor variables. The corresponding means and covariances are the ones in the first
        give_values.shape[0] entries of gen_mean and cov.

    return_cov: bool (default True)
        If false will only compute the conditional gen_mean. Useful if we only wish to predict without
        estimating the variance (or confidence of the model).

    Returns
    -------
    mean: numpyp.ndarray
        The conditional mean of the targets given the inputs

    (mean, cov): tuple 2d
        The conditional gen_mean and covariance of the MVN
    """
    n_features = given_values.shape[0]

    k = sigma[:n_features][:, :n_features]
    k_prime = sigma[:n_features][:, n_features:]

    f = given_values - mean[:n_features]

    l = linalg.cholesky(k + 1e-6 * np.eye(k.shape[0]), lower=True)

    m = linalg.solve_triangular(l, f, lower=True)
    alpha = linalg.solve_triangular(l.T, m)

    cond_mean = mean[n_features:] + np.dot(k_prime.T, alpha)

    if not return_cov:
        return cond_mean

    k_star = sigma[n_features:][:, n_features:]
    if k_star.shape == (1, 1):
        k_star = k_star[0, 0]

    v = linalg.solve_triangular(l, k_prime, lower=True)

    cond_cov = k_star - np.dot(v.T, v)

    return cond_mean, cond_cov


def to_mvn(mean, var, beta, structure, return_mvn=False, rng=None):
    top_sort = topsort(structure)
    inv_top_sort = np.argsort(top_sort)

    I, W = np.eye(beta.shape[0]), beta[np.ix_(top_sort, top_sort)]

    # Use solve triangular to compute the inverse in a numerically stable way
    # by specifying unit_diagonal=True we don't need to pass I - W and can pass just -W
    U = linalg.solve_triangular(-W, I, lower=True, unit_diagonal=True, overwrite_b=True)

    # Compute Sigma as U * S * U^T as found in Schacter and Kenley
    sigma = np.dot(U, np.dot(np.diag(var[top_sort]), U.T))
    sigma = sigma[np.ix_(inv_top_sort, inv_top_sort)]

    if return_mvn:
        return stats.multivariate_normal(mean=mean, cov=sigma, seed=rng)

    return mean, sigma


def sample_from_gn(graph: DiGraph, mean, sigma, beta=None, size=1, rng=None):
    if isinstance(mean, int):
        mean = mean + np.zeros(graph.n_nodes)
    if beta is None:
        beta = graph.A.T.astype(float)
    if isinstance(beta, (int, float)):
        beta = beta * graph
    if isinstance(sigma, int):
        sigma = sigma + np.zeros(mean.shape)

    rng = get_rng(rng)

    samples = []

    top_sort = topsort(graph)

    for _ in range(size):
        new_sample = np.zeros(mean.shape[0])

        for v in top_sort:
            miu = mean[v]
            for u in graph.parents(v):
                miu += beta[v, u] * (new_sample[u] - mean[u])

            new_sample[v] = rng.normal(miu, np.sqrt(sigma[v]))

        samples.append(new_sample)

    return np.asarray(samples)


def update_normal_wishart_parameters(data, mu0, T0, k, v):
    n, d = data.shape

    sample_mean = np.mean(data, axis=0)
    S = np.dot(data.T, data)

    kn, vn = k + n, v + n
    mu_n = (k * mu0 + n * sample_mean) / kn

    Sn = T0 + S + k * np.dot(mu0.reshape(-1, 1), mu0.reshape(1, -1)) - \
        kn * np.dot(mu_n.reshape(-1, 1), mu_n.reshape(1, -1))

    return mu_n, Sn, kn, vn


def mvn_mle(data, rng=None):
    mean, cov = mvn_params_mle(data)
    return stats.multivariate_normal(mean=mean, cov=cov, seed=rng)


def conditional_mvn(mvn, x):
    mean, cov = conditional_mvn_params(mvn.mean, mvn.cov, x, return_cov=True)
    return stats.multivariate_normal(mean=mean, cov=cov, seed=mvn.random_state)
