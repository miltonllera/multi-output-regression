import numpy as np
import scipy.linalg as linalg
from scipy import stats
from scipy.sparse import spmatrix, csr_matrix

from core.misc import get_rng
from structure.graphs import DiGraph, topsort


# noinspection PyTupleAssignmentBalance
def gn_params(network: DiGraph, data, sparse=False, l1_reg=0., l2_reg=0.):
    """
    Compute the MLE of the parameters for each Gaussian factor in the network. The MLE of a Gaussian Network given some
    data vector D is characterized by the unconditional mean, conditional variance and weights of the influence of the
    parents of a node when computing the conditional mean given some data.

    Parameters
    ----------
    network: DiGraph
        The graph where nonzero entries represent the edges with rows representing the tails and columns the heads.

    data: numpy.ndarray
        The data used to compute the MLE of shape (n, d) where n is the number of samples and d is the number of nodes.

    sparse: bool (default False)
        If true will return the weights as a sparse matrix instead of an numpy ndarray. Useful when the graph is sparse.

    l1_reg: float
        The strength of the penalty term used in L1 (LASSO) regularization.

    l2_reg: float
        The strength of the penalty term used in L2 (Ridge Regression) regularization.

    Returns
    -------
    (mean, var, beta): 3-tuple
        The computed parameters of shapes (d,), (d,) and (d, d) respectively with the value of the interaction weights.

    Notes
    -----
    Uses scipy.linalg.lstsq to find the Least Squares Solution to each conditional regression. This solves as:

        1 - Compute the QR decomposition of X: X = Q * R, where Q^T * Q = I and R is upper triangular
        2 - Solve X * w = y <=>  R * w = Q^T * y

    This method is numerically stable and runs in O(N * D^2)

    """
    if l1_reg != 0:
        raise NotImplementedError('L1 regularization not implemented')

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
            X, y = data[:, ps], data[:, node]

            if l2_reg != 0:
                X = np.vstack((X, np.sqrt(l2_reg) * np.identity(len(ps))))
                y = np.hstack((y, np.zeros(len(ps))))

            # # Solving for the coefficients
            w = linalg.lstsq(X, y)[0]

            beta[node, ps] = w
            var[node] -= np.dot(np.dot(w, cov[np.ix_(ps, ps)]), w)

    if sparse:
        beta = csr_matrix(beta)

    return sample_mean, var, beta


def gn_params_mle(network: DiGraph, data, sparse=False):
    return gn_params(network, data, sparse)


def gn_params_ridge(network: DiGraph, data, sparse=False, l2_reg=0.1):
    return gn_params(network, data, sparse, l2_reg=l2_reg)


def mvn_params(data):
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


def to_mvn(mean, var, beta, return_mvn=False, rng=None):
    """
    Transform the parameters of a Gaussian Network into the parameters of the corresponding Multivariate Normal
    distribution using the formulas in Schacter and Kenley (1989), Appendix B (can also be found in Murphy (2009)).

    Parameters
    ----------
    mean: numpy.ndarray
        The unconditional mean vector of size (d,). It is the same as the mean of a MVN

    var: numpy.ndarray
        The variance of each variable conditioned on its parents in the network

    beta: numpy.ndarray or scipy.sparse.spmatrix
        The values of the coefficients used when calculting the conditional distributions of each variable

    return_mvn: bool
        See below in returned values

    rng: numpy.random.RandomState
        If return_mvn is true this is the random number generator used in the frozen multivariate normal instance.

    Returns
    -------
    (mean, sigma): 2-tuple
        The corresponding mean (it's the same as in the input) and covariance matrix of shapes (d,) and (d, d)
        respectively

    mvn: scipy.stats.multivariate_normal
        If return_mvn is true will return a frozen scipy.stats.multivariate_normal instance with the corresponding
         mean and covariance matrix.

    """
    top_sort = topsort(beta.T)

    if isinstance(beta, spmatrix):
        beta = beta.A

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
    """
    Obtain samples from a Gaussian Network by using the conditional probabilities it encodes.

    graph: DigGraph
        The adjacency matrix of the graph. If graph.dtype is float then the adjacency matrix also encodes the weights
         of the arcs and the parameter beta will be ignored

    mean: numpy.ndarray


    """
    if isinstance(mean, int):
        mean = mean + np.zeros(graph.n_nodes)
    if graph.dtype == float or beta is None:
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
    """
    Compute the posterior parameters of a Normal-Wishart distribution. This distribution serves as joint distribution
    over means and covariances

    Parameters
    ----------
    data: numpy.ndarray
        A (N, D) shaped numpy.ndarray with the sample data to use for the update.

    mu0: numpy.ndarray
        The numpy.ndarray of size (D,) with the prior mean

    T0: numpy.ndarray
        The numpy.ndarray of size (D, D) with the prior scatter matrix

    k: int
        The equivalent sample size which controls the strength of the prior mean

    v: int
        The degrees of freedom which controls the strength of prior scatter matrix

    Returns
    -------
    (mu_n, Sn, kn, vn): 4-tuple
        The updated values after seeing the N instances in data

    """
    n, d = data.shape

    sample_mean = np.mean(data, axis=0)
    S = np.dot(data.T, data)

    kn, vn = k + n, v + n
    mu_n = (k * mu0 + n * sample_mean) / kn

    Sn = T0 + S + k * np.dot(mu0.reshape(-1, 1), mu0.reshape(1, -1)) - \
        kn * np.dot(mu_n.reshape(-1, 1), mu_n.reshape(1, -1))

    return mu_n, Sn, kn, vn


def fit_mvn(data, rng=None):
    """
    Return the MVN whose parameters maximize the likelihood of the data as a Scipy frozen multivariate normal rv.

    Parameters
    ----------
    data: numpy.ndarray
        An array of shape (N, D) wih the sample data used to estimate the MVN

    rng: numpy.random.RandomState
        The random generator used to initialize the random variable

    Returns
    -------
    mvn: Scipy.stats.multivariate_normal
        The expected MVN

    """
    mean, cov = mvn_params(data)
    return stats.multivariate_normal(mean=mean, cov=cov, seed=rng)


def conditional_mvn(mvn, x):
    """
    Return the conditional MVN whose parameters maximize the likelihood of the data as a Scipy frozen
    multivariate normal conditioned on some observations.

    Parameters
    ----------
    mvn: Scipy.stats.multivariate_normal
        The joint MVN normal before observing the variables

    x: numpy.ndarray
        The conditioning values. The corresponding dimensions are assumed to be the the x.shape[1] first ones of
        the mean and the top left sub-matrix of size (x.shape[1], x.shape[1]).

    Returns
    -------
    mvn: Scipy.stats.multivariate_normal
        The expected MVN

    """
    mean, cov = conditional_mvn_params(mvn.mean, mvn.cov, x, return_cov=True)
    return stats.multivariate_normal(mean=mean, cov=cov, seed=mvn.random_state)
