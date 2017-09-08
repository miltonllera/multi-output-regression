import numpy as np
from numpy.random import RandomState
from numpy.linalg import norm

from structure.graph_generation import random_dag
from core.gaussian import sample_from_gn, to_mvn

seeds = list(range(101, 200))
rng = RandomState()
variables = list(range(5))

mean = np.zeros(len(variables))
sigma = np.zeros(len(variables)) + 0.2
weight = 2

total_mean = 0
total_cov = 0

for i, s in enumerate(seeds):
    print('Test {0}/{1}'.format(i + 1, len(seeds)))
    graph = random_dag(variables, rng=rng)

    beta = graph.A.T * weight

    mvn = to_mvn(mean, sigma, beta, graph, return_mvn=True)

    n_samples = 200
    sample_seed = rng.randint(0, 2**32-1)

    data_gn = sample_from_gn(graph, mean, sigma, beta, n_samples, sample_seed)
    data_mvn = mvn.rvs(n_samples, sample_seed)

    mean_gn = np.mean(data_gn, axis=0)
    mean_mvn = np.mean(data_mvn, axis=0)

    cov_gn = np.cov(data_gn, rowvar=False)
    cov_mvn = np.cov(data_mvn, rowvar=False)

    mean_norm = norm(mean_mvn - mean_gn)
    cov_norm = norm(cov_mvn - cov_gn)

    total_mean += mean_norm
    total_cov += cov_norm

    print(' Distance between means: {0}\n Distance between covariances: {1}'.format(mean_norm, cov_norm))

    # print('Network:\n mean: {0}\n cov: {1}'.format(mean_gn, cov_gn))
    # print('MVN:\n mean: {0}\n cov: {1}'.format(mean_mvn, cov_mvn))
    #
    # print('Difference:\n mean {0}\n cov{1}'.format(mean_gn - mean_mvn, cov_gn - cov_mvn))

print('Finished: \n Mean norm: {0}\n Cov Norm: {1}'.format(total_mean / len(seeds), total_cov / len(seeds)))
