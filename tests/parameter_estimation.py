import numpy as np
from numpy.random import RandomState
from numpy.linalg import norm

from structure.graph_generation import random_dag
from core.gaussian import sample_from_gn, to_mvn, gn_params_mle

seeds = list(range(101, 200))
rng = RandomState()
variables = list(range(5))

mean = np.zeros(len(variables))
sigma = np.zeros(len(variables)) + 0.2
weight = 2

total_mean = 0
total_cov = 0
total_beta = 0

for i, s in enumerate(seeds):
    print('Test {0}/{1}'.format(i + 1, len(seeds)))
    graph = random_dag(variables, rng=rng)

    beta = graph.A.T * weight

    mvn = to_mvn(mean, sigma, beta, graph, return_mvn=True)

    n_samples = 200
    sample_seed = rng.randint(0, 2**32-1)

    data_gn = sample_from_gn(graph, mean, sigma, beta, n_samples, sample_seed)

    mu, var, b = gn_params_mle(graph, data_gn)

    mean_norm = norm(mean - mu)
    cov_norm = norm(sigma - var)
    beta_norm = norm(beta - b)

    total_mean += mean_norm
    total_cov += cov_norm
    total_beta += beta_norm

    print(' Distance between means: {0}\n Distance between variances: {1}\n Distance between {2}'.
          format(mean_norm, cov_norm, beta_norm))

    # print('Network:\n mean: {0}\n cov: {1}'.format(mean_gn, cov_gn))
    # print('MVN:\n mean: {0}\n cov: {1}'.format(mean_mvn, cov_mvn))
    #
    # print('Difference:\n mean {0}\n cov{1}'.format(mean_gn - mean_mvn, cov_gn - cov_mvn))

print('Finished: \n Mean norm: {0}\n Cov Norm: {1}\n Beta Norm: {2}'.
      format(total_mean / len(seeds), total_cov / len(seeds), total_beta / len(seeds)))
