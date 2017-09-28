import numpy as np
from numpy.random import RandomState
from numpy.linalg import norm

from structure.graph_generation import random_dag
from core.gaussian import sample_from_gn, to_mvn, gn_params
from sklearn.linear_model import LinearRegression

seeds = list(range(101, 200))
rng = RandomState()
variables = list(range(5))
n_samples = 200

gen_mean = np.zeros(len(variables))
gen_var = np.zeros(len(variables)) + 0.2
gen_weight = 2

total_mean = 0
total_cov = 0
total_beta = 0
totla_diff_sklearn = 0

model = LinearRegression(fit_intercept=False)


def sklearn_fit(graph, b):
    mean = 0
    for v, ps in [(v, graph.parents(v)) for v in graph.nodes_iter() if len(graph.parents(v))]:
        coef = model.fit(data_gn[:, ps], data_gn[:, v]).coef_
        mean += norm(coef - b[v, ps])

    return mean / graph.n_nodes


for i, s in enumerate(seeds):
    print('Test {0}/{1}'.format(i + 1, len(seeds)))
    graph = random_dag(variables, rng=rng)

    beta = graph.A.T * gen_weight

    sample_seed = rng.randint(0, 2**32-1)

    data_gn = sample_from_gn(graph, gen_mean, gen_var, beta, n_samples, sample_seed)

    mu, var, b = gn_params(graph, data_gn)
    mu, sigma = to_mvn(mu, var, b, structure=graph)

    # print(beta)
    # print(b)

    mean_norm = norm(gen_mean - mu)
    var_norm = norm(gen_var - var)
    # noinspection PyTypeChecker
    beta_norm = norm(beta - b)
    sklearn_check = sklearn_fit(graph, b)

    total_mean += mean_norm
    total_cov += var_norm
    total_beta += beta_norm
    totla_diff_sklearn += sklearn_check

    print(' Distance between means: {0}\n Distance between variances: {1}'
          '\n Distance between parameters: {2}\n Distance to Sklearn estimate: {3}'.
          format(mean_norm, var_norm, beta_norm, sklearn_check))

print('Finished: \n Mean norm: {0}\n Cov Norm: {1}\n Beta Norm: {2}\n Mean diff sklearn estimate: {3}'.
      format(total_mean / len(seeds), total_cov / len(seeds), total_beta / len(seeds), totla_diff_sklearn / len(seeds)))
