import numpy as np
from numpy.random import RandomState

from core.gaussian import sample_from_gn
from structure.graph_generation import random_dag

from metrics.score import BGe
from mcmc.graphs.state_space import RestrictionViolation
from mcmc.graphs.proposal import get_parent_set_distributions
from mcmc.graphs.checks import check_distribution

n_variables = 15
seeds = list(range(101, 200))
rng = RandomState(19023)
variables = list(range(n_variables))
n_samples = 200


# Data generation parameters
gen_mean = np.zeros(n_variables)
gen_var = np.zeros(n_variables) + 0.2
gen_weight = 2

# Generate some data form a GN
graph = random_dag(variables, rng=rng)
beta = graph.A.T * gen_weight

sample_seed = rng.randint(0, 2**32-1)
data_gn = sample_from_gn(graph, gen_mean, gen_var, beta, n_samples, sample_seed)

# Fit the score and create the parent set distributions
fan_in = 5

bge = BGe(data_gn)
ps_dist = get_parent_set_distributions(variables, fan_in, bge, rng=rng)

print('Checking proposal distribution')
s, f = 0, 0

for v, dist in enumerate(ps_dist):
    print('Node {0}...'.format(v))
    try:
        check_distribution(v, dist, fan_in)
        s += 1
        print('pass')
    except RestrictionViolation:
        f += 1
        print('fail')

    print('Test finished. {0}/{1} successes'.format(s, len(ps_dist)))


print('Checking moves')