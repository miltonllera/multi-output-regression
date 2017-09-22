import numpy as np
from numpy.random import RandomState

from core.gaussian import sample_from_gn
from structure.graph_generation import random_dag

from metrics.score import BGe
from mcmc.graphs.state_space import DAGState, RestrictionViolation
from mcmc.graphs.proposal import get_parent_set_distributions, basic_move, rev_move
from mcmc.graphs.checks import check_consistency

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

# Some random state to start
state = DAGState(random_dag(variables, fan_in, rng=rng))

# Check consistency of first state
check_consistency(state, fan_in)

# Basic moves
moves = [basic_move, rev_move]

s, f = 0, 0

# Randomly apply moves to test them
tests = 100
for i in range(tests):
    print('Test {0}/{1}...'.format(i + 1, len(seeds)))

    state.fan_in_ = fan_in

    m = rng.choice(moves)
    edges = m.moves(state)
    new_state, _, _ = m.propose(state, edges, ps_dist, rng)

    try:
        check_consistency(new_state, fan_in)
        state = new_state
        s += 1
        print('pass')
    except RestrictionViolation:
        f += 1
        print('fail')

print('Test finished. {0}/{1} successes'.format(s, tests))
