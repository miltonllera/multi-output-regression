import numpy as np
import pylab as pl
import seaborn as sns
from numpy.random import RandomState

from metrics.score import BGe
from mcmc.graphs.sampler import MHStructureSampler, GraphDistribution
from mcmc.graphs.proposal import DAGProposal, basic_move, rev_move
from mcmc.diagnostics import running_mean

from structure.graph_generation import random_dag
from core.gaussian import sample_from_gn

sns.set(color_codes=True)
SAVE = False

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

sample_seed = rng.randint(0, 2 ** 32 - 1)
data_gn = sample_from_gn(graph, gen_mean, gen_var, beta, n_samples, sample_seed)

# Fit the score and create the parent set distributions
fan_in = 5

moves = [basic_move, rev_move]


sampler = MHStructureSampler(
    proposal=DAGProposal(moves, move_prob=[0.85, 0.15], score=BGe, fan_in=5, random_state=rng),
    n_steps=10000, sample_freq=100, burn_in=5000, verbose=True, rng=rng
)

trace = sampler.generate_samples(data_gn, return_scores=True, debug=False)

samples, scores = trace

g_dist = GraphDistribution(samples)

params = g_dist.get_param_values(graph.edges())
rm = running_mean(params)

fig, (ax1, ax2) = pl.subplots(nrows=2)

sns.tsplot(rm, err_style='unit_traces', ax=ax1)
sns.tsplot(scores, ax=ax2)

pl.show()
