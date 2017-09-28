import numpy as np
import pylab as pl
import seaborn as sns
from numpy.random import RandomState

from metrics.score import BGe
from structure.graphs import plot_digraph
from mcmc.graphs.sampler import MHStructureSampler, DAGDistribution
from mcmc.graphs.proposal import DAGProposal, basic_move, rev_move, nbhr_move
from mcmc.diagnostics import trace_plots, score_density_plot, edge_prob_scatter_plot

from structure.graph_generation import random_dag
from core.gaussian import sample_from_gn

sns.set(color_codes=True)
SAVE = False

n_variables = 15
seeds = list(range(101, 200))
rng = RandomState(1802)
variables = list(range(n_variables))
n_samples = 200

# Data generation parameters
gen_mean = np.zeros(n_variables)
gen_var = np.zeros(n_variables) + 0.2
gen_weight = 2

# Generate some data form a GN
graph = random_dag(variables, rng=rng, fan_in=5)
beta = graph.A.T * gen_weight

sample_seed = rng.randint(0, 2 ** 32 - 1)
data_gn = sample_from_gn(graph, gen_mean, gen_var, beta, n_samples, sample_seed)

graph_score = BGe(data_gn)(graph)

print('Graph created with {} variables. Dataset with {} samples. Graphs bge score = {:.2f}'.format(
    n_variables, n_samples, graph_score))

plot_digraph(graph)

# Fit the score and create the parent set distributions
fan_in = 5

moves = [basic_move, rev_move, nbhr_move]

sampler = MHStructureSampler(
    proposal=DAGProposal(moves, move_prob=[27/30, 1/15, 1/30], score=BGe, fan_in=5, random_state=rng),
    n_steps=100000, sample_freq=100, burn_in=50000, verbose=True, rng=rng
)

trace1 = sampler.generate_samples(data_gn, return_scores=True, debug=False)
samples1, scores1 = trace1
g_dist1 = DAGDistribution(samples1)

# trace2 = sampler.generate_samples(data, return_scores=True, debug=False)
# samples2, scores2 = trace2
# g_dist2 = GraphDistribution(samples2)

trace_plots(samples1, scores1 - graph_score, graph.edges())
score_density_plot(scores1)

# edge_prob_scatter_plot(g_dist1, g_dist2, graph.edges())

# g_dist = GraphDistribution(samples)
#
# params = g_dist.get_param_values(graph.edges())
# rm = running_mean(params)
#
# fig, (ax1, ax2) = pl.subplots(nrows=2, sharex='col')
#
# sns.tsplot(rm, err_style='unit_traces', ax=ax1)
# sns.tsplot(scores - graph_score, ax=ax2)

pl.show()
