import numpy as np
import seaborn as sns
from numpy.random import RandomState

from sklearn.metrics import mean_squared_error
from metrics.score import BGe
from structure.graphs import plot_digraph
from mcmc.graphs.sampler import MHStructureSampler
from mcmc.graphs.proposal import MBCProposal, basic_move, rev_move, nbhr_move

from core.gaussian import to_mvn, gn_params_ridge
from model.mgnr import MGNREnsemble

from structure.graph_generation import random_mbc
from core.gaussian import sample_from_gn

sns.set(color_codes=True)
SAVE = False

n_variables = 15
n_features = 10
seeds = list(range(101, 200))
rng = RandomState(1802)
variables = list(range(n_variables))
n_samples = 200

# Data generation parameters
gen_mean = np.zeros(n_variables)
gen_var = np.zeros(n_variables) + 0.2
gen_weight = 2

# Generate some data form a GN
graph = random_mbc(n_features, n_variables - n_features, rng=rng, fan_in=5)
beta = graph.A.T * gen_weight

sample_seed = rng.randint(0, 2 ** 32 - 1)
data = sample_from_gn(graph, gen_mean, gen_var, beta, n_samples, sample_seed)
test = sample_from_gn(graph, gen_mean, gen_var, beta, n_samples, sample_seed)

X, Y = data[:, :n_features], data[:, n_features:]
X_test, Y_test = test[:, :n_features], data[:, n_features:]

graph_score = BGe(data)(graph)

print('Graph created with {} variables. Dataset with {} samples. Graphs bge score = {:.2f}'.format(
    n_variables, n_samples, graph_score))

plot_digraph(graph)

# Fit the score and create the parent set distributions
fan_in = 5

moves = [basic_move, rev_move, nbhr_move]
move_probs = [13/15, 1/15, 1/15]

sampler = MHStructureSampler(
    proposal=MBCProposal(moves, move_prob=move_probs, score=BGe, fan_in=5, random_state=rng),
    n_steps=1000, sample_freq=10, burn_in=500, verbose=False, rng=rng
)


def parameter_estimator(structure, data):
    return to_mvn(*gn_params_ridge(structure, data, sparse=True, l2_reg=0.1))


model = MGNREnsemble(
    k=10, parameter_estimator=parameter_estimator, structure_optimization=sampler, rng=rng, verbose=False).fit(X, Y)

predicted = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y, predicted))

print(rmse)
