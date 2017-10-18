import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error
from numpy.random import RandomState

from pymoreg.core.gaussian import sample_from_gn
from pymoreg.core.gaussian import to_mvn, gn_params_ridge
from pymoreg.structure.graph_generation import random_mbc
# from pymoreg.structure.graphs import plot_digraph

from pymoreg.metrics.score import BGe
from pymoreg.mcmc.graphs.proposal import MBCProposal, basic_move, rev_move, nbhr_move
from pymoreg.mcmc.graphs.sampler import MHStructureSampler

from pymoreg.model import MGNREnsemble

sns.set(color_codes=True)
SAVE = False

n_variables = 15
n_features = 10
seeds = list(range(101, 200))
rng = RandomState(1802)
variables = list(range(n_variables))
n_samples = 300

# Data generation parameters
gen_mean = np.zeros(n_variables)
gen_var = rng.gamma(shape=1, size=n_variables)
# gen_weight = 2

# Generate some data form a GN
graph = random_mbc(n_features, n_variables - n_features, rng=rng, fan_in=5)
beta = np.multiply(graph.A.T, rng.normal(0, 2, size=graph.shape))

sample_seed = rng.randint(0, 2 ** 32 - 1)
data = sample_from_gn(graph, gen_mean, gen_var, beta, n_samples, sample_seed)
test = sample_from_gn(graph, gen_mean, gen_var, beta, n_samples, sample_seed)

X, Y = data[:, :n_features], data[:, n_features:]
X_test, Y_test = test[:, :n_features], data[:, n_features:]

graph_score = BGe(data)(graph)

print('Graph created with {} variables. Dataset with {} samples. Graphs bge score = {:.2f}'.format(
    n_variables, n_samples, graph_score))

# plot_digraph(graph)

# Fit the score and create the parent set distributions
fan_in = 5

moves = [basic_move, rev_move, nbhr_move]
move_probs = [13/15, 1/15, 1/15]

sampler = MHStructureSampler(
    proposal=MBCProposal(moves, move_prob=move_probs, score=BGe, fan_in=fan_in, random_state=rng),
    n_steps=10000, sample_freq=100, burn_in=5000, verbose=False, rng=rng
)


def parameter_estimator(structure, data):
    return to_mvn(*gn_params_ridge(structure, data, sparse=True, l2_reg=0.1))


model = MGNREnsemble(
    k=50, parameter_estimator=parameter_estimator, structure_fitter=sampler, rng=rng, verbose=False).fit(X, Y)

predicted = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y, predicted))

print(rmse)
