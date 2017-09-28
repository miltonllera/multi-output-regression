import numpy as np
import seaborn as sns
from numpy.random import RandomState

from metrics.score import BGe
from structure.graphs import plot_digraph
from mcmc.graphs.sampler import MHStructureSampler
from mcmc.graphs.proposal import MBCProposal, basic_move, rev_move, nbhr_move
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

graph_score = BGe(data)(graph)

print('Graph created with {} variables. Dataset with {} samples. Graphs bge score = {:.2f}'.format(
    n_variables, n_samples, graph_score))

plot_digraph(graph)

# Fit the score and create the parent set distributions
fan_in = 5

moves = [basic_move, rev_move, nbhr_move]
move_probs = [13/15, 1/15, 1/15]

# moves = [basic_move, rev_move]
# move_probs = [14/15, 1/15]


X, y = data[:, :n_features], data[:, n_features:]
X_test, y_test = test[:, :n_features], data[:, n_features:]

sampler = MHStructureSampler(
    proposal=MBCProposal(moves, move_prob=move_probs, score=BGe, fan_in=5, random_state=rng),
    n_steps=10000, sample_freq=100, burn_in=5000, verbose=True, rng=rng
)

model = MGNREnsemble(k=100, optimizer=sampler, rng=rng, verbose=True).fit(X, y)

sq_error = 0

for x, y in zip(X_test, y_test):
    predicted = model.predict(X).reshape(1, -1)

    sq_error += (y - predicted) ** 2

rmse = np.sqrt(sq_error / X_test.shape[0])
print(rmse)
