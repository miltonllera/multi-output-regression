import networkx as nx
import numpy as np
import pylab as pl
from numpy.random import RandomState
from recordclass import recordclass

from core.gaussian import sample_from_gn
from structure.graph_generation import random_dag

# Parameters
GraphParams = recordclass('GraphParams', ['n_targets', 'n_features', 'max_parents'])
SamplingParams = recordclass('SamplingParams', ['n_steps', 'sample_freq', 'burn_in', 'move_prob', 'trace'])
DataParams = recordclass('DataParams', ['gen_mean', 'variance', 'gen_weight', 'n_samples'])

# CONFIG
graph_params = GraphParams(n_targets=5, n_features=10, max_parents=None)

# Data generation parameters
data_params = DataParams(mean=0, variance=0.2, weight=2, n_samples=200)

# Sampling parameters
sampling_params = SamplingParams(
    n_steps=1000, sample_freq=10, burn_in=None, move_prob=np.asarray([.40, .40, .185, .015]), trace=True)

# Random Number Generator
rng = RandomState(333)


def get_features_targets():
    return np.arange(graph_params.n_features), np.arange(graph_params.n_targets) + graph_params.n_features


def create_dag():
    n_targets, n_features, max_parents = graph_params
    variables = list(range(n_features + n_features))

    graph = random_dag(variables, fan_in=max_parents, rng=rng)

    return graph


def create_dataset(graph):
    n_targets, n_features, max_parents = graph_params
    n_vars = n_targets + n_features

    mean, variance, weight, n_samples = data_params

    # Sample the data
    mu = np.zeros(n_vars) + mean
    tau = variance * np.ones(n_vars)

    nx.draw_networkx(graph)
    pl.show()

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError()

    return sample_from_gn(graph, mu, tau, weight, size=n_samples, rng=rng)


def split_feature_targets(data, n_features=None):
    if n_features is None:
        n_features = graph_params.n_features

    return data[:, :n_features], data[:, n_features:]
