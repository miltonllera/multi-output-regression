import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
from collections import OrderedDict
from mcmc.graphs.sampler import DAGDistribution


def running_mean(traces):
    if isinstance(traces, pd.DataFrame):
        traces = traces.values
    if isinstance(traces, OrderedDict):
        traces = np.asarray(list(traces.values()))

    rolling_mean = np.zeros(traces.shape)
    rolling_mean[:, 0] = traces[:, 0]

    n = traces.shape[1]

    for i in range(1, traces.shape[1]):
        rolling_mean[:, i] = traces[:, i] + rolling_mean[:, i - 1]
        rolling_mean[:, i - 1] /= i

    rolling_mean[:, n - 1] /= n

    return rolling_mean


# def parameter_autocorrelation(traces):
#     if isinstance(traces, pd.DataFrame):
#         traces = traces.values
#
#     auto_correlation = np.zeros(traces.shape)
#
#     for i in range(traces.shape[0]):
#         auto_correlation[i] = np.correlate(traces[i], traces[i])
#
#     return auto_correlation


def trace_plots(samples, scores, edges):

    if isinstance(samples, list):
        samples = DAGDistribution(samples)
    elif not isinstance(samples, DAGDistribution):
        raise ValueError('Unknown type of variable samples')

    params = samples.get_param_values(edges)
    rm = running_mean(params)

    fig, (ax1, ax2) = pl.subplots(nrows=2, sharex='col')

    sns.tsplot(rm, err_style='unit_traces', ax=ax1)
    sns.tsplot(scores, ax=ax2)

    pl.show()


def edge_prob_scatter_plot(params1: DAGDistribution, params2: DAGDistribution, edges, hue=None):
    probs1 = [params1.edge_prob(e) for e in edges]
    probs2 = [params2.edge_prob(e) for e in edges]

    df = pd.DataFrame()
    df['Run #1'] = probs1
    df['Run #2'] = probs2
    df['hue'] = hue

    sns.lmplot('Run #1', 'Run #2', df, fit_reg=False)
    pl.show()


def score_density_plot(scores):
    sns.distplot(scores)
    pl.show()
