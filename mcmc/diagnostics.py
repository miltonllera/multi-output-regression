import numpy as np
import pandas as pd
from collections import OrderedDict


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


def parameter_autocorrelation(traces):
    if isinstance(traces, pd.DataFrame):
        traces = traces.values

    auto_correlation = np.zeros(traces.shape)

    for i in range(traces.shape[0]):
        auto_correlation[i] = np.correlate(traces[i], traces[i])

    return auto_correlation


def gr_convergence_measure(trace):
    raise NotImplementedError()
