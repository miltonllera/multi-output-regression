import pickle
import numpy as np
import scipy.sparse.csgraph as csg
from scipy.stats import multivariate_normal as mvn
from itertools import chain, product

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError

from core.misc import get_rng
from core.gaussian import gn_params, conditional_mvn_params, to_mvn
from structure.graphs import MBCGraph
from mcmc.graphs.sampler import MHStructureSampler
from mcmc.graphs.proposal import MBCProposal


# noinspection PyAttributeOutsideInit
class MGNR(BaseEstimator, RegressorMixin):
    def __init__(self, parameter_fitter=None, verbose=False):
        if parameter_fitter is None or parameter_fitter == 'mle':
            parameter_fitter = lambda structure, data: to_mvn(*gn_params(structure, data))
        else:
            raise NotImplementedError('Only mle estimation is currently available')

        self.fit_params = parameter_fitter
        self.verbose = verbose

    @property
    def n_vars(self):
        try:
            return self.mean_.shape[0]
        except KeyError:
            raise NotFittedError('Cannot access n_vars property until model is fitted')

    @property
    def n_targets(self):
        try:
            return self.n_targets_
        except KeyError:
            raise NotFittedError('Cannot access n_targets property until model is fitted')

    @property
    def n_features(self):
        try:
            return self.mean_.shape[0] - self.n_targets
        except KeyError:
            raise NotFittedError('Cannot access n_features property until model is fitted')

    @property
    def is_fitted(self):
        try:
            # noinspection PyStatementEffect
            self.n_vars
            return True
        except NotFittedError:
            return False

    def fit(self, X, y, structure: MBCGraph=None):
        if structure is None:
            raise NotImplementedError()

        self.n_targets_ = y.shape[1]

        data = np.hstack((X, y))
        self.mean_, self.sigma_ = self.fit_params(structure, data)

        features, targets = X.shape[1], y.shape[1]

        regression_structure = structure.copy()
        regression_structure[list(range(features))] = False

        n_comp, labels = csg.connected_components(regression_structure, directed=True)

        target_groups = [[] for _ in range(n_comp)]
        feature_groups = [[] for _ in range(n_comp)]

        for n, l in enumerate(labels):
            if n in targets:
                target_groups[l].append(n)
            else:
                feature_groups[l].append(n)

        for i in range(len(target_groups)):
            target_groups[i] = sorted(target_groups[i])

        for i in range(len(feature_groups)):
            feature_groups[i] = sorted(feature_groups[i])

        self.components_ = list(zip(feature_groups, target_groups))

        return self

    def log_prob(self, y, X=None):
        if X is None:
            idx = sorted(chain.from_iterable(self.components_))
            mean, cov = self.mean_[idx], self.sigma_[idx, idx]
        else:
            mean, cov = self.predict(X, return_cov=True)

        return mvn.logpdf(y, mean, cov)

    def prob(self, y, X=None):
        return np.exp(self.log_prob(y, X))

    # noinspection PyUnboundLocalVariable
    def predict(self, X, return_cov=False):
        if not self.is_fitted:
            raise Exception()

        predictions = np.zeros(self.n_targets_, dtype=np.float)

        if return_cov:
            predicted_cov = np.zeros((self.n_targets_, self.n_targets_), dtype=np.float)

        for comp_f, comp_t in self.components_:

            # x = X[comp_f[0]]
            x = X
            comp_vars = list(chain(comp_f, comp_t))
            mean_ = self.mean_[comp_vars]
            cov_ = self.sigma_[np.ix_(comp_vars, comp_vars)]

            cond_params = conditional_mvn_params(mean_, cov_, x, return_cov)

            if return_cov:
                predictions[np.asarray(comp_t) - self.n_features] = cond_params[0]
                x, y = list(zip(product(comp_t, repeat=2)))
                predicted_cov[x, y] = cond_params[1]
            else:
                predictions[np.asarray(comp_t) - self.n_features] = cond_params
                predicted_cov = None

        return predictions if not return_cov else (predictions, predicted_cov)

    @staticmethod
    def from_params(mean, sigma, components, n_targets):
        model = MGNR()

        model.components_ = components
        model.mean_ = mean
        model.sigma_ = sigma
        model.n_targets_ = n_targets

        return model


class MGNREnsemble(BaseEstimator, RegressorMixin):
    # noinspection PyUnusedLocal
    def __init__(self, k=1, optimizer=None, rng=None, verbose=False):
        """
        Initializes the models.

        Parameters
        ----------
        k: int
            The number of sample networks used for prediction. k must be smaller or equal than the number of samples
            returned by the optimizer.
        optimizer: MHStructureOptimizer
            The algorithm used to learn the structure of the model
        rng: RandomState, int or None (default)
            A random state for the class and al its members.
        """
        if optimizer is None:
            raise NotImplementedError()

        if k is None:
            k = optimizer.returned_samples

        if k > optimizer.returned_samples:
            raise ValueError('The optimizer is set to return less samples than expected: {0} > {1}'.format(
                k, optimizer.returned_samples))

        self.rng = get_rng(rng)
        self.optimizer = optimizer
        self.k = k
        self.verbose = True

    @property
    def n_vars(self):
        try:
            return self.models_[0].n_vars
        except KeyError:
            raise NotFittedError

    @property
    def n_targets(self):
        try:
            return self.models_[0].n_targets
        except KeyError:
            raise NotFittedError

    @property
    def n_features(self):
        try:
            return self.models_[0].n_features
        except KeyError:
            raise NotFittedError

    @property
    def is_fit(self):
        try:
            return len(self.models_)
        except KeyError:
            raise NotFittedError

    def get_params(self, deep=True):
        raise NotImplemented()

    def set_params(self, **params):
        raise NotImplemented()

    def fit(self, X, y, samples=None):
        """
        Fits the model using MCMC sampling of the structure space and then uses a point estimation procedure for the
        parameters of each of the nodes. Right now only MLE estimation is available but Ridge, LASSO and Elastic Net
        could be added.

        Parameters
        ----------
        X: array like
            2-D array of feature variables of the data.
        y: array like
            2-D array of target variables
        samples: list of tuples
            Network structures. If None will use the optimizer to find a set of structures.
        Returns
        -------
        out: MGNREnsemble
            The trained model consisting of one or more trained conditional MVN with different base structures to some
            models.
        """
        # Find structure
        if self.verbose:
            print('learning structure...')

        if samples is None:
            samples = self.optimizer.generate_samples((X, y), return_scores=True)

        samples = sorted(zip(*samples), key=lambda s: s[1])

        samples = list(zip(*samples))[0]
        networks = samples[-self.k:]

        if self.verbose:
            print('fiting parameters...')

        self._parameter_fit(X, y, networks)

        if self.verbose:
            print('done')

        return self

    def _parameter_fit(self, X, y, network, estimate_type='mle'):
        if estimate_type == 'mle':
            self.models_ = [MGNR('mle').fit(X, y, net) for net in network]
        else:
            raise NotImplementedError()

    def predict(self, X):
        return np.mean([m.predict(X, return_cov=False) for m in self.models_], axis=0)

    def log_prob(self, y, X=None):
        return np.log(self.prob(y, X))

    def prob(self, y, X=None):
        return np.mean([m.prob(X, y) for m in self.models_])


def save_model(model: MGNREnsemble, path):
    n_targets = model.n_targets

    mvns_params = [(m.mean_, m.sigma_, m.components_) for m in model.models_]

    parameters = {'n_targets': n_targets, 'mvns': mvns_params}

    with open(path, mode='wb') as f:
        pickle.dump(parameters, f)


def load_model(path):
    with open(path, mode='rb') as f:
        params = pickle.load(f)

    n_targets = params['n_targets']
    models = [MGNR.from_params(mean, sigma, components, n_targets) for mean, sigma, components in params['mvns']]

    model = MGNREnsemble()
    model.k = len(models)
    model.models_ = models

    return model
