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
    def __init__(self, fit_params=None, verbose=False):
        if fit_params is None or fit_params == 'mle':
            fit_params = lambda structure, data: to_mvn(*gn_params(structure, data, sparse=True))
        elif fit_params == 'ridge':
            fit_params = lambda structure, data: to_mvn(*gn_params(structure, data, sparse=True, l2_reg=0.1))
        elif callable(fit_params):
            fit_params = fit_params
        else:
            raise NotImplementedError('Only mle estimation is currently available')

        self.fit_params = fit_params
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

        features, targets = np.arange(X.shape[1]), np.arange(y.shape[1]) + X.shape[1]

        regression_structure = structure.copy()
        regression_structure[list(features)] = False

        n_comp, labels = csg.connected_components(regression_structure, directed=False)

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

        predictions = []
        if return_cov:
            covariances = []

        for i, X_i in enumerate(X):
            pred = np.zeros(self.n_targets_, dtype=np.float)

            if return_cov:
                predicted_cov = np.zeros((self.n_targets_, self.n_targets_), dtype=np.float)

            for comp_f, comp_t in self.components_:
                if not len(comp_t):
                    continue

                x = X_i[comp_f]
                comp_vars = list(chain(comp_f, comp_t))
                mean_ = self.mean_[comp_vars]
                cov_ = self.sigma_[np.ix_(comp_vars, comp_vars)]

                cond_params = conditional_mvn_params(mean_, cov_, x, return_cov)

                if return_cov:
                    x, y = list(zip(product(comp_t, repeat=2)))
                    predicted_cov[x, y] = cond_params[1]
                    cond_params = cond_params[0]

                pred[np.asarray(comp_t, dtype=int) - self.n_features] = cond_params

            predictions.append(pred)

            if return_cov:
                covariances.append(predicted_cov)

        predictions = np.asarray(predictions)
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
    def __init__(self, k=1, parameter_estimator=None, structure_optimization=None, rng=None, verbose=False):
        """
        Initializes the models.

        Parameters
        ----------
        k: int
            The number of sample networks used for prediction. k must be smaller or equal than the number of samples
            returned by the struct_opt.
        parameter_estimator: callable
            The algorithm used to determine the values of the regression coefficients.
        structure_optimization: MHStructureOptimizer
            The algorithm used to learn the structure of the model
        rng: RandomState, int or None (default)
            A random state for the class and al its members.
        """
        if structure_optimization is None:
            raise NotImplementedError()

        if k is None:
            k = structure_optimization.returned_samples

        if k > structure_optimization.returned_samples:
            raise ValueError('The structure_optimization is set to return less samples than expected: {0} > {1}'.format(
                k, structure_optimization.returned_samples))

        self.rng = get_rng(rng)
        self.param_estimator = parameter_estimator
        self.struct_opt = structure_optimization
        self.k = k
        self.verbose = verbose

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

    # noinspection PyAttributeOutsideInit
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
            Network structures. If None will use the struct_opt to find a set of structures.
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
            samples = self.struct_opt.generate_samples((X, y), return_scores=True)

        samples = sorted(zip(*samples), key=lambda s: s[1])

        samples = list(zip(*samples))[0]
        networks = samples[-self.k:]

        if self.verbose:
            print('fiting parameters...')

        self.models_ = [MGNR(self.param_estimator).fit(X, y, net) for net in networks]

        if self.verbose:
            print('done')

        return self

    def predict(self, X):
        return np.mean([m.predict(X, return_cov=False) for m in self.models_], axis=0)

    def log_prob(self, Y, X=None):
        return np.log(self.prob(Y, X))

    def prob(self, Y, X=None):
        return np.mean([m.prob(X, Y) for m in self.models_])


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
