import math
from abc import ABC, abstractmethod

import torch
from scipy.special import gammaln


class BasePrior(ABC):
    """Base class for the prior over graphs p(G).

    Any subclass of `BasePrior` must return the contribution of log p(G) for a
    given variable with `num_parents` parents. We assume that the prior is modular.

    Parameters
    ----------
    num_variables : int (optional)
        The number of variables in the graph. If not specified, this gets
        populated inside the scorer class.
    """

    def __init__(self, num_variables=None):
        self._num_variables = num_variables
        self._log_prior = None

    def __call__(self, num_parents):
        if self.log_prior is None:
            raise ValueError("log_prior has not been initialized.")
        return self.log_prior[num_parents]

    @property
    @abstractmethod
    def log_prior(self):
        pass

    @property
    def num_variables(self):
        if self._num_variables is None:
            raise RuntimeError("The number of variables is not defined.")
        return self._num_variables

    @num_variables.setter
    def num_variables(self, value):
        self._num_variables = value


class UniformPrior(BasePrior):
    @property
    def log_prior(self):
        if self._log_prior is None:
            self._log_prior = torch.zeros(self.num_variables)
        return self._log_prior


class ErdosRenyiPrior(BasePrior):
    def __init__(self, num_variables=None, num_edges_per_node=1.0):
        super().__init__(num_variables)
        self.num_edges_per_node = num_edges_per_node

    @property
    def log_prior(self):
        if self._log_prior is None:
            num_edges = self.num_variables * self.num_edges_per_node  # Default value
            p = num_edges / ((self.num_variables * (self.num_variables - 1)) // 2)
            all_parents = torch.arange(self.num_variables)
            self._log_prior = all_parents * math.log(p) + (
                self.num_variables - all_parents - 1
            ) * math.log1p(-p)
        return self._log_prior


class EdgePrior(BasePrior):
    def __init__(self, num_variables=None, beta=1.0):
        super().__init__(num_variables)
        self.beta = beta

    @property
    def log_prior(self):
        if self._log_prior is None:
            self._log_prior = torch.arange(self.num_variables) * math.log(self.beta)
        return self._log_prior


class FairPrior(BasePrior):
    @property
    def log_prior(self):
        if self._log_prior is None:
            all_parents = torch.arange(self.num_variables)
            self._log_prior = (
                -gammaln(self.num_variables + 1)
                + gammaln(self.num_variables - all_parents + 1)
                + gammaln(all_parents + 1)
            )
        return self._log_prior
