from abc import ABC, abstractmethod


class Env(ABC):
    """
    Base class for environments, showing which methods should be implemented.
    A common assumption for all environments is that all nodes of the DAG (except s_f)
    can be represented as a fixed length 1-D tensor, and all actions are discrete,
    represented by a number in {0, ..., n_actions - 1}.
    """
    @abstractmethod
    def is_actions_valid(self, states, actions):
        """
        :param states: FloatTensor of shape k x state_dim
        :param actions: LongTensor of shape k
        :return: True if all actions are valid at the states
        """
        pass

    def step(self, states, actions):
        """
        Function that takes a tensor representing a set of states and a set of actions,
        and returns a triplet corresponding to the new states, and done's.
        Should raise a ValueError if one of the actions is invalid.
        :param states: FloatTensor of shape k x state_dim
        :param actions: LongTensor of shape k
        :return: tuple (new_states, dones) FloatTensor of shape k x state_dim and BoolTensor of shape k.        """
        if not self.is_actions_valid(states, actions):
            raise ValueError('Actions are not valid')

    @abstractmethod
    def reward(self, states):
        """
        :param states: FloatTensor of shape k x state_dim
        :return: FloatTensor of shape k representing the rewards at the input states
        """
        pass

    @abstractmethod
    def mask_maker(self, states):
        """
        :param states: FloatTensor of shape k x state_dim
        :return: BoolTensor of shape k x n_actions-1 representing which actions are valid at the input states. 
        True when action is INVALID, i.e. it should be MASKed out !
        """
        pass

    @abstractmethod
    def backward_mask_maker(self, states):
        """
        :param states: FloatTensor of shape k x state_dim
        :return: BoolTensor of shape k x n_actions representing which actions could have led to the input states.
        True when action is INVALID, i.e. it should be MASKed out !
        """
        pass