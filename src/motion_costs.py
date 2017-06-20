#!/usr/bin/env python
import numpy as np
import numpy.linalg as npla
from abc import ABCMeta, abstractmethod
from kinmodel import Jacobian


class StateCost(object):
    """Base class for all cost function classes. Derived classes must implement, at a minimum, the
    cost method, and must call __init__ with the appopriate values.

    Args:
    name: string - the name of this cost function
    obs_func: function - takes state_dict as its only argument, and outputs a dict mapping
        observation variable names to (not necessarily scalar) observation values
    required_state_vars: list(string) - keys which must be present obs_func's input dict
    required_obs_vars: list(string) - keys which must be present in obs_func's output dict
    """
    __metaclass__ = ABCMeta
    def __init__(self, name, obs_func, required_state_vars, required_obs_vars):
        self.name = name
        self._obs_func = obs_func
        self._required_state_vars = required_state_vars
        self._required_obs_vars = required_obs_vars

    @abstractmethod
    def cost(self, state_dict):
        """The value of this cost function for a single system state, given by state_dict. The 
        state_dict arg should be a dict mapping state variable names to scalar values. The dict must
        contain the keys in the list returned by get_required_obs_vars(), but can contain other
        state variables as well.

        Args:
        state_dict: dict - the system state at which to compute the cost (must contain each of the
            values in self._required_state_vars as a key)
        """
        pass

    def get_required_obs_vars(self):
        """Returns a list of the observation variable names needed by this cost function. Derived
        classes must provide the list required_obs_vars in the constructor.
        """
        return self._required_obs_vars

    def get_required_state_vars(self):
        """Returns a list of the state variable names needed by this cost function. Derived
        classes must provide the list required_state_vars in the constructor.
        """
        return self._required_state_vars

    def gradient(self, state_dict):
        """Computes the derivative of this cost function with respect to each of the state variables
        in the list returned by self.get_required_state_vars(). The result is returned as a dict
        mapping each state variable's name to its respective derivative. The default implementation uses
        finite differences to compute this, but you can override this method with a more efficient
        algorithm if one is available (e.g. an analytic gradient).

        Args:
        state_dict: dict - the state of the system at which to compute the gradient
        """
        EPSILON = 1.0e-4
        jacobian_dict = {}
        initial_cost = self.cost(state_dict)
        perturbed_state_dict = state_dict.copy()
        for state_var in self.get_required_state_vars():
            perturbed_state_dict[state_var] = perturbed_state_dict[state_var] + EPSILON
            jacobian_dict[state_var] = (self.cost(perturbed_state_dict) - initial_cost) / EPSILON
            perturbed_state_dict[state_var] = perturbed_state_dict[state_var] - EPSILON
        return Jacobian(jacobian_dict, row_names=[self.name])


class StateErrorCost(StateCost):
    """Base class for all cost function classes which output an error between an observation and a
    reference value. Derived classes must implement, at a minimum, the _error_func method, and must
    call __init__ with the appopriate values.

    Args:
    ref_value: dict - maps observation names to their reference (not necessarily scalar) values
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, ref_value, obs_func, required_state_vars, required_obs_vars):
        self._ref_value = ref_value
        super(StateErrorCost, self).__init__(name, obs_func, required_state_vars, required_obs_vars)

    def cost(self, state_dict):
        """The value of this cost function for a single system state, given by state_dict.

        This default implementation computes the cost as the error between an observation and a
        static reference value.
        """
        obs = self._obs_func(state_dict)
        return self._error_func(obs, self._ref_value)

    @abstractmethod
    def _error_func(self, ref_value, obs):
        """Computes the (scalar) error between a reference value and an observation computed from
        the current system state using self._obs_func.

        Note that while the error itself is a scalar, the reference value and observation can be any
        type -- derived classes should each provide an appropriate _error_func.

        Args:
        ref_value: nominal reference value at which this cost attains a minimum
        obs: observation at the current state, which should lie in the same space as ref_value
        """
        pass


class QuadraticDisplacementCost(StateErrorCost):
    """StateErrorCost child class for all costs which are the sum of squared errors between a reference
    value and some function of the state.

    Args:
    name: string - this cost function's name
    ref_value: dict - the scalar reference value for each observed variable
    obs_func: func - function that takes a state dict and returns an observation dict (must contain
        all of the keys present in ref_value)
    required_state_vars: list - the keys in state_dict that are required for obs_func to compute
        an observation
    """
    def __init__(self, name, ref_value, obs_func, required_state_vars):
        required_obs_vars = ref_value.keys()
        super(QuadraticDisplacementCost, self).__init__(name, ref_value, obs_func,
                required_state_vars, required_obs_vars)

    def _error_func(self, ref_value, obs):
        # Add up the squared errors between each element in ref_value and the corresponding value
        # in obs
        total_cost = 0
        for obs_var in self.get_required_obs_vars():
            try:
                total_cost += (obs[obs_var] - ref_value[obs_var]) ** 2
            except KeyError:
                raise ValueError('Observation dict missing the variable: \'' + obs_var + '\'')
        return total_cost


class ManipulabilityCost(StateCost):

    intent_states = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

    def __init__(self, name, object_joint_names, get_object_manip_jacobian, intent):
        super(ManipulabilityCost, self).__init__(name,
                get_object_manip_jacobian, # obs_func
                object_joint_names, # required_state_vars
                intent.keys()) #TODO: what are the names of the required obs vars?

        # assert set(intent) <= set(self.intent_states), 'intent must be a dict with a subset of these keys: %s' % \
        #                                                self.intent_states
        self.intent = intent
        self.intent_pose_indices = []
        intent_list = []
        for i, state in enumerate(self.intent_states):
            if state in self.intent:
                self.intent_pose_indices.append(i)
                intent_list.append(self.intent[state])
        self.intent_array = np.array(intent_list)

    def cost(self, state_dict):
        assert set(self.get_required_state_vars()) <= set(state_dict.keys()), 'state_dict must contain the required ' \
                                                                              'state variables.\nrequired: %s\n' \
                                                                              'state_dict:%s' % \
                                                                              (self.get_required_state_vars(),
                                                                               state_dict.keys())

        # Get the Jacobian object
        jacobian = self._obs_func(state_dict)

        # Subset for the dimensions we care about, this will throw an error if the required_obs_vars are not a subset
        # of the Jacobian rows
        jacobian = jacobian.subset(row_names=self.get_required_obs_vars())

        # Invert, multiply with the intent array and take the norm (squared ?)
        joint_velocities = jacobian.pinv()*self.intent
        return npla.norm(joint_velocities.values()) ** 2


class BasisManipulabilityCost(ManipulabilityCost):
    def __init__(self, name, object_joint_names, get_object_manip_jacobian, intent_dimension):
        # Make sure we have chosen a possible pose dimension
        # assert intent_dimension in self.intent_states, "intent_dimensions must be one of %s" % self.intent_states

        # Make a 1-d unit intent
        intent = {intent_dimension: 1.0}

        # Pass along to parent init
        super(BasisManipulabilityCost, self).__init__(name, object_joint_names,
                get_object_manip_jacobian, intent)


class WeightedCostCombination(StateCost):
    """StateCost child class for costs which are weighted sums of other StateCosts.

    Args:
    cost_funcs: list(StateCost) - list of all the cost functions whose values to sum
    weights: list(float) - weights to apply to the corresponding cost functions in cost_funcs
        (defaults to None, which assigns each cost function a weight of 1/len(cost_funcs))
    """
    def __init__(self, name, cost_funcs, weights=None):
        # Generate cost_funcs and weights dicts
        self.cost_funcs = {cost_func.name: cost_func for cost_func in cost_funcs}
        if weights is None:
            self.weights = {cost_func_name: 1.0/len(cost_funcs) for cost_func_name in self.cost_funcs}
        else:
            self.weights = dict(zip(self.cost_funcs.keys(), np.array(weights)/np.sum(weights)))
        if len(cost_funcs) != len(self.weights):
            raise ValueError('Cost functions and weights lists must have same length')
        required_obs_vars = list(set().union(*[cost_func.get_required_obs_vars() for cost_func in cost_funcs]))
        required_state_vars = list(set().union(*[cost_func.get_required_state_vars() for cost_func in cost_funcs]))

        def obs_func(state_dict):
            # Return a dict of the costs returned by each cost function
            return {name: self.cost_funcs[name].cost(state_dict) for name in self.cost_funcs}

        super(WeightedCostCombination, self).__init__(name, obs_func, required_state_vars,
            required_obs_vars)

    def cost(self, state_dict):
        obs = self._obs_func(state_dict)
        total_error = 0
        for cost_func_name in self.cost_funcs:
            total_error += obs[cost_func_name] * self.weights[cost_func_name]
        return total_error

    def cost_basis_vector(self, state_dict):
        return np.array([self.cost_funcs[cost_func_name].cost(state_dict) for cost_func_name in self.cost_funcs])

    def gradient_bases(self, state_dict):
        # This is a (C, X) Jacobian object with each row being the direction of maximal ascent for that basis cost
        # function.
        return Jacobian.vstack([cost_func.gradient(state_dict) for cost_func in self.cost_funcs.values()])

    def get_basis_names(self):
        return self.cost_funcs.keys()

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights
