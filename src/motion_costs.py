from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.linalg as npla
from tools import colvec


class StateCost(object):
    """Base class for all other cost function classes. Derived classes must implement, at a minimum,
    the .cost(state_dict) method, and populate the self.required_state_vars instance variable
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def cost(self, state_dict):
        """The value of this cost function for a single system state, given by state_dict. The 
        state_dict arg should be a dict mapping state variable names to scalar values. The dict must
        contain the keys in the list returned by get_required_state_vars(), but can contain other
        state variables as well.
        """
        pass

    def get_required_state_vars(self):
        """Returns a list of the state variable names needed by this cost function. Derived classes
        must define self.required_state_vars.
        """
        return self.required_state_vars

    def jacobian(self, state_dict):
        """Computes the derivative of this cost function with respect to each of the state variables
        in the list returned by self.get_required_state_vars(). The result is returned as a dict
        mapping each state variable's name to it's respective derivative. The default 
        """
        EPSILON = 1.0e-4
        jacobian_dict = {}
        initial_cost = self.cost(state_dict)
        perturbed_state_dict = state_dict.copy()
        for state_var in self.get_required_state_vars():
            perturbed_state_dict[state_var] = perturbed_state_dict[state_var] + EPSILON
            jacobian_dict[state_var] = (self.cost(perturbed_state_dict) - initial_cost) / EPSILON
            perturbed_state_dict[state_var] = perturbed_state_dict[state_var] - EPSILON
        return jacobian_dict


class QuadraticDisplacementCost(StateCost):
    def __init__(self, state_names, neutral_state_values):
        if len(state_names) != neutral_state_values:
            raise ValueError('State names and neutral values lists must have same length')
        self.required_state_vars = state_names
        self.neutral_state_values = neutral_state_values

    def cost(self, state_dict):
        total_cost = 0
        for state_var in self.required_state_vars:
            try:
                total_cost += (state_dict[state_var] - self.neutral_state_values[state_var]) ** 2
            except KeyError:
                raise ValueError('State dict missing the variable: \'' + state_var + '\'')
        return total_cost


class ManipulabilityCost(StateCost):

    intent_states = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

    def __init__(self, object_joint_names, get_object_human_jacobian, intent):
        self.required_state_vars = object_joint_names

        assert set(intent) <= set(self.intent_states), 'intent must be a dict with a subset of these keys: %s' % \
                                                       self.intent_states

        self.intent = intent
        self.get_object_human_jacobian = get_object_human_jacobian

        self.intent_pose_indices = []
        intent_list = []

        for i, state in enumerate(self.intent_states):
            if state in self.intent:
                self.intent_pose_indices.append(i)
                intent_list.append(self.intent[state])

        self.intent_array = np.array(intent_list)

    def cost(self, state_dict):
        jac_dict = self.get_object_human_jacobian()
        jac_list = []
        for state in self.required_state_vars:
            jac_list.append(jac_dict[state])

        jac = np.array(jac_list).T
        jac = jac[self.intent_pose_indices, :]

        return npla.norm(np.dot(npla.pinv(jac), self.intent_array))


class WeightedCostCombination(StateCost):
    def __init__(self, cost_funcs, weights=None):
        if len(cost_funcs) != len(weights):
            raise ValueError('Cost functions and weights lists must have same length')
        self.cost_funcs = cost_funcs
        self.required_state_vars = []
        for cost_func in cost_funcs:
            for required_var in cost_func.get_required_state_vars():
                if required_var not in self.required_state_vars:
                    self.required_state_vars.append(required_var)

        if weights is None:
            uniform_weight = 1.0 / len(cost_funcs)
            self.weights = [uniform_weight] * len(cost_funcs)
        else:
            self.weights = weights

    def cost(self, state_dict):
        total_cost = 0
        for (weight, cost_func) in zip(self.weights, self.cost_funcs):
            total_cost += cost_func.cost(state_dict) * weight
        return total_cost