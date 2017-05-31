#!/usr/bin/env python
import numpy as np
import numpy.linalg as npla
from abc import ABCMeta, abstractmethod
from kinmodel import Jacobian


class StateCost(object):
    """Base class for all other cost function classes. Derived classes must implement, at a minimum,
    the .cost(state_dict) method, and populate the self.required_state_vars instance variable
    """
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name


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
        return Jacobian(jacobian_dict, row_names=[self.name])


class QuadraticDisplacementCost(StateCost):
    def __init__(self, name, reference):
        super(QuadraticDisplacementCost, self).__init__(name)
        self.required_state_vars = reference.keys()
        self.neutral_state_values = reference

    def cost(self, state_dict):
        total_cost = 0
        for state_var in self.required_state_vars:
            try:
                total_cost += (state_dict[state_var] - self.neutral_state_values[state_var]) ** 2
            except KeyError:
                raise ValueError('State dict missing the variable: \'' + state_var + '\'')
        return total_cost

    def jacobian(self, state_dict):
        jacobian_dict = {}

        for state_var in self.required_state_vars:
            try:
                jacobian_dict[state_var] = 2 * (state_dict[state_var] -
                                                              self.neutral_state_values[state_var])
            except KeyError:
                raise ValueError('State dict missing the variable: \'' + state_var + '\'')
        return Jacobian(jacobian_dict, row_names=[self.name])


class ManipulabilityCost(StateCost):

    intent_states = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

    def __init__(self, name, object_joint_names, get_object_manip_jacobian, intent):
        super(ManipulabilityCost, self).__init__(name)
        self.required_state_vars = object_joint_names

        # assert set(intent) <= set(self.intent_states), 'intent must be a dict with a subset of these keys: %s' % \
        #                                                self.intent_states

        self.intent = intent
        self.get_object_human_jacobian = get_object_manip_jacobian

        self.intent_pose_indices = []
        intent_list = []

        for i, state in enumerate(self.intent_states):
            if state in self.intent:
                self.intent_pose_indices.append(i)
                intent_list.append(self.intent[state])

        self.intent_array = np.array(intent_list)

    def cost(self, state_dict):
        assert set(self.required_state_vars) <= set(state_dict.keys()), 'state_dict must contain the required state ' \
                                                                        'variables.\nrequired: %s\nstate_dict:%s' % \
                                                                        (self.required_state_vars, state_dict.keys())

        # Get the dict of columns for the jacobian
        jacobian = self.get_object_human_jacobian(state_dict)

        # Subset for the dimensions we care about
        jacobian = jacobian.subset(row_names=self.intent.keys())

        print(jacobian.pinv())
        print(self.intent)
        print(jacobian.pinv()*self.intent)
        print(npla.norm(jacobian.pinv()*self.intent) ** 2)

        # Invert, multiply with the intent array and take the norm (squared ?)
        return npla.norm(jacobian.pinv()*self.intent) ** 2


class BasisManipulabilityCost(ManipulabilityCost):

    def __init__(self, name, object_joint_names, get_object_manip_jacobian, intent_dimension):

        # Make sure we have chosen a possible pose dimension
        # assert intent_dimension in self.intent_states, "intent_dimensions must be one of %s" % self.intent_states

        # Make a 1-d unit intent
        intent = {intent_dimension: 1.0}

        # Pass along to parent init
        super(BasisManipulabilityCost, self).__init__(name, object_joint_names, get_object_manip_jacobian, intent)


class WeightedCostCombination(StateCost):
    def __init__(self, name, cost_funcs, weights=None):
        super(WeightedCostCombination, self).__init__(name)

        self.cost_funcs = {cost_func.name: cost_func for cost_func in cost_funcs}

        if weights is None:
            self.weights = {cost_func_name: 1.0/len(cost_funcs) for cost_func_name in self.cost_funcs}
        else:
            self.weights = np.array(weights)/np.sum(weights)

        if len(cost_funcs) != len(self.weights):
            raise ValueError('Cost functions and weights lists must have same length')
        self.required_state_vars = list(set().union(*[cost_func.get_required_state_vars() for cost_func in cost_funcs]))

    def cost(self, state_dict):
        print(state_dict)
        total_cost = 0.0
        for cost_func_name in self.cost_funcs:
            total_cost += self.cost_funcs[cost_func_name].cost(state_dict) * self.weights[cost_func_name]
        return total_cost

    def cost_basis_vector(self, state_dict):
        return np.array([self.cost_funcs[cost_func_name].cost(state_dict) for cost_func_name in self.cost_funcs])

    def jacobian_bases(self, state_dict):
        jacobian_bases_dicts = {}
        for cost_func_name in self.cost_funcs:
            jacobian_bases_dicts[cost_func_name] = self.cost_funcs[cost_func_name].jacobian(state_dict)

        return jacobian_bases_dicts
