from abc import ABCMeta, abstractmethod

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