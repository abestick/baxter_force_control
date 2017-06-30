#!/usr/bin/env python
from abc import ABCMeta, abstractmethod
from motion_costs import StateCost, WeightedCostCombination
from kinmodel.track_mocap import FrameTracker
from kinmodel import Twist
from copy import deepcopy


class ControlLaw(object):
    """
    An abstract base class for a control law which computes controls based on system states and potentially derivatives
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def compute_control(self, states):
        pass

    def copy(self):
        """
        Deep copies itself
        :return: an identical ControlLaw object
        """
        return deepcopy(self)

    @abstractmethod
    def set_parameters(self, **kwargs):
        pass


class LearnableControlLaw(ControlLaw):
    """
    An abstract class for a ControlLaw which is parameterized in a way that these parameters can be learnt.
    Implementations should provide the relevant functions to facilitate learning, but the actual learning code is beyond
    the scope of this family of classes, as it may be able to be done in different ways.
    """
    __metaclass__ = ABCMeta


class CostDescent(ControlLaw):

    def __init__(self, rate, cost_function, inverse_dynamics):
        """
        Constructor
        :param float rate: the rate at which to descend the cost, should be positive for descent, negative for ascent
        :param StateCost cost_function: the cost function which is to be descended
        :param inverse_dynamics: a function handle which maps states and state derivatives to an dict of inputs
        """
        assert isinstance(cost_function, StateCost), 'cost_function must be a StateCost object.'

        self.rate = float(rate)
        self.cost_function = cost_function
        self.inverse_dynamics = inverse_dynamics

    def compute_control(self, states, rate=None):
        """
        Computes the control to apply to descend the cost
        :param states: the current states
        :param rate: optional update of the controller rate
        :return: a dict of controls
        """

        if rate is not None:
            self.rate = rate

        # This is a (1, X) Jacobian
        partial_cost_states = self.cost_function.gradient(states)

        # This is a dict of states which gives the direction of maximal cost descent
        # NOTE: Jacobian facilitates right multiplication with a dictionary.
        #       This is equivalent to the dot product of a horizontal vector with the Jacobian, with ordering handled.
        descending_dynamics = {self.cost_function.name: -self.rate} * partial_cost_states

        return self.inverse_dynamics(states, descending_dynamics)

    def get_parameters(self):
        return {'rate': self.rate}

    def set_parameters(self, rate_dict):
        self.rate = rate_dict['rate']


class KinematicCostDescent(CostDescent):
    """
    A special case of Cost Descent where the input to the system is the pose velocity of a kinematic tree and the states
    are the joint angles of this kinematic tree.
    """

    def __init__(self, rate, cost_function, frame_tracker, input_frame_name, twist_control=False):
        """
        Constructor
        :param float rate: the rate at which to descend the cost, should be positive for descent, negative for ascent
        :param StateCost cost_function: the cost function which is to be descended
        :param frame_tracker: the FrameTracker which will track the input frame
        :param input_frame_name: the frame whose pose velocity is the input to the system
        """
        assert isinstance(frame_tracker, FrameTracker), 'frame_tracker must be a FrameTracker object.'
        assert frame_tracker.is_tracked(input_frame_name), "frame_tracker must have frame '%s' attached" % \
                                                           input_frame_name

        self.frame_tracker = frame_tracker
        self.input_frame_name = input_frame_name
        self.twist_control = twist_control

        super(KinematicCostDescent, self).__init__(rate, cost_function, self.inverse_dynamics)

    def inverse_dynamics(self, states, state_derivatives):
        """Returns the pose velocity of of the input frame for a particular state and state derivative"""
        if self.twist_control:
            return {self.input_frame_name: Twist.from_dict(self.inputs_states_jacobian(states) * state_derivatives)}
        else:
            return self.inputs_states_jacobian(states) * state_derivatives

    def inputs_states_jacobian(self, states):
        """Returns the jacobian of the input frame"""
        return self.frame_tracker.compute_jacobian('root', self.input_frame_name, states)


class WeightedKinematicCostDescent(KinematicCostDescent, LearnableControlLaw):
    """
    A special case of KinematicCostDescent whereby the cost is a linear combination of base costs. The coefficients of
    these base costs can thus be learnt.
    """
    def __init__(self, cost_function, frame_tracker, input_frame_name, twist_control=False):
        """
        Constructor
        :param WeightedCostCombination cost_function: the weighted cost function which is to be descended
        :param frame_tracker: the FrameTracker which will track the input frame
        :param input_frame_name: the frame whose pose velocity is the input to the system
        """

        assert isinstance(cost_function, WeightedCostCombination), \
            'cost_function must be a WeightedCostCombination object.'

        super(WeightedKinematicCostDescent, self).__init__(1.0, cost_function, frame_tracker, input_frame_name,
                                                           twist_control)

    def step_bases(self, states):
        """
        Provides the bases of inputs for each individual cost when unweighted
        :param states: a dict of the current states
        :param weights: optional dict for dynamic weighting
        :return: a Jacobian object whose columns are the inputs for each individual base cost
        """

        # This is a (C, X) Jacobian object with each row being the direction of maximal ascent for that basis cost
        # function.
        partial_cost_vector_states = self.cost_function.gradient_bases(states)

        # This is a (U, X) Jacobian object
        partial_input_states = self.inputs_states_jacobian(states)

        # This is a (U, C) = (U, X) * (X, C) Jacobian object
        # Note we take the transpose and not the pseudo inverse since we are interested in the rows of
        # partial_cost_vector_states which form the bases of the weighted gradient descent
        # u = B(x).pinv * dx_star = J(x) * (w1*dx1_star + w2*dx2_star + ... + wN*dxN_star) = J(x) * Jc_vec(x).T * w
        return partial_input_states * partial_cost_vector_states.T()

    def get_parameters(self):
        """Returns the current weightings"""
        return self.cost_function.get_weights()

    def set_parameters(self, weights):
        """Sets the current weights"""
        self.cost_function.set_weights(weights)

    def scale(self, scale):
        """Scales the weights uniformly"""
        self.cost_function.scale(scale)


