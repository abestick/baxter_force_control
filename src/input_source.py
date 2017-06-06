#!/usr/bin/env python
import numpy as np
from control_law import LearnableControlLaw, ControlLaw, WeightedKinematicCostDescent
from abc import ABCMeta, abstractmethod
from kinmodel.track_mocap import MocapTracker, FrameTracker
import kinmodel
from system import Steppable
from collections import deque


class InputSource(Steppable):
    """
    An abstract base class for anything that produces an input to an environment
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def step(self, **kwargs):
        """
        Produces inputs for one iteration
        :return: dict of inputs
        :rtype: dict
        """
        dict()

    @abstractmethod
    def get_input_names(self):
        """
        Returns the keys to the input dictionary that is returned by step
        :return: list of strings
        :rtype: list
        """
        return list()


class InputTracker(InputSource):
    """
    An abstract base class for any InputSource which generates it's inputs from observations
    """
    __metaclass__ = ABCMeta

    def __init__(self, tracker):
        """
        Constructor
        :param tracker: the tracker which tracks the inputs
        """
        self.tracker = tracker


class MocapInputTracker(InputTracker):
    """
    An InputTracker which tracks input from Mocap Data
    """

    def __init__(self, mocap_tracker):
        """
        Constructor
        :param MocapTracker mocap_tracker: The mocap tracker which tracks the input
        """
        assert isinstance(mocap_tracker, MocapTracker), "Must pass a MocapTracker to constructor"
        super(MocapInputTracker, self).__init__(mocap_tracker)

    def step(self):
        """
        Produces a dict of inputs for a single iteration
        :return: dict of inputs
        """
        return self.tracker.step()

    def get_input_names(self):
        return self.tracker.get_state_names()


#TODO: Delete this?
class FrameInputTracker(InputTracker):
    """
    A class which tracks inputs which are frames deduced from some states and a kinematic model
    """

    def __init__(self, frame_tracker, input_frame_name):
        assert isinstance(frame_tracker, FrameTracker), 'frame_tracker must be a FrameTracker object.'
        assert frame_tracker.is_tracked(input_frame_name), "frame_tracker must have frame '%s' attached" % \
                                                           input_frame_name

        self.input_frame_name = input_frame_name

        super(FrameInputTracker, self).__init__(frame_tracker)

    def set_convention(self, convention):
        self.source.set_convention(convention)

    def get_convention(self):
        return self.source.get_convention()

    def step(self, states):
        return self.tracker.get_observables(configs=states, frames=[self.input_frame_name])

    def get_input_names(self):
        return [self.input_frame_name + '_' + element
                for element in kinmodel.Transform.conventions[self.get_convention()]]


class Controller(Steppable):
    """
    A Steppable which implements a control law on the current states
    """

    def __init__(self, control_law):
        """
        Constructor
        :param ControlLaw control_law: The control law used to calculate inputs
        """
        assert isinstance(control_law, ControlLaw), 'control_law must be a ControlLaw object'

        self.control_law = control_law

    def step(self, states):
        """Performs one iteration of control"""
        return self.control_law.compute_control(states)

    def controller_type(self):
        """Returns the type of control law being implemented"""
        return str(type(self.control_law))


class ControllerEstimator(Steppable):
    """
    A Steppable which estimates the parameters of a ControlLaw based on current input and state
    """
    __metaclass__ = ABCMeta

    def __init__(self, control_law):
        """
        Constructor
        :param LearnableControlLaw control_law: Must be a LearnableControlLaw object, the control law to  be learnt
        """
        assert isinstance(control_law, LearnableControlLaw), 'control_law must be a LearnableControlLaw object'

        self.control_law = control_law


class WeightedKinematicCostDescentEstimator(ControllerEstimator):
    """
    A ControllerEstimator for the WeightedKinematicCostDescent ControlLaw. Estimates the weight vector
    """

    def __init__(self, control_law, window_size=None):
        """
        Constructor
        :param WeightedKinematicCostDescent control_law: the WeightedKinematicCostDescent ControlLaw to be learnt
        :param window_size: the amount of data points upon which to retrospectively learn the weights.
                            None means infinite
        """
        assert isinstance(control_law, WeightedKinematicCostDescent), \
            'control_law must be a WeightedKinematicCostDescent object'

        super(WeightedKinematicCostDescentEstimator, self).__init__(control_law)
        self.weight_names = list(self.control_law.get_weights.keys())

        self.window_size = window_size

        self.data_buffer = deque(maxlen=window_size)
        self.buffers = []

    def step(self, states, inputs):
        """
        Steps through one iteration
        :param states: the current states
        :param inputs: the current input
        :return: the estimated weights
        """
        # This is a (U, C) Jacobian object
        input_bases = self.control_law.step_bases(states).reorder(column_names=self.weight_names)

        # Vectorize the inputs so that they match the rows of input_bases
        self.data_buffer.append((input_bases.vectorize(inputs), input_bases.J()))

        # append any registered buffers with the data
        for i in range(len(self.buffers)):
            self.buffers[i].append(inputs, input_bases)

        # if we have enough data to regress over our window size, perform the regression and return the weights
        if len(self.data_buffer) == self.window_size:
            weights = self.learn_weights()
            self.control_law.set_weights(weights)
            return weights

        # otherwise return an empty dict
        else:
            return {}

    def learn_weights(self):
        """
        Regresses over the current acquired data regardless of it's size
        :return: a dictionary of the weights of the WeightedCost
        """
        input_arrays, input_bases_arrays = zip(*self.data_buffer)
        targets = np.concatenate(input_arrays)
        features = np.concatenate(input_bases_arrays, axis=0)

        return {weight_name: val for weight_name, val in zip(self.weight_names, np.linalg.lstsq(features, targets)[0])}

    def register_buffer(self, buffer):
        """
        Registers a data structure with an append function
        :param buffer: any data structure with an append function
        :return: None
        """
        self.buffers.append(buffer)

    def learn_controller(self):
        """
        Learns the weights and returns a Controller object which uses the same ControlLaw and the learnt weights
        :return: Controller
        """
        weights = self.learn_weights()
        self.control_law.set_weights(weights)
        return Controller(self.control_law.copy())
