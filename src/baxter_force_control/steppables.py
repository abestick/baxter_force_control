#!/usr/bin/env python
from abc import ABCMeta, abstractmethod
from inspect import getargspec
import time
import numpy as np
import rospy
import tf
from sensor_msgs.msg import JointState, PointCloud
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, TwistStamped, Twist, Vector3, Point32, \
    WrenchStamped, Wrench
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Header
from scipy.optimize import nnls
from control_law import LearnableControlLaw, ControlLaw, WeightedKinematicCostDescent
from collections import deque
from motion_costs import StateCost
import kinmodel
from scipy.linalg import block_diag


class Steppable(object):
    """An abstract base class for all objects that can be stepped through iteratively within a System"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def step(self, **kwargs):
        pass

    def get_step_inputs(self):
        """
        Returns the names of the inputs to the step function
        :return: a list of strings which are the names of the inputs to the step function
        """
        return getargspec(self.step)[0][1:]


class EdgePublisher(Steppable):

    def __init__(self, topic_name, message_type, constructor, bag=None, get_time=None):
        self.pub = rospy.Publisher(topic_name, message_type, queue_size=100)
        self.constructor = constructor
        self.bag = bag
        self.get_time = lambda: rospy.Time(get_time()) if get_time is not None else lambda: None

    def step(self, states):
        if len(states) > 0:
            msg = self.constructor(states)
            self.pub.publish(msg)

            if self.bag is not None:
                self.bag.write(self.pub.name, msg, t=self.get_time())


class JointPublisher(EdgePublisher):

    def __init__(self, topic_name, bag=None, get_time=None):
        super(JointPublisher, self).__init__(topic_name, JointState, self.dict_to_joint_state, bag, get_time)

    def dict_to_joint_state(self, states):
        names, position = zip(*states.items())
        return JointState(header=Header(stamp=rospy.Time.now()), name=names, position=position)


class PosePublisher(EdgePublisher):

    def __init__(self, topic_name, reference_frame, bag=None, get_time=None):
        self.reference_frame = reference_frame
        super(PosePublisher, self).__init__(topic_name, PoseStamped, self.transform_to_pose_msg, bag, get_time)

    def transform_to_pose_msg(self, transform):
        (_, transform), = transform.items()
        pose = transform.pose(convention='quaternion')
        header = Header(stamp=rospy.Time.now(), frame_id=self.reference_frame)
        return PoseStamped(header=header, pose=Pose(position=Point(*pose[:3]), orientation=Quaternion(*pose[3:])))


class TwistPublisher(EdgePublisher):

    def __init__(self, topic_name, reference_frame, bag=None, get_time=None):
        self.reference_frame = reference_frame
        super(TwistPublisher, self).__init__(topic_name, TwistStamped, self.twist_to_twist_msg, bag, get_time)

    def twist_to_twist_msg(self, twist):
        (_, twist), = twist.items()
        header = Header(stamp=rospy.Time.now(), frame_id=self.reference_frame)
        return TwistStamped(header=header, twist=Twist(linear=Vector3(*twist.nu()), angular=Vector3(*twist.omega())))


class WrenchPublisher(EdgePublisher):

    def __init__(self, topic_name, reference_frame, bag=None, get_time=None):
        self.reference_frame = reference_frame
        super(WrenchPublisher, self).__init__(topic_name, WrenchStamped, self.twist_to_wrench_msg, bag, get_time)

    def twist_to_wrench_msg(self, twist):
        (_, twist), = twist.items()
        header = Header(stamp=rospy.Time.now(), frame_id=self.reference_frame)
        return WrenchStamped(header=header, wrench=Wrench(force=Vector3(*twist.nu()), torque=Vector3(*twist.omega())))


class PointCloudPublisher(EdgePublisher):

    def __init__(self, topic_name, reference_frame, bag=None, get_time=None):
        self.reference_frame = reference_frame
        super(PointCloudPublisher, self).__init__(topic_name, PointCloud, self.array_to_point_cloud, bag, get_time)

    def array_to_point_cloud(self, array_dict):
        (_, array), = array_dict.items()
        header = Header(stamp=rospy.Time.now(), frame_id=self.reference_frame)
        return PointCloud(header=header, points=[Point32(*row) for row in array])


class TFPublisher(Steppable):

    def __init__(self, parent_frame, child_frame):
        self.br = tf.TransformBroadcaster()
        self.parent_frame = parent_frame
        self.child_frame = child_frame

    def step(self, transform):
        (_, transform), = transform.items()
        transform = transform
        self.br.sendTransform(transform.p(),
                         tf.transformations.quaternion_from_matrix(transform.R(homog=True)),
                         rospy.Time.now(),
                         self.child_frame,
                         self.parent_frame)


class TopicBagger(Steppable):

    step = None

    def __init__(self, topic_name, message_type, bag, get_time=None, dummy_arg=False):
        self.sub = rospy.Subscriber(topic_name, message_type, self._update)
        self.msg = None
        self.bag = bag
        self.get_time = lambda: rospy.Time(get_time()) if get_time is not None else lambda: None
        self.step = self._step_dummy if dummy_arg else self._step_empty()

    def _update(self, msg):
        self.msg = msg

    def _step_dummy(self, dummy):
        self._step_empty()

    def _step_empty(self):
        if self.msg is not None:
            self.bag.write(self.sub.name, self.msg, t=self.get_time())


class TFBagger(TopicBagger):

    def __init__(self, bag, get_time=None, dummy_arg=False):
        super(TFBagger, self).__init__('tf', TFMessage, bag, get_time, dummy_arg)
        self.transforms = {}

    def _update(self, msg):
        self.transforms.update({'%s.%s' % (t.header.frame_id, t.child_frame_id): t for t in msg.transforms})
        self.msg = msg
        self.msg.transforms = self.transforms.values()


class Constant(Steppable):
    """
    Produces a constant output
    """

    def __init__(self, constant_dict):
        self.values = constant_dict

    def step(self):
        return self.values


class Mask(Steppable):
    """
    Masks its input using a constant dictionary
    """
    def __init__(self, mask_dict):
        self.mask = mask_dict

    def step(self, states):
        # mask the required keys
        states.update(self.mask)
        return states


class Selector(Steppable):
    """
    Extracts a subset from its input, potentially also renaming
    """
    def __init__(self, keys):
        """
        Contructor
        :param keys: a dict or a list of keys. If a dict, keys map to new key names to be assinged on selection
        """
        if isinstance(keys, dict):
            self.keys = keys

        else:
            self.keys = {key:key for key in keys}

    def step(self, states):
        return {new_key: states[key] for key, new_key in self.keys.items() if key in states}


class Modifier(Steppable):

    def __init__(self, mod_func):
        self.mod_func = mod_func

    def step(self, states):
        return {k: self.mod_func(v) for k, v in states.items()}


class Delay(Steppable):
    """
    Adds a delay of a set number of steps
    """

    def __init__(self, num_delays):
        self.delays = num_delays
        self.history = deque([{}]*num_delays, maxlen=num_delays)

    def step(self, states):
        self.history.append(states)
        return self.history.popleft()


class Averager(Steppable):
    """
    Computes the average over a particular time window
    """

    def __init__(self, window_size):
        self.window_size = window_size

        self.data_buffer = deque(maxlen=window_size)

        self.order = None

    def step(self, states):
        # Vectorize the inputs so that they match each other
        self.data_buffer.append(self._vectorize(states))

        # if we have enough data to average over our window size, perform the mean and return it
        if len(self.data_buffer) == self.window_size:
            average = np.mean(self.data_buffer, axis=0)
            return {key: average[i] for i, key in enumerate(self.order)}

        # otherwise return an empty dict
        else:
            return {}

    def _vectorize(self, states):
        """
        Puts the values in the same order as the first
        :param states: the state dict
        :return: a numpy array
        """

        # if the order has not been set yet
        if self.order is None:
            self.order, vectors = zip(*states.items())
            return np.array(vectors, dtype=object)

        else:
            return np.array([states[key] for key in self.order], dtype=object)


class Differentiator(Steppable):
    """
    Differentiates the input
    """
    def __init__(self, fixed_step=None):
        self.fixed_step = fixed_step
        self.last_states = None
        self.last_time = 0

    def step(self, states):

        # if running on a fixed time step, this will assign that, otherwise take the difference from last iteration
        # if there is no fixed and this is the first step, step will be the current time, but will not be used this step
        step = time.time() - self.last_time if self.fixed_step is None else self.fixed_step

        # if we are not on the first step compute the derivatives
        if self.last_states is not None:
            derivs = {state_name: (states[state_name] - self.last_states[state_name]) /
                                step for state_name in states}

        # otherwise set to zero
        else:
            derivs = {state_name: states[state_name] - states[state_name] for state_name in states}

        # update the delay attribute and the last time
        self.last_states = states.copy()
        self.last_time += step

        return derivs


class MeasurementSource(Steppable):
    """
    An abstract base class for steppables that provide measurements
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def step(self):
        pass


class MocapMeasurement(MeasurementSource):
    """
    Produces measurements from a mocap source
    """
    def __init__(self, mocap_source, source_name):
        self._mocap_source = mocap_source
        self._mocap_stream = mocap_source.get_stream()
        self._source_name = source_name

    def step(self):
        frame, timestamps = self._mocap_stream.read()
        return {self._source_name: frame}


class Estimator(Steppable):
    """
    An abstract base class for
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def step(self, measurement):
        pass


class MocapFrameEstimator(Estimator):
    """
    Estimates a transform to a particular tracked frame in the mocap data
    """
    def __init__(self, mocap_frame_tracker, tracked_frame_name, wrt_world=True):
        self._mocap_frame_tracker = mocap_frame_tracker
        self._frame_name = tracked_frame_name
        self._wrt_world = wrt_world

    def step(self, measurement):
        # make sure the dictionary has a single value which is the mocap frame
        assert len(measurement)==1, 'can only pass a single frame to the MocapFrameEstimator step function'

        # extract the frame and return the transform estimate
        (_, frame), = measurement.items()

        transform = self._mocap_frame_tracker.process_frame(frame)
        if transform is not None:
            if self._wrt_world:
                transform = transform.inv()

        else:
            transform = kinmodel.Transform()

        return {self._frame_name: transform}


class Transformer(Steppable):
    """
    Transforms geometric primitives with a Transform object
    """
    def step(self, transform, primitives):
        assert len(transform)==1, 'can only pass a single transform to the Transformer step function'

        # extract the transform
        (_, transform), = transform.items()
        return_dict = {}

        # apply to each of the primitives
        for primitive_name, primitive in primitives.items():
            return_dict[primitive_name] = transform.transform(primitive)

        return return_dict


class SystemState(Steppable):
    """
    An abstract base class for classes that provide the states of a system
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self._states = {}

    def get_state_names(self):
        return self._states.keys()

    @abstractmethod
    def _step_states(self, **kwargs):
        pass


class MocapSystemState(SystemState, Estimator):
    """
    Estimates the state of a system based on mocap data
    """
    def __init__(self, mocap_trackers):
        self._mocap_trackers = mocap_trackers
        super(MocapSystemState, self).__init__()

    def step(self, measurement):
        self._step_states(measurement)
        return self._states.copy()

    def _step_states(self, measurement):
        (_, frame), = measurement.items()
        for mocap_tracker in self._mocap_trackers:
            self._states.update(mocap_tracker.process_frame(frame))


class CostCalculator(Steppable):

    def __init__(self, state_cost):
        self.state_cost = state_cost
        assert isinstance(self.state_cost, StateCost), 'state_cost must be a StateCost object.'

    def step(self, states):
        return {self.state_cost.name: self.state_cost.cost(states)}


class Controller(Steppable):
    """
    A Steppable which implements a control law on the current states
    """

    def __init__(self, control_law, output_type=None):
        """
        Constructor
        :param ControlLaw control_law: The control law used to calculate inputs
        """
        assert isinstance(control_law, ControlLaw), 'control_law must be a ControlLaw object'

        if output_type is not None:
            assert isinstance(output_type, kinmodel.GeometricPrimitive)

        self.output_type = output_type

        self.control_law = control_law

    def step(self, states):
        """Performs one iteration of control"""
        return self.control_law.compute_control(states)

    def controller_type(self):
        """Returns the type of control law being implemented"""
        return str(type(self.control_law))


class DynamicController(Controller):
    """
    A parameterized controller whose parameters may change over time
    """
    def __init__(self, control_law, persistent_control=False):
        self.persistent_control = persistent_control
        super(DynamicController, self).__init__(control_law)

    def step(self, states, parameters):
        if len(parameters) == 0 and not self.persistent_control:
            return {}

        else:
            self.control_law.set_parameters(parameters)
            return super(DynamicController, self).step(states)


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

    def __init__(self, control_law, reference_frame, window_size=None, non_negative=False):
        """
        Constructor
        :param WeightedKinematicCostDescent control_law: the WeightedKinematicCostDescent ControlLaw to be learnt
        :param window_size: the amount of data points upon which to retrospectively learn the weights.
                            None means infinite
        """
        assert isinstance(control_law, WeightedKinematicCostDescent), \
            'control_law must be a WeightedKinematicCostDescent object'

        super(WeightedKinematicCostDescentEstimator, self).__init__(control_law)
        self.weight_names = list(self.control_law.get_parameters().keys())

        self.window_size = window_size if window_size is not None else -1

        self.data_buffer = deque(maxlen=window_size)
        self.buffers = []

        self.least_squares = nnls if non_negative else np.linalg.lstsq
        self.reference_frame = reference_frame

    def step(self, states, input_twist, base_twist, base_transform):
        """
        Steps through one iteration
        :param states: the current states
        :param input_twist: the current input
        :param base_twist: the pose velocity of the object base frame
        :return: the estimated weights
        """

        (input_name, input_twist), = input_twist.items()
        (base_name, base_twist), = base_twist.items()
        (_, base_transform), = base_transform.items()
        base_rot = base_transform.R(homog=False)
        base_rot_six_d = block_diag(base_rot, base_rot)

        # This is a (U, C) Jacobian object showing the twist of max increase for each basis expressed in the base frame
        input_bases = self.control_law.step_bases(states).reorder(row_names=input_twist.names(prefix=input_name+'_'),
                                                                  column_names=self.weight_names)

        # We want to express each twist in the reference_frame, so we take the relative twist
        input_bases.in_place_dot(base_rot_six_d, left=True)
        input_bases -= base_twist.xi().reshape((-1, 1))

        # Vectorize the inputs so that they match the rows of input_bases
        self.data_buffer.append((input_twist.xi(), input_bases.J()))

        # append any registered buffers with the data
        for i in range(len(self.buffers)):
            self.buffers[i].append(input_twist, input_bases)

        # if we have enough data to regress over our window size, perform the regression and return the weights
        if len(self.data_buffer) == self.window_size:
            weights = self.learn_weights()
            self.control_law.set_parameters(weights)
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
        targets = np.concatenate(input_arrays).squeeze()
        features = -np.concatenate(input_bases_arrays, axis=0)
        fit = self.least_squares(features, targets)
        if len(fit[1]) == 0:
            return {weight_name: 0.0 for weight_name in self.weight_names + ['adjusted_r_squared']}

        result = {weight_name: val for weight_name, val in zip(self.weight_names, fit[0])}
        n, k = features.shape
        result['adjusted_r_squared'] = 1 - (1 - float(fit[1]) / np.sum((targets-targets.mean()) ** 2)) * (n - 1) / (n - k - 1)
        return result

    def register_buffer(self, buffer):
        """
        Registers a data structure with an append function
        :param buffer: any data structure with an append function
        :return: None
        """
        self.buffers.append(buffer)

    def get_controller(self):
        return Controller(self.control_law.copy())

    def learn_controller(self):
        """
        Learns the weights and returns a Controller object which uses the same ControlLaw and the learnt weights
        :return: Controller
        """
        weights = self.learn_weights()
        self.control_law.set_parameters(weights)
        return self.get_controller()

    def get_window_size(self):
        return self.window_size
