#!/usr/bin/env python
from abc import ABCMeta, abstractmethod
from inspect import getargspec
import time
import numpy as np
import rospy
import tf
import quadprog
from tf.transformations import quaternion_matrix
from sensor_msgs.msg import JointState, PointCloud
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, TwistStamped, Twist, Vector3, Point32, \
    WrenchStamped, Wrench, Vector3Stamped, PointStamped
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Header
from scipy.optimize import nnls
from control_law import LearnableControlLaw, ControlLaw, WeightedKinematicCostDescent
from motion_costs import StateCost, WeightedCostCombination
from collections import deque
from motion_costs import StateCost
import kinmodel
import pandas as pd
from kinmodel.syms import get_jacobian_func


def pad_list(l, n, filler=None):
    l += [filler] * (n-len(l))
    return l


def transform_msg_to_transform(transform_msg, inv=False):
    quat = (transform_msg.transform.rotation.x, transform_msg.transform.rotation.y,
            transform_msg.transform.rotation.z, transform_msg.transform.rotation.w)
    trans = (transform_msg.transform.translation.x, transform_msg.transform.translation.y,
             transform_msg.transform.translation.z)
    homog = quaternion_matrix(quat)
    homog[:3, 3] = trans

    T = kinmodel.Transform(homog, transform_msg.header.frame_id, transform_msg.child_frame_id)

    return T.inv() if inv else T


def twist_msg_to_twist(msg):
    omega = (msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z)
    nu = (msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z)
    frames = msg.header.frame_id.split('/')
    reference_frame, target, observation_frame, reference_point = frames
    return kinmodel.Twist(omega=omega, nu=nu, reference_frame=reference_frame,
                          target=target, observation_frame=observation_frame,
                          reference_point=reference_point)


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


class Debugger(Steppable):

    def __init__(self, func=None):
        self.func = func

    def step(self, output):
        print(output)
        if self.func:
            self.func(output)
        raw_input('Press ENTER to continue\n')


class EdgePublisher(Steppable):

    def __init__(self, topic_name, message_type, constructor, bag=None, get_time=None):
        self.pub = rospy.Publisher(topic_name, message_type, queue_size=100)
        self.constructor = constructor
        self.bag = bag
        self.get_time = rospy.Time.now if get_time is None else lambda: rospy.Time(get_time())

    def step(self, states):
        if len(states) > 0:
            msg = self.constructor(states)
            self.pub.publish(msg)

            if self.bag is not None:
                self.bag.write(self.pub.name, msg, t=self.get_time())


class JointPublisher(EdgePublisher):

    def __init__(self, topic_name, bag=None, get_time=None):
        super(JointPublisher, self).__init__(topic_name, JointState, self.dict_to_joint_state, bag, get_time)
        self.names = []

    def dict_to_joint_state(self, states):
        self.names += list(set(states) - set(self.names))
        position = [states[name] for name in self.names if name in states]
        return JointState(header=Header(stamp=self.get_time()), name=self.names, position=position)


class JointCommandPublisher(EdgePublisher):
    def __init__(self, topic_name, bag=None, get_time=None):
        super(JointCommandPublisher, self).__init__(topic_name, JointState, self.command_to_joint_state, bag, get_time)

    def command_to_joint_state(self, states):
        (_, command_msg), = states.items()
        msg = JointState()
        msg.name = command_msg.names
        if command_msg.mode == 2:
            msg.velocity = command_msg.command

        else:
            msg.position = command_msg.command
        return msg


class PosePublisher(EdgePublisher):

    def __init__(self, topic_name, reference_frame, bag=None, get_time=None):
        self.reference_frame = reference_frame
        super(PosePublisher, self).__init__(topic_name, PoseStamped, self.transform_to_pose_msg, bag, get_time)

    def transform_to_pose_msg(self, transform):
        (_, transform), = transform.items()
        pose = transform.pose(convention='quaternion')
        header = Header(stamp=self.get_time(), frame_id=self.reference_frame)
        return PoseStamped(header=header, pose=Pose(position=Point(*pose[:3]), orientation=Quaternion(*pose[3:])))


class PointPublisher(EdgePublisher):

    def __init__(self, topic_name, reference_frame, bag=None, get_time=None):
        self.reference_frame = reference_frame
        super(PointPublisher, self).__init__(topic_name, WrenchStamped, self.v3_to_point_msg, bag, get_time)

    def v3_to_point_msg(self, vector):
        (_, vector), = vector.items()
        header = Header(stamp=self.get_time(), frame_id=self.reference_frame)
        return WrenchStamped(header=header, wrench=Wrench(force=Vector3(*vector.q())))


class TwistPublisher(EdgePublisher):

    def __init__(self, topic_name, bag=None, get_time=None, all_frames=True):
        super(TwistPublisher, self).__init__(topic_name, TwistStamped, self.twist_to_twist_msg, bag, get_time)
        self.all_frames = all_frames

    def twist_to_twist_msg(self, twist):
        (_, twist), = twist.items()
        frames = '/'.join([twist.reference_frame(), twist.target(), twist.observation_frame(),
                           twist.reference_point()]) if self.all_frames else twist.reference_frame()
        header = Header(stamp=self.get_time(), frame_id=frames)
        return TwistStamped(header=header, twist=Twist(linear=Vector3(*twist.nu()), angular=Vector3(*twist.omega())))


class Vector3Publisher(EdgePublisher):
    def __init__(self, topic_name, bag=None, get_time=None, all_frames=True):
        super(Vector3Publisher, self).__init__(topic_name, Vector3Stamped, self.vector_to_vector3_msg, bag, get_time)
        self.all_frames = all_frames

    def vector_to_vector3_msg(self, vector):
        (_, vector), = vector.items()
        frames = '/'.join([vector.reference_frame(), vector.origin(), vector.target()]) \
            if self.all_frames else vector.reference_frame()
        header = Header(stamp=self.get_time(), frame_id=frames)
        return Vector3Stamped(header=header, vector=Vector3(*vector.q()))


class WrenchPublisher(EdgePublisher):

    def __init__(self, topic_name, reference_frame, bag=None, get_time=None):
        self.reference_frame = reference_frame
        super(WrenchPublisher, self).__init__(topic_name, WrenchStamped, self.twist_to_wrench_msg, bag, get_time)

    def twist_to_wrench_msg(self, twist):
        (_, twist), = twist.items()
        header = Header(stamp=self.get_time(), frame_id=self.reference_frame)
        return WrenchStamped(header=header, wrench=Wrench(force=Vector3(*twist.nu()), torque=Vector3(*twist.omega())))


class PointCloudPublisher(EdgePublisher):

    def __init__(self, topic_name, reference_frame, bag=None, get_time=None):
        self.reference_frame = reference_frame
        super(PointCloudPublisher, self).__init__(topic_name, PointCloud, self.array_to_point_cloud, bag, get_time)

    def array_to_point_cloud(self, array_dict):
        (_, array), = array_dict.items()
        header = Header(stamp=self.get_time(), frame_id=self.reference_frame)
        return PointCloud(header=header, points=[Point32(*row) for row in array])


class TFPublisher(Steppable):

    def __init__(self, get_time=None):
        self.br = tf.TransformBroadcaster()
        self.get_time = rospy.Time.now if get_time is None else lambda: rospy.Time(get_time())

    def step(self, transform):
        (_, transform), = transform.items()
        transform = transform
        self.br.sendTransform(transform.p(),
                              tf.transformations.quaternion_from_matrix(transform.R(homog=True)),
                              self.get_time(),
                              transform.target(),
                              transform.reference_frame())


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
        self.transforms.update({'%s_%s' % (t.header.frame_id, t.child_frame_id): t for t in msg.transforms})
        self.msg = msg
        self.msg.transforms = self.transforms.values()


class Iterator(Steppable):

    def __init__(self, iterable, name):
        self.name = name
        self.iterable = iterable
        self.generator = iter(self.iterable)

    def reset(self):
        self.generator = iter(self.iterable)

    def step(self):
        return {self.name: self.generator.next()}


class BagReader(Steppable):

    def __init__(self, bag, publish=False):
        self.bag = bag
        self.messages = []
        self.pubs = {}
        self.publish = publish
        last_time = -1
        for topic_name, message, time_stamp in self.bag.read_messages():
            name = topic_name[1:]
            step_time = time_stamp.to_sec()

            if self.publish:
                if topic_name not in self.pubs:
                    self.pubs[topic_name] = rospy.Publisher(topic_name, type(message), queue_size=100)

            if step_time != last_time:
                self.messages.append({name: message})

            else:
                self.messages[-1][name] = message

            last_time = step_time

        self.message_generator = iter(self.messages)

    def close(self):
        self.bag.close()

    def step(self):
        msgs = self.message_generator.next()
        if self.publish:
            for topic, msg in msgs.items():
                self.pubs[topic].publish(msg)
        return msgs

    def reset(self):
        self.message_generator = iter(self.messages)

    def __getitem__(self, s):
        if isinstance(s, basestring):
            return self.get_topic(s)
        return self.messages.__getitem__(s)

    def __len__(self):
        return len(self.messages)

    def get_messages(self):
        return self.messages[:]

    def get_topic(self, topic):
        return [msgs[topic] for msgs in self.messages]


class TFMsgFrameReader(Steppable):

    def __init__(self, parent, child, tf_topic_name='tf', inv=False):
        self.parent = parent
        self.child = child
        self.tf_topic_name = tf_topic_name
        self.inv = inv

    def step(self, msgs):
        tf_msg = msgs[self.tf_topic_name]
        transform_msg = next(transform for transform in tf_msg.transforms
                             if transform.header.frame_id == self.parent and transform.child_frame_id==self.child)

        return {self.child: transform_msg_to_transform(transform_msg, self.inv)}


class MsgReader(Steppable):

    def __init__(self, topic_name, converter):
        self.topic_name = topic_name
        self.converter = converter

    def step(self, msgs):
        msg = msgs[self.topic_name]
        return self.converter(msg)


class JointStateMsgReader(MsgReader):

    def __init__(self, topic_name):
        super(JointStateMsgReader, self).__init__(topic_name, self._joint_state_converter)

    def _joint_state_converter(self, msg):
        return {name: value for name, value in zip(msg.name, msg.position)}


class TwistMsgReader(MsgReader):

    def __init__(self, topic_name):
        super(TwistMsgReader, self).__init__(topic_name, self._twist_converter)

    def _twist_converter(self, msg):
        return {self.topic_name: twist_msg_to_twist(msg)}


class Vector3MsgReader(MsgReader):

    def __init__(self, topic_name):
        super(Vector3MsgReader, self).__init__(topic_name, self._vector3_converter)

    def _vector3_converter(self, msg):
        v = (msg.vector.x, msg.vector.y, msg.vector.z)
        frames = msg.header.frame_id.split('/')
        reference_frame, origin, target, = frames
        return {self.topic_name: kinmodel.Vector(np.append(v, 0), reference_frame=reference_frame,
                                                 origin=origin, target=target)}


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


class Mux(Steppable):

    def __init__(self, first_name, second_name):
        self.first_name = first_name
        self.second_name = second_name

    def step(self, first, second):
        return {self.first_name: first.values()[0], self.second_name: second.values()[0]}


class MultiMuxBase(Steppable):

    __metaclass__ = ABCMeta

    def __init__(self, channels):
        channel_step_funcs = [None, self.step, self._step_2, self._step_3, self._step_4, self._step_5, self._step_6,
                              self._step_7, self._step_8, self._step_9, self._step_10, self._step_11, self._step_12]

        self.step = channel_step_funcs[channels]
        assert self.step is not None, 'Must provide a channel number between 1 and %d' % len(channel_step_funcs)

    def step(self, first):
        return self._step_base(first)

    @abstractmethod
    def _step_base(self, *dicts):
        pass

    def _step_2(self, first, second):
        return self._step_base(first, second)

    def _step_3(self, first, second, third):
        return self._step_base(first, second, third)

    def _step_4(self, first, second, third, fourth):
        return self._step_base(first, second, third, fourth)

    def _step_5(self, first, second, third, fourth, fifth):
        return self._step_base(first, second, third, fourth, fifth)

    def _step_6(self, first, second, third, fourth, fifth, sixth):
        return self._step_base(first, second, third, fourth, fifth, sixth)

    def _step_7(self, first, second, third, fourth, fifth, sixth, seventh):
        return self._step_base(first, second, third, fourth, fifth, sixth, seventh)

    def _step_8(self, first, second, third, fourth, fifth, sixth, seventh, eighth):
        return self._step_base(first, second, third, fourth, fifth, sixth, seventh, eighth)

    def _step_9(self, first, second, third, fourth, fifth, sixth, seventh, eighth, ninth):
        return self._step_base(first, second, third, fourth, fifth, sixth, seventh, eighth, ninth)

    def _step_10(self, first, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth):
        return self._step_base(first, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth)

    def _step_11(self, first, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth, eleventh):
        return self._step_base(first, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth, eleventh)

    def _step_12(self, first, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth, eleventh, twelfth):
        return self._step_base(first, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth, eleventh, twelfth)


class MultiMux(MultiMuxBase):

    def __init__(self, names):
        channels = len(names)
        super(MultiMux, self).__init__(channels)
        self.names = names

    def _step_base(self, *dicts):
        vals = [d.values()[0] for d in dicts]
        names = [d.keys()[0] for d in dicts] if self.names is None else self.names
        return dict(zip(names, vals))


class Merger(MultiMuxBase):

    def __init__(self, suffixes):
        channels = len(suffixes)
        super(Merger, self).__init__(channels)
        self.suffixes = suffixes

    def _step_base(self, *dicts):
        ret = {}
        for d, s in zip(dicts, self.suffixes):
            new_d = {k+s: v for k, v in d.items()}
            ret.update(new_d)

        return ret


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


class ExponentialFilter(Steppable):

    def __init__(self, alpha, beta=0):
        self.alpha = alpha
        self.beta = beta
        self.last_s = None
        self.last_b = None

    def step(self, states):
        if self.last_s is None:
            self.last_s = states
            self.last_b = {k: 0.0*v for k, v in states.items()}
            return states

        s = {}
        b = {}
        for k in states:
            s[k] = self.alpha*states[k] + (1 - self.alpha)*(self.last_s[k] + self.last_b[k])
            b[k] = self.beta*(s[k] - self.last_s[k]) + (1 - self.beta)*(self.last_b[k])

        self.last_s = s
        self.last_b = b

        return s


class Differentiator(Steppable):
    """
    Differentiates the input
    """
    def __init__(self, fixed_step=None, noise_thresh=0):
        self.fixed_step = fixed_step
        self.last_states = None
        self.last_time = 0
        self.thresh = noise_thresh

    def step(self, states):

        # if running on a fixed time step, this will assign that, otherwise take the difference from last iteration
        # if there is no fixed and this is the first step, step will be the current time, but will not be used this step
        step = time.time() - self.last_time if self.fixed_step is None else self.fixed_step

        # if we are not on the first step compute the derivatives
        if self.last_states is not None:
            derivs = {}
            for state_name in states:
                d = (states[state_name] - self.last_states[state_name]) / step
                derivs[state_name] = d if abs(d) > self.thresh else 0.0*d

        # otherwise set to zero
        else:
            derivs = {state_name: val - val for state_name, val in states.items()}

        # update the delay attribute and the last time
        self.last_states = states.copy()
        self.last_time += step

        return derivs


class Adder(Steppable):

    def step(self, first, second):
        (first_name, first), = first.items()
        (second_name, second), = second.items()
        return {'%s_plus_%s' % (first_name, second_name): first + second}


class Subtractor(Steppable):

    def step(self, first, second):
        (first_name, first), = first.items()
        (second_name, second), = second.items()
        return {'%s_minus_%s' % (first_name, second_name): first - second}


class Multiplier(Steppable):

    def step(self, first, second):
        (first_name, first), = first.items()
        (second_name, second), = second.items()
        return {'%s_mul_%s' % (first_name, second_name): first * second}


class Divider(Steppable):

    def __init__(self, first, second):
        self.first = first
        self.second = second

    def step(self, states):
        if not {self.first, self.second}.issubset(states):
            return {}
        first = states[self.first]
        second = states[self.second]
        if first + second == 0:
            result = -1
        else:
            result = first / (first + second)
        return {'%s_div_%s' % (self.first, self.second): result}


class KinTreeFrameEstimator(Steppable):

    def __init__(self, kin_tree_tracker, frame_name):
        self.kin_tree_tracker = kin_tree_tracker
        self.frame_name = frame_name


class Magnitude(Steppable):

    def __init__(self, element_wise, pos_only=False, name='mag', keys=None):
        self.pos_only = pos_only
        self.element_wise = element_wise
        self.name = name
        self.keys = keys

    def step(self, states):
        if self.keys is not None:
            states = {k: states[k] for k in self.keys}

        if self.pos_only:
            states = {k: v.trans() if hasattr(v, 'trans') else v for k, v in states.items()}

        mags = {k: np.linalg.norm(v) for k, v in states.items()}

        return mags if self.element_wise else {self.name: np.linalg.norm(mags.values())}


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

    def __init__(self, inv=False):
        self.inv = inv
        # super(Transformer, self).__init__()

    def step(self, transform, primitives):
        assert len(transform)==1, 'can only pass a single transform to the Transformer step function'

        # extract the transform
        (_, transform), = transform.items()
        if self.inv:
            transform = transform.inv()
        return_dict = {}

        # apply to each of the primitives
        for primitive_name, primitive in primitives.items():
            return_dict[primitive_name] = transform * primitive

        return return_dict


class RelativeTwist(Steppable):

    def step(self, first, second):
        final_twist = first.values()[0] + second.values()[0]
        final_twist_name = '%s_%s' % (final_twist.observation_frame(), final_twist.reference_point())
        return {final_twist_name: final_twist}


class TwistTransformer(Steppable):

    def __init__(self, inv=False, neg=False, rot_trans='both'):
        self.inv = inv
        self.neg = neg
        self.rot_trans = rot_trans

        if rot_trans == 'both':
            self.step = self.step_both

        elif rot_trans == 'rot':
            self.step = self.step_rot

        elif rot_trans == 'trans':
            self.step = self.step_trans

    def step(self, **kwargs):
        raise NotImplementedError()

    def step_rot(self, twist, rotation):
        (twist_name, twist), = twist.items()
        (_, rotation), = rotation.items()

        rotation = rotation.rot()

        if self.inv:
            rotation = rotation.T()

        return {twist_name: rotation.apply_rotation(twist)}

    def step_trans(self, twist, translation):
        (twist_name, twist), = twist.items()
        (_, translation), = translation.items()

        translation = translation.trans()

        if self.neg:
            translation = -translation

        if translation.reference_frame() == twist.reference_frame():
            return {twist_name: translation.apply_translation(twist)}

        else:
            raise ValueError('Translation reference frame (%s) should be that of the twist (%s) '%
                             (translation.reference_frame(), twist.reference_frame()))

    def step_both(self, twist, rotation, translation):
            (twist_name, twist), = twist.items()
            (_, rotation), = rotation.items()
            (_, translation), = translation.items()

            rotation = rotation.rot()
            translation = translation.trans()

            if self.inv:
                rotation = rotation.T()

            if self.neg:
                translation = -translation

            # if translation is in the rotation frames reference frame then it should be performed last
            if translation.reference_frame() == rotation.reference_frame():
                return {twist_name: translation.apply_translation(rotation.apply_rotation(twist))}

            elif translation.reference_frame() == twist.reference_frame():
                return {twist_name: rotation.apply_rotation(translation.apply_translation(twist))}

            else:
                raise ValueError('Translation reference frame (%s) should be either that of the twist (%s) '
                                 'or that of the rotation (%s)' %
                                 (translation.reference_frame(), twist.reference_frame(), rotation.reference_frame()))


class TwistReverser(Steppable):

    def step(self, twist, translation):

        (twist_name, twist) = twist.items()
        (_, translation) = translation.items()

        translation = translation.trans()

        assert translation.reference_frame() == twist.reference_frame()
        assert translation.target() == twist.reference_point()
        assert translation.origin() == twist.observation_frame()

        return {'neg_%s' % twist_name: translation.apply_translation(-twist)}


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


class CostGradient(Steppable):

    def __init__(self, cost_function, neg=True):
        assert isinstance(cost_function, StateCost), 'cost_function must be a StateCost object'

        self.cost_function = cost_function
        self.neg = neg

    def step(self, states):
        gradient = -self.cost_function.gradient(states) if self.neg else self.cost_function.gradient(states)
        return gradient.column_dict()


class WeightedCostGradients(CostGradient):

    def __init__(self, cost_function, neg=True):

        assert isinstance(cost_function, WeightedCostCombination), \
            'control_law must be a WeightedKinematicCostDescent object'

        super(WeightedCostGradients, self).__init__(cost_function, neg)
        self.cost_function = cost_function

        self.weight_names = list(self.cost_function.get_basis_names())

    def step(self, states):
        """
        Steps through one iteration
        :param states: the current states
        :return: the gradients
        """

        # This is a (C, X) Jacobian object with each row being the direction of maximal ascent for that basis cost
        # function.
        gradients = -self.cost_function.gradient_bases(states).T() \
            if self.neg else self.cost_function.gradient_bases(states).T()
        return {self.cost_function.name + '_gradients': gradients}


class JacobianOperator(Steppable):

    def __init__(self, kin_tree_tracker, base_frame, manip_frame, inv=False, position_only=False, theano=True):
        if theano:
            self._compute_jacobian = get_jacobian_func(kin_tree_tracker.kin_tree(), base_frame, manip_frame)[0]
        else:
            self.kin_tree_tracker = kin_tree_tracker
            self._compute_jacobian = self._compute_jacobian_tree
        self.base_frame = base_frame
        self.manip_frame = manip_frame
        self.inv = inv
        self.position_only = position_only

    def _compute_jacobian_tree(self, **states):
        return self.kin_tree_tracker.compute_jacobian(self.base_frame, self.manip_frame, states)

    def step(self, states, velocities):
        jacobian = self._compute_jacobian(**states)

        if self.position_only:
            jacobian = jacobian.position_only()

        if self.inv:
            jacobian = jacobian.pinv()
            (_, vel_primitive), = velocities.items()
            return jacobian * vel_primitive

        else:
            return {self.manip_frame: jacobian * velocities}


class JacobianCalculator(Steppable):

    def __init__(self, kin_tree_tracker, base_frame, manip_frame, inv=False, position_only=False, theano=True):
        if theano:
            self._compute_jacobian = get_jacobian_func(kin_tree_tracker.kin_tree(), base_frame, manip_frame)[0]
        else:
            self.kin_tree_tracker = kin_tree_tracker
        self.base_frame = base_frame
        self.manip_frame = manip_frame
        self.inv = inv
        self.position_only = position_only

    def _compute_jacobian_tree(self, **states):
        return self.kin_tree_tracker.compute_jacobian(self.base_frame, self.manip_frame, states)

    def step(self, states):
        jacobian = self.kin_tree_tracker.compute_jacobian(self.base_frame, self.manip_frame, states)

        if self.position_only:
            jacobian = jacobian.position_only()

        if self.inv:
            jacobian = jacobian.pinv()

        return {'J_%s_%s' % (self.base_frame, self.manip_frame): jacobian}


class WeightedJacobianOpoerator(JacobianOperator):

    def __init__(self, kin_tree_tracker, base_frame, manip_frame, position_only=False):
        super(WeightedJacobianOpoerator, self).__init__(kin_tree_tracker, base_frame, manip_frame, False, position_only)

    def step(self, states, jacobian):
        # This is a (X, C) jacobian whose columns are joint velocities of maximum ascent
        (gradients_name, jacobian), = jacobian.items()

        # This is a (6/3, X) jacobian which produces twists/velocities from joint angle velocities
        kin_jacobian = self.kin_tree_tracker.compute_jacobian(self.base_frame, self.manip_frame, states)

        if self.position_only:
            kin_jacobian = kin_jacobian.position_only()

        name = '_'.join(gradients_name.split('_')[:-1]) + '_twists'

        return {name : kin_jacobian * jacobian}


class Regressor(Steppable):

    def __init__(self, window_sizes, method='LS'):
        solver_dict = {'LS': self._fit_ls,
                       'NNLS': self._fit_nnls,
                       'QP': self._fit_quad_prog}
        self.window_sizes = window_sizes

        self.data_buffers = [deque(maxlen=window_size) for window_size in window_sizes]
        self.predictor_names = None

        self.least_squares = solver_dict[method]

    @staticmethod
    def _fit_ls(features, targets):
        fit = np.linalg.lstsq(features, targets)
        if len(fit[1]) == 0:
            return None, None
        return fit[0], fit[1][0]

    @staticmethod
    def _fit_nnls(features, targets):
        if np.linalg.matrix_rank(features) < 1:
            return None, None
        fit = nnls(features, targets)
        return fit

    @staticmethod
    def _fit_quad_prog(features, targets):
        P = features.T.dot(features)
        neg_q = features.T.dot(targets)
        G = np.identity(features.shape[1])
        h = np.zeros(features.shape[1])
        A = np.ones((1, features.shape[1]))
        b = np.array([1])
        fit = quadprog_solve_qp(P, neg_q, G, h, A, b)
        return fit[0], fit[1]

    def step(self, predictors, observation):
        (_, observation), = observation.items()

        if self.predictor_names is None:
            self.predictor_names = predictors.keys()

        y = np.array(observation).squeeze()
        X = np.vstack([np.array(predictors[predictor_name]).squeeze() for predictor_name in self.predictor_names]).T

        ret = {}

        for data_buffer in self.data_buffers:
            data_buffer.append((y, X))

            # if we have enough data to regress over our window size, perform the regression and return the weights
            if len(data_buffer) == data_buffer.maxlen:
                ret.update(self.regress(data_buffer))
        return ret

    def regress(self, data_buffer):
        """
        Regresses over the current acquired data regardless of it's size
        :return: a dictionary of the weights of the WeightedCost
        """
        window_size = data_buffer.maxlen
        Y, XX = zip(*data_buffer)
        targets = np.concatenate(Y).squeeze()
        features = -np.concatenate(XX, axis=0)
        weights, residual = self.least_squares(features, targets)
        if residual is None:
            return {weight_name + '_%d' % window_size: 0.0 for weight_name in self.predictor_names + ['adjusted_r_squared']}

        result = {weight_name + '_%d' % window_size: val for weight_name, val in zip(self.predictor_names, weights)}
        n, k = features.shape
        result['adjusted_r_squared_%d' % window_size] = residual
        # result['adjusted_r_squared_%d' % window_size] = 1 - (1 - residual / np.sum((targets-targets.mean()) ** 2)) * (n - 1) / (n - k - 1) if np.sum((targets-targets.mean()) ** 2) > 0 else residual
        return result


class RegressorBase(Steppable):

    def __init__(self, window_sizes, method='LS'):
        solver_dict = {'LS': self._fit_ls,
                       'NNLS': self._fit_nnls,
                       'QP': self._fit_quad_prog}
        self.window_sizes = window_sizes

        self.data_buffers = [deque(maxlen=window_size) for window_size in window_sizes]
        self.predictor_names = None

        self.least_squares = solver_dict[method]

    @staticmethod
    def _fit_ls(features, targets):
        fit = np.linalg.lstsq(features, targets)
        if len(fit[1]) == 0:
            return None, None
        return fit[0], fit[1][0]

    @staticmethod
    def _fit_nnls(features, targets):
        if np.linalg.matrix_rank(features) < 1:
            return None, None
        fit = nnls(features, targets)
        return fit

    @staticmethod
    def _fit_quad_prog(features, targets):
        P = features.T.dot(features)
        neg_q = features.T.dot(targets)
        G = np.identity(features.shape[1])
        h = np.zeros(features.shape[1])
        A = np.ones((1, features.shape[1]))
        b = np.array([1])
        fit = quadprog_solve_qp(P, neg_q, G, h, A, b)
        return fit[0], fit[1]

    def step(self, bases):
        if self.predictor_names is None:
            self.predictor_names = bases['w_order']

        A = bases['A']
        b = bases['b']

        if bases['w_order'] != self.predictor_names:
            old_A = A.copy()
            for i, name in enumerate(self.predictor_names):
                col = bases['w_order'].index(name)
                A[:, i] = old_A[:, col]

        ret = {}

        for data_buffer in self.data_buffers:
            data_buffer.append((b, A))

            # if we have enough data to regress over our window size, perform the regression and return the weights
            if len(data_buffer) == data_buffer.maxlen:
                ret.update(self.regress(data_buffer))
        return ret

    def regress(self, data_buffer):
        """
        Regresses over the current acquired data regardless of it's size
        :return: a dictionary of the weights of the WeightedCost
        """
        window_size = data_buffer.maxlen
        Y, XX = zip(*data_buffer)
        targets = np.concatenate(Y).squeeze()
        features = np.concatenate(XX, axis=0)

        features = features[~np.isnan(targets), :]
        targets = targets[~np.isnan(targets)]

        if len(targets) == 0:
            return self.zeros(window_size)

        weights, residual = self.least_squares(features, targets)
        if residual is None:
            return self.zeros(window_size)

        result = {weight_name + '_%d' % window_size: val for weight_name, val in zip(self.predictor_names, weights)}
        n, k = features.shape
        result['adjusted_r_squared_%d' % window_size] = residual
        # result['adjusted_r_squared_%d' % window_size] = 1 - (1 - residual / np.sum((targets-targets.mean()) ** 2)) * (n - 1) / (n - k - 1) if np.sum((targets-targets.mean()) ** 2) > 0 else residual
        return result

    def zeros(self, window_size):
        return {weight_name + '_%d' % window_size: 0.0 for weight_name in self.predictor_names + ['adjusted_r_squared']}



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


class WeightedKinematicCostDescentEstimatorBases(Steppable):
    """
    A ControllerEstimator for the WeightedKinematicCostDescent ControlLaw. Estimates the weight vector
    """

    def __init__(self, disturbance=True, method='LS', zero_thresh=1e-15):
        self.disturbance=disturbance
        self.method = method
        self.zero_thresh = zero_thresh

    def _step(self, cost_descents, input, J_RG, J_HG, RG_V_RH):
        cost_descents = cost_descents.values()[0]
        J_RG = J_RG.values()[0]
        J_HG = J_HG.values()[0]
        RG_V_RH = RG_V_RH.values()[0]

        h_order = J_HG.column_names()
        o_order = J_RG.column_names()
        x_order = h_order + o_order
        w_order = cost_descents.column_names()

        dCdx = cost_descents # (X, C)
        J_RGp = J_RG.pinv() # (O, 3)
        J_RGp_J_HG = J_RGp*J_HG # (O, H)

        dCdx.reorder(row_names=x_order)
        J_RGp.reorder(row_names=o_order)
        J_RGp_J_HG.reorder(row_names=o_order)
        u = np.array([input[name] for name in h_order])
        d = np.array(RG_V_RH).squeeze()[:len(J_RG)]
        if not self.disturbance or np.linalg.norm(d) < 0.4:
            d *= 0

        B = np.vstack((np.eye(len(h_order)), J_RGp_J_HG))
        D = np.vstack((np.zeros((len(o_order), 3)), J_RGp))
        dCdx = np.array(dCdx)

        Bp = np.linalg.pinv(B)
        Bp_dCdx = np.dot(Bp, dCdx)
        u_ff = -Bp.dot(D.dot(d)).squeeze()
        if self.disturbance:
            A = np.hstack((Bp_dCdx, u_ff[:, None]))
            b = u if np.linalg.norm(u) > self.zero_thresh else self.nan(u.shape)
            if b.shape == (4, 4):
                print 1
            w_order.append('ff_gain')
        else:
            A = Bp_dCdx
            b = u - u_ff if np.linalg.norm(u) > self.zero_thresh else self.nan(u.shape)

        return A, b, w_order, u, u_ff, h_order

    def step(self, cost_descents, input, J_RG, J_HG, RG_V_RH):
        A, b, w_order, u, u_ff, h_order = self._step(cost_descents, input, J_RG, J_HG, RG_V_RH)
        return {'A': A, 'b': b, 'w_order':w_order, 'u':u, 'u_ff':u_ff, 'h_order': h_order}

    def nan(self, shape):
        a = np.empty(shape)
        a[:] = np.NAN
        return a


class WeightedKinematicCostDescentEstimator(WeightedKinematicCostDescentEstimatorBases):
    """
    A ControllerEstimator for the WeightedKinematicCostDescent ControlLaw. Estimates the weight vector
    """

    def __init__(self, disturbance=True, method='LS'):
        super(WeightedKinematicCostDescentEstimator, self).__init__(disturbance, method)

    def step(self, cost_descents, input, J_RG, J_HG, RG_V_RH):
        A, b, w_order, _, _, _ = self._step(cost_descents, input, J_RG, J_HG, RG_V_RH)

        if self.method == 'NNLS':
            fit = nnls(A, b.squeeze())
            w = fit[0]

        elif self.method == 'LS':
            w = np.linalg.pinv(A).dot(b)

        else:
            raise ValueError('Wrong method string: %s' % self.method)

        return dict(zip(w_order, w.squeeze()))


class WeightedKinematicCostDescentController(Steppable):

    def __init__(self, suffix='_1'):
        self.suffix = suffix

    def step(self, A_info, weights):
        ret = {}
        A = A_info['A']
        w_order = A_info['w_order'][:]
        if 'ff_gain' in w_order:
            ff_idx = w_order.index('ff_gain')
            w_order.remove('ff_gain')
            ret['u_ff'] = A[:, ff_idx]*weights['ff_gain'+self.suffix]
            A = np.delete(A, ff_idx, 1)

        else:
            ret['u_ff'] = np.zeros(len(A))

        ret['u_order'] = A_info['h_order']
        w = np.array([weights[w+self.suffix] for w in w_order])
        ret['u_orig'] = A_info['u']
        ret['u_c'] = A.dot(w)
        ret['u_hat'] = ret['u_c'] + ret['u_ff']
        ret['u_err'] = ret['u_orig'] - ret['u_hat']
        U_c = {'u_'+w:A[:, i] for i, w in enumerate(w_order)}
        U_c['ff'] = A_info['u_ff']
        w_dict = {'w_'+w: v for w, v in weights.items()}
        ret.update(U_c)
        ret.update(w_dict)
        return ret


class Joobie(Steppable):

    def step(self, jacobian, joint_vels):
        u_order = joint_vels.pop('u_order')
        (_, jacobian), = jacobian.items()
        jacobian.reorder(column_names=u_order)
        v_order = jacobian.row_names()
        ret = {}
        for k, th in joint_vels.items():
            if k[0] == 'u':
               ret['V_%s' % k[2:]] =  jacobian * th

        ret['V_order'] = v_order
        return ret


class PandaSink(Steppable):
    def __init__(self):
        self.df = pd.DataFrame()

    def step(self, states):
        self.df = self.df.append([states], ignore_index=True)

    def get_data(self):
        return self.df.copy()


def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    """
    Solves min x.T P x - q for Gx >= h, Ax = b
    :param P:
    :param q:
    :param G:
    :param h:
    :param A:
    :param b:
    :return:
    """

    qp_G = P # .5 * (P + P.T)  # make sure P is symmetric
    qp_a = q
    if A is not None:
        qp_C = np.vstack([A, G]).T
        qp_b = np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)

