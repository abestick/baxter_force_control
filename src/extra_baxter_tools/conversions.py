import numpy as np
import rospy
from geometry_msgs.msg import Pose, Transform, PoseStamped, TransformStamped, Point, Vector3, PointStamped, \
    Vector3Stamped, Quaternion, QuaternionStamped, Twist, TwistStamped
from trajectory_msgs.msg import JointTrajectory, MultiDOFJointTrajectory, \
    JointTrajectoryPoint, MultiDOFJointTrajectoryPoint
from definitions import BaxterDefs as bd


raw_type_map = {
    'bool': bool,
    'int8': int,
    'uint8': int,
    'int16': int,
    'uint16': int,
    'int32': int,
    'uint32': int,
    'int64': int,
    'uint64': int,
    'float32': float,
    'float64': float,
    'string': str,
    'time': rospy.Time,
    'duration': rospy.Duration
}


stamp_map = {Pose: PoseStamped,
             Transform: TransformStamped,
             Point: PoseStamped,
             Vector3: Vector3Stamped,
             Quaternion: QuaternionStamped,
             Twist: TwistStamped}


stamp_attr_map = {PoseStamped: 'pose',
                  TransformStamped: 'transform',
                  PointStamped: 'point',
                  Vector3Stamped: 'vector',
                  QuaternionStamped: 'quaternion',
                  TwistStamped: 'twist'}


def ros_eval(type_str):
    """
    Converts a ros defined variable into a tuple containing the variable name and its type
    :param typed_variable_str: the string of the definitoin of the variable
    :return: type
    """
    if '[]' in type_str:
        return ros_eval(type_str.split('[]')[0]),

    elif '/' in type_str:
        variable_pkg_name, type_name = type_str.split('/')
        variable_pkg = __import__(variable_pkg_name + '.msg')
        return getattr(variable_pkg.msg, type_name)

    elif type_str in raw_type_map:
        return raw_type_map[type_str]

    else:
        try:
            return eval(type_str)
        except AttributeError:
            raise ValueError('%s is not a built-in ros type, array or other message type. ')


def ros_eval_slot(msg_type, slot_name):
    return ros_eval(msg_type._slot_types[msg_type.__slots__.index(slot_name)])


def stamp(msg, **kwargs):
    stamped_type = stamp_map[type(msg)]
    kwargs[stamp_attr_map[stamped_type]] = msg
    return stamped_type(**kwargs)


def stamp_wrapped(x_to_msg):
    def wrapped(x):
        return stamp(x_to_msg(x))

    return wrapped


def tuple_wrapped(x_to_msg):
    def wrapped(x):
        return x_to_msg(x),

    return wrapped


def unstamp(msg):
    return getattr(msg, stamp_attr_map[type(msg)])


def cartesian_to_array(msg):
    return np.array([msg.x, msg.y, msg.z])


def quaternion_to_array(msg):
    return np.array([msg.x, msg.y, msg.z, msg.w])


def pose_to_array(msg):
    return np.append(cartesian_to_array(msg.position), quaternion_to_array(msg.orientation))


def transform_to_array(msg):
    return np.append(cartesian_to_array(msg.translation), quaternion_to_array(msg.rotation))


def twist_to_array(msg):
    return np.append(cartesian_to_array(msg.linear), cartesian_to_array(msg.angular))


destructor_map = {Pose: pose_to_array,
                  Transform: transform_to_array,
                  Point: cartesian_to_array,
                  Vector3: cartesian_to_array,
                  Quaternion: quaternion_to_array,
                  Twist: twist_to_array}


def se3_to_array(msg):
    if type(msg) in stamp_attr_map:
        msg = unstamp(msg)

    return destructor_map[type(msg)](msg)


def spatial_to_arrays(msg):

    if type(msg) in stamp_attr_map:
        msg = unstamp(msg)

    if isinstance(msg, (Point, Vector3)):
        return cartesian_to_array(msg), None

    else:
        return tuple(np.split(se3_to_array(msg), [3]))


def array_to_pose_msg(array):
    return Pose(position=Point(*array[:3]), orientation=Quaternion(*array[3:]))


def array_to_transform_msg(array):
    return Transform(translation=Vector3(*array[:3]), rotation=Quaternion(*array[3:]))


def array_to_point_msg(array):
    return Point(*array)


def array_to_vector3_msg(array):
    return Vector3(*array)


def array_to_quaternion_msg(array):
    return Quaternion(*array)


def array_to_twist_msg(array):
    return Twist(linear=Vector3(*array[:3]), angular=Vector3(*array[3:]))


constructor_map = {Pose: array_to_pose_msg,
                   Transform: array_to_transform_msg,
                   Point: array_to_point_msg,
                   Vector3: array_to_vector3_msg,
                   Quaternion: array_to_quaternion_msg,
                   Twist: array_to_twist_msg}

constructor_map.update({stamp_map[msg_type]: stamp_wrapped(constructor_map[msg_type]) for msg_type in constructor_map})
constructor_map.update({(msg_type,): tuple_wrapped(constructor_map[msg_type]) for msg_type in constructor_map})


def array_to_spatial_msg(array, msg_type):
    return constructor_map[msg_type](array)


def trajectory_point_to_array(msg):
    if isinstance(msg, JointTrajectoryPoint):
        return joint_trajectory_point_to_array(msg)

    elif isinstance(msg, MultiDOFJointTrajectoryPoint):
        return multi_dof_point_to_array(msg)


def joint_trajectory_point_to_array(msg, component='positions'):
    """

    :param JointTrajectoryPoint msg:
    :param component:
    :return:
    """

    return np.array(getattr(msg, component))


def multi_dof_point_to_array(msg, component='transforms'):
    """

    :param MultiDOFJointTrajectoryPoint msg:
    :param component:
    :return:
    """
    if component == 'transforms':
        return transform_to_array(msg.transforms[0])

    else:
        return twist_to_array(getattr(msg, component)[0])


def trajectory_to_arrays(msg):
    time = []
    points = []

    for point in msg.points:
        time.append(point.time_from_start.to_sec())
        points.append(trajectory_point_to_array(point))

    return np.array(points), np.array(time)


def arrays_to_joint_trajectory_msg(points, time, component='positions', names=None):
    if names is None:
        names = bd.arm.joint_names

    msg = JointTrajectory(joint_names=names)
    for point, t in zip(points, time):
        kwargs = {component: point, 'time_from_start': rospy.Duration(t)}
        msg.points.append(JointTrajectoryPoint(**kwargs))
    return msg


def arrays_to_multi_dof_trajectory_msg(points, time, component='transforms'):
    msg = MultiDOFJointTrajectory()
    for point, t in zip(points, time):
        kwargs = {component: array_to_spatial_msg(point, ros_eval_slot(MultiDOFJointTrajectoryPoint, component)),
                  'time_from_start': rospy.Duration(t)}
        msg.points.append(MultiDOFJointTrajectoryPoint(**kwargs))
    return msg
