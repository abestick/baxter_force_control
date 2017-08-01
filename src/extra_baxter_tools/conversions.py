import numpy as np
import rospy
from geometry_msgs.msg import Pose, Transform, PoseStamped, TransformStamped, Point, Vector3, PointStamped, \
    Vector3Stamped, Quaternion
from trajectory_msgs.msg import JointTrajectory, MultiDOFJointTrajectory, \
    JointTrajectoryPoint, MultiDOFJointTrajectoryPoint
from definitions import BaxterDefs as bd


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


def se3_to_array(msg):
    if isinstance(msg, Pose):
        return pose_to_array(msg)

    elif isinstance(msg, Transform):
        return transform_to_array(msg)

    elif isinstance(msg, PoseStamped):
        return pose_to_array(msg.pose)

    elif isinstance(msg, TransformStamped):
        return pose_to_array(msg.transform)


def spatial_to_arrays(msg):
    if isinstance(msg, (Point, Vector3)):
        return cartesian_to_array(msg), None

    elif isinstance(msg, PointStamped):
        return cartesian_to_array(msg.point), None

    elif isinstance(msg, Vector3Stamped):
        return cartesian_to_array(msg.vector), None

    else:
        return tuple(np.split(se3_to_array(msg), [3]))


def array_to_pose_msg(array):
    return Pose(position=Point(*array[:3]), orientation=Quaternion(*array[3:]))


def array_to_transform_msg(array):
    return Transform(translation=Vector3(*array[:3]), rotation=Quaternion(*array[3:]))


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


def arrays_to_joint_trajectory_msg(positions, time, names=None):
    if names is None:
        names = bd.arm.joint_names

    msg = JointTrajectory(joint_names=names)
    for pos, t in zip(positions, time):
        msg.points.append(JointTrajectoryPoint(positions=pos, time_from_start=rospy.Duration(t)))
    return msg


def arrays_to_multi_dof_trajectory_msg(poses, time):
    msg = MultiDOFJointTrajectory()
    for pose, t in zip(poses, time):
        msg.points.append(MultiDOFJointTrajectoryPoint(transforms=[array_to_transform_msg(pose)], time_from_start=rospy.Duration(t)))
    return msg
