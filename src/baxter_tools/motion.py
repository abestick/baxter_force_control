from definitions import BaxterDefs as bd
from trajectory_msgs.msg import JointTrajectory
from conversions import trajectory_to_arrays, arrays_to_joint_trajectory_msg
import numpy as np


def saturate_trajectory_pos(joint_trajectory, uniform=False):
    """

    :param JointTrajectory joint_trajectory:
    :param bool uniform:
    :return:
    """
    positions, time = trajectory_to_arrays(joint_trajectory)
    new_positions = positions.copy()
    min_scale = 1

    for i, pos in enumerate(positions):
        # normalize the joint array so that now they must be less than or equal to 1
        abs_normalized_pos = np.abs((pos - bd.arm.neutrals) / (0.5 * bd.arm.ranges))

        if np.any(abs_normalized_pos > 1):
            scale_factor = 1.0 / np.max(abs_normalized_pos)
            min_scale = max(min_scale, scale_factor)
            new_positions[i, :] = scale_factor * pos + (1 - scale_factor) * bd.arm.neutrals

    if uniform:
        new_positions = min_scale * positions + (1 - min_scale) * bd.arm.neutrals

    return arrays_to_joint_trajectory_msg(new_positions, time)


def saturate_trajectory_vel(joint_trajectory, uniform=False):
    """

    :param JointTrajectory joint_trajectory:
    :param bool uniform:
    :return:
    """
    positions, time = trajectory_to_arrays(joint_trajectory)
    time_deltas = np.diff(time)
    new_deltas = time_deltas.copy()

    for i, time_delta in enumerate(time_deltas):
        diff = positions[i+1] - positions[i]
        overshoot = np.abs(diff / time_delta) - bd.arm.vel_limits
        if np.any(overshoot > 0):
            new_deltas[i] = diff / bd.arm.vel_limits

    if uniform:
        max_scale = np.max(new_deltas / time_deltas)
        new_deltas = time_deltas * max_scale

    new_time = np.insert(np.cumsum(new_deltas), 0, time[0])

    return arrays_to_joint_trajectory_msg(positions, new_time)
