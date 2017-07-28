import numpy as np
from baxter_pykdl import baxter_kinematics
from conversions import spatial_to_arrays, se3_to_array, array_to_transform_msg
from tf.transformations import quaternion_multiply, quaternion_inverse
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint, JointTrajectory, \
    JointTrajectoryPoint


def quaternion_difference(q1, q2):
    return quaternion_multiply(q1, quaternion_inverse(q2))


def pose_difference(p1, p2):
    pos_diff = p1[:3] - p2[:3]
    return np.append(pos_diff, quaternion_difference(p1[3:], p2[3:]))


def pose_sum(p1, p2):
    pos_sum = p1[:3] + p2[:3]
    return np.append(pos_sum, quaternion_multiply(p1[3:], p2[3:]))


class ExtendedBaxterKinematics(baxter_kinematics):

    def __init__(self, limb):
        super(ExtendedBaxterKinematics, self).__init__(limb)

    def inverse_from_msg(self, msg):
        position, orientation = spatial_to_arrays(msg)
        return self.inverse(list(position), list(orientation))

    def soft_inverse(self, pose, known_feasible, delta=0.01, iterations=100):
        position, orientation = np.split(pose, [3])
        if len(orientation) == 0:
            orientation = None
        joints = self.inverse(position, orientation)

        if joints is not None:
            return joints, pose

        elif iterations == 0:
            return None, None

        else:
            new_pose = pose_sum(delta * pose_difference(known_feasible, pose), pose)
            return self.soft_inverse(new_pose, known_feasible, delta, iterations-1)

    def soft_inverse_from_msg(self, pose, known_feasible, delta=0.01, iterations=100):
        position, orientation = spatial_to_arrays(pose)
        known_position, known_orientation = spatial_to_arrays(known_feasible)

        if orientation is None:
            orientation = []
            known_orientation = []

        pose = np.append(position, orientation)
        known_feasible = np.append(known_position, known_orientation)

        return self.soft_inverse(pose, known_feasible, delta, iterations)

    def invert_trajectory_msg(self, trajectory, known_feasible, delta=0.01, iterations=100):
        """

        :param MultiDOFJointTrajectory trajectory:
        :return:
        """
        joint_trajectory_msg = JointTrajectory(header=trajectory.header)
        pose_trajectory_msg = MultiDOFJointTrajectory(header=trajectory.header)
        points = [point for point in trajectory.points]

        deviation = np.zeros(7)
        deviation[-1] = 1

        for point in points:
            pose = point.transforms[0]
            joints, real_pose = self.soft_inverse_from_msg(pose, known_feasible, delta, iterations)
            deviation = pose_sum(pose_difference(se3_to_array(pose), real_pose), deviation)
            pose_trajectory_msg.points.append(MultiDOFJointTrajectoryPoint(
                transforms=[array_to_transform_msg(real_pose)]), time_from_start=point.time_from_start)

            if joints is None:
                joint_trajectory_msg.points.append(JointTrajectoryPoint())
                return joint_trajectory_msg, pose_trajectory_msg, deviation

            joint_trajectory_msg.points.append(JointTrajectoryPoint(positions=joints,
                                                                    time_from_start=point.time_from_start))

        return joint_trajectory_msg, pose_trajectory_msg, deviation