#!/usr/bin/python
import rospy
import baxter_interface
from baxter_pykdl import baxter_kinematics
import numpy as np
import numpy.linalg as npla
from math import sqrt
import threading
import pid_controllers
import tf.transformations
import tf
from sensor_msgs.msg import PointCloud

LIMB = 'left'
WRENCH = [10,0,0,0,0,0]
VEL_COMMAND = [0, 0, -0.1, 0, 0, 0]
MAX_JOINT_VEL = 1.5
MAX_NS_JOINT_VEL = 0.5

# NEUTRAL = {'left': 
#         {'left_w0': -0.09625729443980971,
#          'left_w1': -0.7029466960484908,
#          'left_w2': 0.03528155812136451,
#          'left_e0': 0.057524279545703015,
#          'left_e1': 1.131694326262464,
#          'left_s0': -0.8736020587007431,
#          'left_s1': -0.4670971499111085}}

NEUTRAL = {'left': 
        {'left_w0': 0.0,
         'left_w1': -0.55,
         'left_w2': 0.0,
         'left_e0': 0.75,
         'left_e1': 0.0,
         'left_s0': 1.26,
         'left_s1': 0.0},
         'right': 
        {'right_w0': 0.0,
         'right_w1': -0.55,
         'right_w2': 0.0,
         'right_e0': 0.75,
         'right_e1': 0.0,
         'right_s0': 1.26,
         'right_s1': 0.0}}

class EndpointVelocityController():
    def __init__(self, limb):
        self._neutral_config = NEUTRAL[limb]
        self._kin = baxter_kinematics(limb)
        self._limb = baxter_interface.Limb(limb)

    def set_endpoint_velocity(self, velocity_cmd):
        # Neutral position return velocity
        # Compute nullspace velocities
        velocity_dict = {joint: self._neutral_config[joint] - self._limb.joint_angle(joint) for joint in self._limb.joint_names()}

        #Clamp the nullspace velocity command so it's within limits for each joint
        vel_scale_factor = max([abs(velocity) for velocity in velocity_dict.values()]) / MAX_NS_JOINT_VEL
        if vel_scale_factor > 1:
            velocity_dict = {joint: velocity_dict[joint] / vel_scale_factor for joint in self._limb.joint_names()}
        
        # Project into the nullspace
        nullspace = null(np.array(self._kin.jacobian()))[1]
        velocity_dict = project_to_nullspace(nullspace, self._limb.joint_names(), velocity_dict)

        # End effector command velocity
        # Compute the joint velocity commands
        jacobian_pinv = np.array(self._kin.jacobian_pseudo_inverse())
        joint_velocity_command = jacobian_pinv.dot(velocity_cmd)
        command_velocity_dict = {joint: joint_velocity_command[i] for i, joint in enumerate(self._limb.joint_names())}

        #Clamp the EE velocity command so it's within limits for each joint
        ee_vel_scale_factor = max([abs(velocity) for velocity in command_velocity_dict.values()]) / MAX_JOINT_VEL
        if ee_vel_scale_factor > 1:
            command_velocity_dict = {joint: command_velocity_dict[joint] / ee_vel_scale_factor for joint in self._limb.joint_names()}

        # Command the velocities
        velocity_dict = {joint:command_velocity_dict[joint] + velocity_dict[joint] for joint in self._limb.joint_names()}
        self._limb.set_joint_velocities(velocity_dict)

    def set_neutral_config(self, neutral_config_dict):
        self._neutral_config = neutral_config_dict

class ContinuousEndpointPoseController():
    def __init__(self, limb):
        self._limb = baxter_interface.Limb(limb)
        self._vel_controller = EndpointVelocityController(limb)
        self._run = True

        # Choose the endpoint's current pose as the starting setpoint
        self._orientation_setpoint = np.array(self._limb.endpoint_pose()['orientation'])
        self._position_setpoint = np.array(self._limb.endpoint_pose()['position'])

        # Create the controllers for each DoF of the endpoint velocity
        self._dof_controllers = {
                'x': pid_controllers.PidController(k_p=3.0).set_desired_value(self._position_setpoint[0]),
                'y': pid_controllers.PidController(k_p=3.0).set_desired_value(self._position_setpoint[1]),
                'z': pid_controllers.PidController(k_p=3.0).set_desired_value(self._position_setpoint[2]),
                'rpy': pid_controllers.OrientationPidController(k_p=1.0).set_desired_value(self._orientation_setpoint)
        }
        
        # Start the velocity control loop
        self.start()

    def stop(self):
        self._run = False

    def start(self):
        self._run = True
        threading.Thread(target=self._run_controller).start()

    def _update_endpoint_velocity(self):
        CONTROL_DOFS = ['x', 'y', 'z', 'rpy']
        IGNORE_DOFS = []#'roll', 'pitch', 'yaw']
        current_ee_vel = {dof: 0.0 for dof in CONTROL_DOFS}
        current_ee_pos = {dof: self._limb.endpoint_pose()['position'][i] for i, dof, in enumerate(CONTROL_DOFS[0:3])}
        current_ee_ori = self._limb.endpoint_pose()['orientation']
        #Seems like the quaternion is actually in the correct order?
        # current_ee_ori = np.concatenate((current_ee_ori[3:4], current_ee_ori[0:3]))

        velocity_cmd = [self._dof_controllers[dof].get_control_cmd(current_ee_pos[dof], current_ee_vel[dof]) for dof in CONTROL_DOFS[0:3]]
        ori_velocity_cmd = self._dof_controllers['rpy'].get_control_cmd(current_ee_ori, 0.0)
        velocity_cmd.extend(ori_velocity_cmd)
        self._vel_controller.set_endpoint_velocity(velocity_cmd)

    def _run_controller(self):
        rate = rospy.Rate(100)
        while (not rospy.is_shutdown()) and self._run:
            self._update_endpoint_velocity()
            rate.sleep()

    def update_position_setpoint(self, setpoint):
        self._position_setpoint = setpoint
        for i, dof in enumerate(['x', 'y', 'z']):
            self._dof_controllers[dof].set_desired_value(setpoint[i])

def null(a, rtol=1e-5):
    u, s, v = npla.svd(a)
    rank = (s > rtol*s[0]).sum()
    return rank, v[rank:].T.copy()

def project_to_nullspace(nullspace, joint_order, joint_velocities):
    # Make a vector of joint velocities
    velocities = np.array([joint_velocities[joint] for joint in joint_order])

    # Compute the projection onto each nullspace vector
    proj_velocities = np.zeros_like(velocities)
    for n in range(nullspace.shape[1]):
        proj_velocities += nullspace[:,n].dot(velocities) * nullspace[:,n]

    # Convert back to a velocities dict
    return {joint: proj_velocities[i] for i, joint in enumerate(joint_order)}

class MocapPointTracker:
    def __init__(self, vel_controller_left, vel_controller_right):
        self._vel_controller_left = vel_controller_left
        self._vel_controller_right = vel_controller_right
        self._tf_listener = tf.TransformListener()
        self._position_offset_left = None
        self._position_offset_right = None

    def new_frame_callback(self, message):
        try:
            message = self._tf_listener.transformPointCloud('/base', message)
            data = point_cloud_to_array(message)
            point = data[0,:,0]
            if not np.any(np.isnan(point)):
                # Initialize the position offset
                if self._position_offset_left is None:
                    self._position_offset_left = point - np.array(self._vel_controller_left._limb.endpoint_pose()['position'])
                    self._position_offset_right = point - np.array(self._vel_controller_right._limb.endpoint_pose()['position'])
                else:
                    self._vel_controller_left.update_position_setpoint(point - self._position_offset_left)
                    self._vel_controller_right.update_position_setpoint(point - self._position_offset_right)
        except tf.LookupException:
            print("Couldn't find the mocap transformation")

def point_cloud_to_array(message):
    num_points = len(message.points)
    data = np.empty((num_points, 3, 1))
    for i, point in enumerate(message.points):
        data[i,:,0] = [point.x, point.y, point.z]
    return data

def main():
    rospy.init_node('baxter_velocity_control')
    vel_controller_left = ContinuousEndpointPoseController('left')
    vel_controller_right = ContinuousEndpointPoseController('right')
    tracker = MocapPointTracker(vel_controller_left, vel_controller_right)
    sub = rospy.Subscriber('/mocap_point_cloud', PointCloud, tracker.new_frame_callback)
    rospy.spin()


    

if __name__ == "__main__":
    main()
