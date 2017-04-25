#!/usr/bin/python

import rospy
import baxter_interface
from baxter_pykdl import baxter_kinematics
import numpy as np
import numpy.linalg as npla
from math import sqrt
import threading
import tf.transformations
import tf
from inspect import getargspec, ismethod

MAX_JOINT_VEL = 1.5 # Max commanded joint velocity
MAX_NS_JOINT_VEL = 0.5 # Max nullspace joint velocity
TIME_STEP = 0.05 # time delta 

# Arm neutral configs
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


class DiscreteNLS():
    """
    A class describing a discrete nonlinear system
    It takes the form:
    x(k+1) = x(k) + f(x(k),u(k))
    y(k) = g(x(k), u(k))

    """

    def __init__(self, f, g, x_0, time_step=1, t_0=0):

        if not check_arg_count(f, 2):
            raise ValueError("The system function f must take 2 arguments")

        if not check_arg_count(g, 2):
            raise ValueError("The system function g must take 2 arguments")

        self.f = f
        self.g = g
        self.x_0 = np.array(x_0)
        self.x = self.x_0.copy()
        self.time_step = time_step
        self.t_0 = t_0
        self.t = self.t_0
        self.k = 0
        self.y = None

    def reset():
        """
        Resets the system to initial conditions
        """

        self.t = self.t_0
        self.k = 0
        self.x = self.x_0.copy()
    
    def step(u):
        """
        Steps the system by one time step
        """
        
        self.x += self.f(self.x, u)
        self.y = self.g(self.x, self.u)
        self.k += 1
        self.t += self.time_step
        return self.x, self.y


class NMPC():
    """
    A Nonlinear Model Predictive Controller class

    Args:
    limb: 'left'|'right'
    """

    def __init__(self, limb):
        self._limb = baxter_interface.Limb(limb)
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
        #Stop the controller
        self._run = False

    def start(self):
        #Start the controller
        self._run = True
        threading.Thread(target=self._run_controller).start()

    def _update_endpoint_velocity(self):
        # Measures the current endpoint pose error and commands a new set of joint velocities
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
        """Sets the endpoint position setpoint to a new value. Right now orientation is not
        supported. The controller just maintains the endpoint orientation that it measured when
        it was initialized.

        Args:
        setpoint: (3,) ndarray - the desired endpoint position [x, y, z]
        """
        self._position_setpoint = setpoint
        for i, dof in enumerate(['x', 'y', 'z']):
            self._dof_controllers[dof].set_desired_value(setpoint[i])


class ArmNMPC():
    """
    A container class for an NMPC controller specific to the Baxter Arm
    Args:
    limb: 'left'|'right'
    """

    def __init__(self, limb, time_step):
        self._neutral_config = NEUTRAL[limb]
        self._kin = baxter_kinematics(limb)
        self._limb = baxter_interface.Limb(limb)
        self.time_step = time_step
        
        # Choose the endpoint's current pose as the starting setpoint
        x_0 = np.array(self._limb.endpoint_pose()['position'] + 
                self._limb.endpoint_pose()['orientation'])

        # Define discrete NLS system
        self.system = DiscreteNLS(self.f, self.g, x_0, time_step=TIME_STEP)

    def f(x, u):
        """
        The state increment function of the discrete NLS
        """

        return self.time_step * u
    
    def g(x, u):
        """
        The output function of the discrete NLS. This should be the inverse kinematics
        """

        return np.dot(self._kin.jacobian(), x)

    def set_neutral_config(self, neutral_config_dict):
        """Sets the arm's neutral configuration to something other than the default value.

        Args:
        neutral_config_dict: dict - {'left_s1':value, ..., 'left_w2':value}
        """
        self._neutral_config = neutral_config_dict


def check_arg_count(func, args_to_pass):
    
    # get the specs of the function
    argspec = getargspec(func)

    # check for args or kwargs in which case we can pass any number of args
    if argspec.varargs is not None or argspec.keywords is not None:
        return True

    # otherwise find range of acceptable arguments
    max_args = len(argspec.args) - int(ismethod(func)
    min_args = max_args - len(argspec.defaults)

    # return true if passing args_to_pass many args would work
    return  min_args <= args_to_pass <= max_args

def null(a, rtol=1e-5):
    """ Computes the nullspace of a

    Returns:
    rank: the rank of the matrix (# of dims with singular values > rtol)
    nullspace: basis vectors for the nullspace
    """
    u, s, v = npla.svd(a)
    rank = (s > rtol*s[0]).sum()
    return rank, v[rank:].T.copy()

def project_to_nullspace(nullspace, joint_order, joint_velocities):
    """Projects a dict of joint velocities into the nullspace of a manipulator. The resulting
    joint velocity dict will not produce any endpoint motion of the arm.

    Args:
    nullspace: basis vectors for the nullspace returned from null()
    joint_order: list - the order in which joints are represented in the Jacobian
        ['left_s1', ...,'left_w2']
    joint_velocities: dict - input joint velocities to project
    """
    # Make a vector of joint velocities
    velocities = np.array([joint_velocities[joint] for joint in joint_order])

    # Compute the projection onto each nullspace vector
    proj_velocities = np.zeros_like(velocities)
    for n in range(nullspace.shape[1]):
        proj_velocities += nullspace[:,n].dot(velocities) * nullspace[:,n]

    # Convert back to a velocities dict
    return {joint: proj_velocities[i] for i, joint in enumerate(joint_order)}

class MocapPointTracker:
    """Moves both of Baxter's arms to maintain a constant position relative to a single motion
    capture marker.

    Args:
    vel_controller_left/vel_controller_right: the ContinuousEndpointPoseControllers for each arm
    """
    def __init__(self, vel_controller_left, vel_controller_right):
        self._vel_controller_left = vel_controller_left
        self._vel_controller_right = vel_controller_right
        self._tf_listener = tf.TransformListener()
        self._position_offset_left = None
        self._position_offset_right = None

    def new_frame_callback(self, message):
        """Called for each new sensor_msgs/PointCloud message received from the mocap system

        Args:
        message: the PointCloud message object
        """
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
    # Converts a sensor_msgs/PointCloud to a (num_markers, 3, 1) ndarray
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
