#!/usr/bin/python
import argparse
import numpy as np
import numpy.linalg as npla
import threading

import baxter_interface
import phasespace.load_mocap as load_mocap
import rospy
import tf
from baxter_pykdl import baxter_kinematics
from kinmodel.track_kinmodel import KinematicTreeTracker, KinematicTreeExternalFrameTracker
from sensor_msgs.msg import JointState
from tf.transformations import quaternion_from_euler

import kinmodel
from kinmodel.tools import colvec
from kinmodel.tools import multidot

MAX_JOINT_VEL = 1.5  # Max commanded joint velocity
MAX_NS_JOINT_VEL = 0.5  # Max nullspace joint velocity

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

NEUTRAL_ARRAY = np.array([0.0, -0.55, 0.0, 0.75, 0.0, 1.26, 0.0])


class TaskHeirarchy():
    """
    Stores task space or joint space velocity vectors in a priority order and successively computes the optimal joint
    angles to perform the tasks
    """

    def __init__(self, joint_dims, rate):
        """

        :param joint_dims: the number of joints 
        """
        self.joint_dims = joint_dims
        self.dt = 1.0 / rate
        self.tasks = []
        self.taskspace_flags = []
        self.jacobian_getters = []

        # a placeholder for storing a function that gets the jacobian so we can automatically split it up later
        self.get_full_jacobian = None

    def add_task(self, task, jacobian, taskspace=True, priority=-1):

        # If priority is not set, add it to the end
        if priority == -1:
            priority = len(self.tasks)

        # make sure the task is a col vec
        task = np.array(task).reshape((-1, 1))

        # add it to the list
        self.tasks.insert(priority, task.copy())
        self.jacobian_getters.insert(priority, jacobian)
        self.taskspace_flags.insert(priority, taskspace)

    def add_tasks_dict(self, tasks_dict):
        """
        Adds a dictionary of tasks, keys being priority ints, values being tuples
        :param tasks_dict: a dictionary of int keyed tasks
        :return: None
        """

        for priority in tasks_dict:
            task, jacobian, taskspace = tasks_dict[priority]
            self.add_task(task, jacobian, taskspace, priority)

    def update_task(self, task, priority):
        # make sure the task is a col vec
        task = np.array(task).reshape((-1, 1))

        self.tasks[priority] = task.copy()

    def update_tasks_dict(self, tasks_dict):
        # update all tasks in dictionary
        for priority in tasks_dict:
            self.update_task(tasks_dict[priority], priority)

    def null(self, J):
        # compute null(J) = I - JpJ
        return np.identity(self.joint_dims) - np.dot(npla.pinv(J), J)

    def slice_jacobian_position(self):
        # take the positional component of the Jacobian
        return np.array(self.get_full_jacobian()[0:3, :])

    def slice_jacobian_rotation(self):
        # take the roational component of the jacobian
        return np.array(self.get_full_jacobian()[3:6, :])

    def setup_cascaded_pose_vel(self, pose_vel, jacobian_getter, starting_priority=0):

        # assign the function handle to get hte full Jacobian
        self.get_full_jacobian = jacobian_getter

        # add the position reference as the first task and the rotation reference as the second
        self.add_task(pose_vel[0:3], self.slice_jacobian_position, True, starting_priority)
        self.add_task(pose_vel[3:6], self.slice_jacobian_rotation, True, 1 + starting_priority)

    def direct_cascaded_pose(self, pose_vel, secondary=True):
        """
        Directly computes the cascaded pose scenario by directly applying the analytical solution specific to this case
        TODO: The general approach needs to be debugged so that its output for such a setup mimics this output
        :param pose_vel: the reference velocity of the pose
        :param secondary: flag to switch on or off rotational control
        :return: the joint angles for the arm
        """

        # an int which indicates how many of the desired tasks are feasible (between 0 and 2)
        feasibility = 0

        # initialize the velocity vector
        dtheta = np.zeros((7, 1))

        # make sure pose velocity reference is a column vector
        pose_vel = np.array(pose_vel).reshape((-1, 1))

        # take the position component of the reference
        dx1 = pose_vel[0:3]

        # get the positional Jacobian
        J1 = self.slice_jacobian_position()

        # if it is not full rank then do not invert and just return the current dtheta command and the feasibility
        if npla.matrix_rank(J1, tol=0.1) != 3:
            rospy.logdebug("Feasability: %d" % feasibility)
            return dtheta, feasibility

        # if we got here, the first task is feasible, so increment feasibility
        feasibility = 1

        # get the inverse of the positional Jacobian
        J1p = npla.pinv(J1)

        # add the joint angles which minimize error from this reference
        dtheta += J1p.dot(dx1)

        # get the rotational Jacobian and the nullspace of the positional Jacobian
        J2 = self.slice_jacobian_rotation()
        N1 = self.null(J1)

        # singularity_test1 is more conservative
        # J = self.get_full_jacobian()
        # singularity_test1 = npla.matrix_rank(J, tol=0.1) == 6

        # Test for singularities when applying the second task
        singularity_test2 = npla.matrix_rank(np.dot(J2, N1), tol=0.1) == 3

        # if it is not full rank then do not invert and just return the current dtheta command and the feasibility
        if not singularity_test2 or not secondary:
            rospy.logdebug("Feasability: %d" % feasibility)
            return dtheta, feasibility

        # if we got here then both tasks are feasible
        feasibility = 2

        # compute the various components of the solution dtheta = dtheta1 + N1*(J2N1)p *(dx2-J2*J1p*dx1)
        J2N1p = npla.pinv(np.dot(J2, N1))
        dx2 = pose_vel[3:6]
        dx21_proj = dx2 - multidot(J2, J1p, dx1)
        dtheta2 = multidot(N1, J2N1p, dx21_proj)

        # add the joint angles which minimize error of the second task within the nullspace of the first task's solution
        dtheta += dtheta2

        # return the joint angles and feasibility
        rospy.logdebug("Feasability: %d" % feasibility)
        return dtheta, feasibility


    def direct_full_pose(self, pose_vel, neutral_error):
        """
        Directly computes the cascaded pose scenario by directly applying the analytical solution specific to this case
        TODO: The general approach needs to be debugged so that its output for such a setup mimics this output
        :param pose_vel: the reference velocity of the pose
        :param secondary: flag to switch on or off rotational control
        :return: the joint angles for the arm
        """

        # an int which indicates how many of the desired tasks are feasible (between 0 and 2)
        feasibility = 0

        # initialize the velocity vector
        dtheta = np.zeros((7, 1))

        # make sure pose velocity reference is a column vector
        dx1 = np.array(pose_vel).reshape((-1, 1))

        # get the positional Jacobian
        J1 = np.array(self.get_full_jacobian())

        # if it is not full rank then do not invert and just return the current dtheta command and the feasibility
        if npla.matrix_rank(J1, tol=0.1) != 6:
            rospy.logdebug("Feasability: %d" % feasibility)
            return dtheta, feasibility

        # if we got here, the first task is feasible, so increment feasibility
        feasibility = 1

        # get the inverse of the positional Jacobian
        J1p = npla.pinv(J1)

        # add the joint angles which minimize error from this reference
        dtheta += J1p.dot(dx1)

        N1 = self.null(J1)

        # project neutral configuration onto the nullspace of pose task
        dtheta2 = multidot(N1, colvec(neutral_error))

        # add the joint angles which minimize error of the second task within the nullspace of the first task's solution
        dtheta += dtheta2

        # return the joint angles and feasibility
        rospy.logdebug("Feasability: %d" % feasibility)
        return dtheta, feasibility


class EndpointVelocityController():
    """Generates joint velocity commands to produce a specified endpoint velocity for one of
    Baxter's arms.

    Args:
    limb: 'left'|'right'
    """

    def __init__(self, limb, rate, cascaded=False):
        self.cascaded = cascaded
        self._neutral_config = NEUTRAL[limb]
        self._kin = baxter_kinematics(limb)
        self._limb = baxter_interface.Limb(limb)
        self._nullspace_projector = TaskHeirarchy(7, rate)
        rospy.logdebug(limb)
        self._nullspace_projector.setup_cascaded_pose_vel(np.zeros(7), self._kin.jacobian)
        rospy.logdebug(self._nullspace_projector.get_full_jacobian())

        # pretend our initial state was moved to from neutral state
        joint_velocity_command = self.get_joint_array() - NEUTRAL_ARRAY

        # Clamp the joint velocity command so it's within limits for each joint
        joint_vel_scalar = max(abs(joint_velocity_command.flatten())) / (MAX_JOINT_VEL*0.1)

        if joint_vel_scalar > 1:
            joint_velocity_command /= joint_vel_scalar

        # set this command as the last one so if we start in a singularity we get out by
        # moving closer to neutral
        self._last_velocity_cmd = colvec(joint_velocity_command)

    def set_endpoint_velocity(self, velocity_cmd, secondary=True):
        """Sends joint velocity commands to produce the instantaneous endpoint velocity specified
        by velocity_cmd

        Args:
        velocity_cmd: (6,) ndarray - the desired endpoint velocity [x', y', z', r', p' y']
        """
        # End effector command velocity
        # Compute the joint velocity commands
        if self.cascaded:
            joint_velocity_command, feasible = self._nullspace_projector.direct_cascaded_pose(velocity_cmd, secondary)

        else:
            joint_velocity_command, feasible = self._nullspace_projector.direct_full_pose(velocity_cmd,
                                                                                          NEUTRAL_ARRAY - NEUTRAL_ARRAY)
        # rospy.logdebug("joint command: " + str(joint_velocity_command.flatten()))
        # if we are stuck in a singularity, reverse the last command in order to get us out
        if feasible == 0:
            joint_velocity_command = -self._last_velocity_cmd

        # otherwise, we have out joint velocity command
        else:
            # Clamp the joint velocity command so it's within limits for each joint
            joint_vel_scalar = max(abs(joint_velocity_command.flatten())) / MAX_JOINT_VEL

            if joint_vel_scalar > 1:
                joint_velocity_command /= joint_vel_scalar

            # store this command so that if it moves us to a singularity we can move back out
            # Note if this line is placed outside of the else, then in the case where numerical errors have rendered us
            # moving from one singularity to another,it will bounce between the two, when inside the else, it will
            # continue moving in the same reversed direction until outside singularity
            self._last_velocity_cmd = np.array(joint_velocity_command)

        # convert the command velocity array into a dictionary
        command_velocity_dict = {joint: joint_velocity_command[i, 0] for i, joint in
                                 enumerate(self._limb.joint_names())}

        # send the velocity command
        self._limb.set_joint_velocities(command_velocity_dict)

    def set_neutral_config(self, neutral_config_dict):
        """Sets the arm's neutral configuration to something other than the default value.

        Args:
        neutral_config_dict: dict - {'left_s1':value, ..., 'left_w2':value}
        """
        self._neutral_config = neutral_config_dict

    def get_joint_array(self):
        """
        :return: returns the current joint angles as a numpy array
        """
        return np.array([self._limb.joint_angle(joint_name) for joint_name in self._limb.joint_names()])


class ContinuousEndpointPoseController():
    """A closed loop endpoint pose controller which uses joint velocity commands to move the
    specified limb's endpoint to the desired pose.

    Args:
    limb: 'left'|'right'
    """

    def __init__(self, limb, get_jacobian, configuration_ref, k_conf=1.0, k_grip=1.0, rate=100, box_joints_topic=None):
        rospy.logdebug('Initializing ContinuousEndpointPoseController')
        self.rate = rate
        self._limb = baxter_interface.Limb(limb)
        self._vel_controller = EndpointVelocityController(limb, rate)
        self._run = True

        # Choose the endpoint's current pose as the starting setpoint
        self._orientation_setpoint = tuple(self._limb.endpoint_pose()['orientation'])
        self._position_setpoint = tuple(self._limb.endpoint_pose()['position'])

        # Create the controllers for the config
        rospy.logdebug('Initializing ObjectConfigKinematics')
        self._config_kinematics = ObjectConfigKinematics(get_jacobian, configuration_ref, k_conf, k_grip, box_joints_topic)
        self._config_kinematics.set_dynamic_configuration_ref('configuration_ref')

        # Start the velocity control loop
        self.start()
        rospy.logdebug('ContinuousEndpointPoseController init complete')

    def stop(self):
        # Stop the controller
        self._run = False

    def start(self):
        # Start the controller
        self._run = True
        t = threading.Thread(target=self._run_controller)
        t.daemon = True
        t.start()

    def _update_endpoint_velocity(self):
        # Get the endpoint veloctiy to close the error between current joint angles and config
        velocity_cmd = self._config_kinematics.compute_optimal_robot_effector_velocity()

        rospy.logdebug("\npose command:  " + str(velocity_cmd.flatten()))
        self._vel_controller.set_endpoint_velocity(velocity_cmd)
        # self._vel_controller.set_endpoint_velocity([0.0]*3 + [.0, 0.0, 0.02], True)

    def _run_controller(self):
        rate = rospy.Rate(self.rate)
        while (not rospy.is_shutdown()) and self._run:
            self._update_endpoint_velocity()
            rate.sleep()


class ObjectConfigKinematics():

    def __init__(self, frame_tracker, configuration_ref, k_conf=1.0, k_grip=0.1, box_joints_topic=None, forward_kinematics=None):
        """
        Class which computes kinematics on an object relative to the robots grip on it and a desired configuration
        :param get_jacobian: a function handle which returns the current objects jacobian, takes no arguments
        :param configuration_ref: the initial reference configuration, this can be updated dynamically
        :param forward_kinematics: a function handle which computes the forward kinematics from a joint array
        """

        if box_joints_topic is None:
            box_joints_topic = 'box/joint_states'

        self.k_conf = k_conf
        self.k_grip = k_grip

        # copy across member variables
        self._frame_tracker = frame_tracker
        self.get_jacobian = self._frame_tracker.compute_jacobian_matrix
        self.configuration_ref = colvec(configuration_ref)
        self.forward_kinematics = forward_kinematics

        # block until we get out first joint update
        self.joints = colvec(rospy.wait_for_message(box_joints_topic, JointState).position)

        # set up a subscriber to this joint topic
        self.sub_joints = rospy.Subscriber(box_joints_topic, JointState, self._update_joints)

        # placeholder for a reference subscriber
        self.sub_config = None

        self._tf_listener = tf.TransformListener()

        self.optimal_grip_orientation = self.get_grip_transform()

        rospy.logdebug('ObjectConfigKinematics init complete')

    def set_dynamic_configuration_ref(self, topic_name):
        """
        sets up a subscriber to a given topic which should publish JointState messags
        :param topic_name: the topic the boxes joint angles are published on
        """
        self.sub_config = rospy.Subscriber(topic_name, JointState, self._update_configuration_ref)

    def _update_configuration_ref(self, msg):
        # copy across angles into a column vector
        self.configuration_ref = colvec(msg.position)

    def _update_joints(self, msg):
        # copy across angles into a column vector
        self.joints = colvec(msg.position)

    def compute_optimal_robot_effector_position(self):
        """
        Computes the optimal position of the end effector given the current configuration reference
        :return: 
        """
        return self.forward_kinematics(self.configuration_ref)

    def compute_optimal_robot_effector_velocity(self):
        """
        Computes the optimal velocity end effector in order to close the error from the configuration reference
        :return: 
        """

        config_error = self.configuration_ref - self.joints
        # Perform this before grip correction so that appropriate tf frames are published by the get_jacobian call
        config_correction = colvec(self.get_jacobian().dot(-self.k_conf*(config_error)))
        grip_error = self.compute_grip_error()
        grip_correction = self.k_grip * colvec(grip_error)
        rospy.logdebug('\nconfig error     : ' + str(config_error.flatten()) +
                       '\ngrip error       : ' + str(grip_error.flatten()[[3, 5]]) +
                       '\nconfig correction: ' + str(config_correction.flatten()) +
                       '\ngrip correction  : ' + str(grip_correction.flatten()))
        return config_correction + grip_correction


    def compute_grip_error(self):
        obj_grip_error = self.optimal_grip_orientation - self.get_grip_transform()
        obj_grip_error[1] = 0
        base_grip_error = self._frame_tracker.compute_transform('base', 'trans3').R().dot(colvec(obj_grip_error))[:, 0]

        grip_diff = np.zeros(6)
        grip_diff[3:] = base_grip_error

        return grip_diff


    def get_grip_transform(self):
        homog_transform = self._frame_tracker.compute_transform('trans3', 'left_hand')
        return np.array(tf.transformations.euler_from_matrix(homog_transform.rot().R()))


def setup_trackers(box_joints_topic):
    parser = argparse.ArgumentParser()
    parser.add_argument('kinmodel_json_optimized', help='The kinematic model JSON file')
    # parser.add_argument('mocap_npz')
    args = parser.parse_args()

    #Load the calibration sequence
    # calib_data = np.load(args.mocap_npz)
    # ukf_mocap = load_mocap.MocapArray(calib_data['full_sequence'][:,:,:], FRAMERATE)

    # Load the mocap stream
    ukf_mocap = load_mocap.PointCloudStream('/mocap_point_cloud')
    tracker_kin_tree = kinmodel.KinematicTree(json_filename=args.kinmodel_json_optimized)
    kin_tree = kinmodel.KinematicTree(json_filename=args.kinmodel_json_optimized)


    # Add the external frames to track
    frame_tracker = KinematicTreeExternalFrameTracker(kin_tree, 'object_base')
    frame_tracker.attach_frame('joint2', 'trans2')
    frame_tracker.attach_frame('joint3', 'trans3')
    frame_tracker.attach_tf_frame('joint3', 'left_hand')
    frame_tracker.attach_tf_frame('joint2', 'base')

    def new_frame_callback(i, joint_angles):
        frame_tracker.set_config({'joint2':joint_angles[0], 'joint3':joint_angles[1]})

    tracker = KinematicTreeTracker(tracker_kin_tree, ukf_mocap, joint_states_topic=box_joints_topic,
            object_tf_frame='/object_base', new_frame_callback=new_frame_callback)
    # ukf_output = tracker.run()

    return tracker, frame_tracker

def main():
    rospy.init_node('baxter_velocity_control', log_level=rospy.DEBUG)
    tracker, frame_tracker = setup_trackers('/box/joint_states')
    tracker.start()

    rospy.logdebug("Updating frame tracker")
    while not frame_tracker._update():
        pass

    rospy.logdebug("Updated.")

    rate = rospy.get_param('~rate', 100)
    vel_controller_left = ContinuousEndpointPoseController('left', frame_tracker, np.array([0.2, 0.2]), 0.2, 0.4, rate)
    rospy.spin()


if __name__ == "__main__":
    main()
