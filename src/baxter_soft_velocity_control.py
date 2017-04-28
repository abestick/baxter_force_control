#!/usr/bin/python
import rospy
import baxter_interface
from baxter_pykdl import baxter_kinematics
import numpy as np
import numpy.linalg as npla
import threading
import pid_controllers
from std_msgs.msg import Float32MultiArray
from tf.transformations import quaternion_from_euler


MAX_JOINT_VEL = 1.5 # Max commanded joint velocity
MAX_NS_JOINT_VEL = 0.5 # Max nullspace joint velocity

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
        self.dt = 1.0/rate
        self.tasks = []
        self.taskspace_flags = []
        self.jacobian_getters = []

        # These are placeholders for storing variables used multiple times to optimize calculation
        self.dtheta_ia_list = []
        self.dtheta_1ia_list = []
        self.J_suc_i_list = []
        self.N_suc_i_list = []
        self.N_suc_1i_list = []

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
        for priority in tasks_dict:
            self.update_task(tasks_dict[priority], priority)

    def dtheta_i(self, i=0):

        # clear
        self.clear_time_variants()

        # BASE CASE: i=K
        if i == len(self.tasks)-1:
            return self.dtheta_ia(i)

        # RECURSIVE CASE:
        return self.dtheta_ia(i) + np.dot(self.N_suc_i(i), self.dtheta_i(i + 1))

    def dtheta_ia(self, i):
        # if the task is in joint space
        if not self.taskspace_flags[i]:
            self.dtheta_ia_list[i] = self.tasks[i]
            return self.dtheta_ia_list[i]

        # if we computed it earlier pull it out

        if self.dtheta_ia_list[i] is not None:
            return self.dtheta_ia_list[i]

        # otherwise compute and update dtheta_ia
        dtheta_ia = np.dot(npla.pinv(self.J_suc_i(i)), self.tasks[i] - np.dot(self.J_i(i), self.dtheta_1ia(i)))
        self.dtheta_ia_list[i] = dtheta_ia
        return dtheta_ia

    def N_suc_i(self, i):

        return self.null(self.J_suc_i(i))

    def N_suc_1i(self, i):
        product = np.identity(self.joint_dims)

        for j in range(0, i - 1):
            product = np.dot(product, self.N_suc_i(j))

        return product

    def dtheta_1ia(self, i):
        sum_product = np.zeros((self.joint_dims, 1))
        for j in range(0, i - 1):
            sum_product += np.dot(self.N_suc_1i(j - 1), self.dtheta_ia(j))

        return sum_product

    def null(self, J):
        return np.identity(self.joint_dims) - np.dot(npla.pinv(J), J)

    def J_suc_i(self, i):
        if self.J_suc_i_list[i] is not None:
            return self.J_suc_i_list[i]

        J_suc_i = np.dot(self.J_i(i), self.N_suc_1i(i - 1))
        self.J_suc_i_list[i] = J_suc_i
        return J_suc_i

    def J_i(self, i):
        return self.jacobian_getters[i]()

    def clear_time_variants(self):
        """
        Clears storage lists that depend on time
        :return: None
        """
        tasks = len(self.tasks)
        self.dtheta_ia_list = [None]*tasks
        self.dtheta_1ia_list = [None]*tasks
        self.J_suc_i_list = [None]*tasks
        self.N_suc_i_list = [None]*tasks
        self.N_suc_1i_list = [None]*tasks

    def setup_cascaded_pose_vel(self, pose_vel, jacobian_getter, starting_priority=0):
        self.get_full_jacobian = jacobian_getter

        self.add_task(pose_vel[0:3], self.slice_jacobian_position, True, starting_priority)
        self.add_task(pose_vel[3:6], self.slice_jacobian_rotation, True, 1 + starting_priority)

        return self.dtheta_i()

    def update_cascaded_pose_vel(self, pose_vel, starting_priority=0):
        self.update_task(pose_vel[0:3], starting_priority)
        self.update_task(pose_vel[3:6], 1 + starting_priority)
        return self.dtheta_i()

    def slice_jacobian_position(self):
        return np.array(self.get_full_jacobian()[0:3, :])

    def slice_jacobian_rotation(self):
        return np.array(self.get_full_jacobian()[3:6, :])

    def direct_cascaded_pose(self, pose_vel, secondary=True):
        feasibility = 0
        dtheta = np.zeros((7, 1))
        pose_vel = np.array(pose_vel).reshape((-1, 1))

        dx1 = pose_vel[0:3]
        J = self.get_full_jacobian()
        J1 = self.slice_jacobian_position()
        if npla.matrix_rank(J1, tol=0.1) != 3:
            rospy.logdebug("Feasability: %d" % feasibility)
            return dtheta

        feasibility = 1
        J1p = npla.pinv(J1)
        dtheta += J1p.dot(dx1)

        next_J = None
        next_singularity_test = npla.matrix_power

        J2 = self.slice_jacobian_rotation()
        N1 = self.null(J1)
        
        # ingularity_test1 is more conservative
        # singularity_test1 = npla.matrix_rank(J, tol=0.1) == 6
        singularity_test2 = npla.matrix_rank(np.dot(J2, N1), tol=0.1) == 3


        if not singularity_test2 or not secondary:
            rospy.logdebug("Feasability: %d" % feasibility)
            return dtheta

        feasibility = 2
        J2N1p = npla.pinv(np.dot(J2, N1))
        dx2 = pose_vel[3:6]
        dx21_proj = dx2 - multidot(J2, J1p, dx1)

        dtheta2 = multidot(N1, J2N1p, dx21_proj)
        dtheta += dtheta2

        rospy.logdebug("Feasability: %d" % feasibility)
        return dtheta


class EndpointVelocityController():
    """Generates joint velocity commands to produce a specified endpoint velocity for one of
    Baxter's arms.

    Args:
    limb: 'left'|'right'
    """

    def __init__(self, limb, rate):
        self._neutral_config = NEUTRAL[limb]
        self._kin = baxter_kinematics(limb)
        self._limb = baxter_interface.Limb(limb)
        self._nullspace_projector = TaskHeirarchy(7, rate)
        rospy.logdebug(limb)
        self._nullspace_projector.setup_cascaded_pose_vel(np.zeros(7), self._kin.jacobian)
        rospy.logdebug(self._nullspace_projector.get_full_jacobian())

    def set_endpoint_velocity(self, velocity_cmd, secondary=True):
        """Sends joint velocity commands to produce the instantaneous endpoint velocity specified
        by velocity_cmd

        Args:
        velocity_cmd: (6,) ndarray - the desired endpoint velocity [x', y', z', r', p' y']
        """
        # End effector command velocity
        # Compute the joint velocity commands
        joint_velocity_command = self._nullspace_projector.direct_cascaded_pose(velocity_cmd, secondary)


        #Clamp the EE velocity command so it's within limits for each joint
        ee_vel_scale_factor = max(abs(joint_velocity_command.flatten())) / MAX_JOINT_VEL

        if True:
            if ee_vel_scale_factor > 1:
                joint_velocity_command /= ee_vel_scale_factor

        command_velocity_dict = {joint: joint_velocity_command[i, 0] for i, joint in
                                 enumerate(self._limb.joint_names())}

        self._limb.set_joint_velocities(command_velocity_dict)

    def set_neutral_config(self, neutral_config_dict):
        """Sets the arm's neutral configuration to something other than the default value.

        Args:
        neutral_config_dict: dict - {'left_s1':value, ..., 'left_w2':value}
        """
        self._neutral_config = neutral_config_dict


class ContinuousEndpointPoseController():
    """A closed loop endpoint pose controller which uses joint velocity commands to move the
    specified limb's endpoint to the desired pose.

    Args:
    limb: 'left'|'right'
    """

    def __init__(self, limb, rate=100):
        self.rate = rate
        self._limb = baxter_interface.Limb(limb)
        self._vel_controller = EndpointVelocityController(limb, rate)
        self._run = True

        # Choose the endpoint's current pose as the starting setpoint
        self._orientation_setpoint = tuple(self._limb.endpoint_pose()['orientation'])
        self._position_setpoint = tuple(self._limb.endpoint_pose()['position'])

        # Create the controllers for each DoF of the endpoint velocity
        self._dof_controllers = {
                'xyz': pid_controllers.PositionPidController(k_p=3.0),
                'rpy': pid_controllers.OrientationPidController(k_p=1.0)
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
        current_ee_vel = self._limb.endpoint_velocity()['linear']
        endpoint_pose = self._limb.endpoint_pose()
        current_ee_pos = list(endpoint_pose['position'])
        current_ee_ori = list(endpoint_pose['orientation'])
        #Seems like the quaternion is actually in the correct order?
        # current_ee_ori = np.concatenate((current_ee_ori[3:4], current_ee_ori[0:3]))
        rospy.logdebug("===================")
        rospy.logdebug(self._limb.name)
        rospy.logdebug("target:  " + str(self._position_setpoint + tuple(self._orientation_setpoint)))
        rospy.logdebug("current: " + str(current_ee_pos + current_ee_ori))
        
        pos_velocity_cmd = self._dof_controllers['xyz'].get_control_cmd(current_ee_pos, current_ee_vel,
                                                                        reference=self._position_setpoint)
        ori_velocity_cmd = self._dof_controllers['rpy'].get_control_cmd(current_ee_ori,
                                                                        reference=self._orientation_setpoint)
        velocity_cmd = pos_velocity_cmd + ori_velocity_cmd
        rospy.logdebug("command: " + str(velocity_cmd))
        self._vel_controller.set_endpoint_velocity(velocity_cmd)
        #self._vel_controller.set_endpoint_velocity([.0, 0.0, -0.05] + [0.0]*3, True)

    def _run_controller(self):
        rate = rospy.Rate(self.rate)
        while (not rospy.is_shutdown()) and self._run:
            self._update_endpoint_velocity()
            rate.sleep()

    def update_pose_setpoint(self, setpoint):
        """Sets the endpoint position setpoint to a new value. Right now orientation is not
        supported. The controller just maintains the endpoint orientation that it measured when
        it was initialized.

        Args:
        setpoint: (6,) array-like - the desired endpoint pose [x, y, z, r, p, y]
        """
        self._position_setpoint = setpoint[:3]
        self._orientation_setpoint = quaternion_from_euler(*setpoint[3:])

    def update_pose_setpoint_cb(self, msg):
        self.update_pose_setpoint(msg.data)


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


def multidot(*args):
    if len(args) == 2:
        return np.dot(*args)

    return np.dot(multidot(*args[0:-1]), args[-1])


def main():
    rospy.init_node('baxter_velocity_control', log_level=rospy.DEBUG)
    rate = rospy.get_param('~rate', 100)
    #vel_controller_left = ContinuousEndpointPoseController('left', rate)
    vel_controller_right = ContinuousEndpointPoseController('right', rate)
    #sub_left = rospy.Subscriber('left/pose/reference', Float32MultiArray, vel_controller_left.update_pose_setpoint_cb)
    sub_right = rospy.Subscriber('right/pose/reference', Float32MultiArray, vel_controller_right.update_pose_setpoint_cb)
    rospy.spin()

if __name__ == "__main__":
    main()
