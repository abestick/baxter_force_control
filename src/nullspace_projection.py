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


class TaskHeirarchy():
    """
    Stores task space or joint space velocity vectors in a priority order and successively computes the optimal joint
    angles to perform the tasks
    """

    def __init__(self, joint_dims):
        """
        
        :param joint_dims: the number of joints 
        """
        self.joint_dims = joint_dims
        self.tasks = []
        self.taskspace_flags = []
        self.jacobian_getters =[]

        # These are placeholders for storing variables used multiple times to optimize calculation
        self.dtheta_ia_list = []
        self.dtheta_1ia_list = []
        self.J_suc_i_list = []
        self.N_suc_i_list = []
        self.N_suc_1i_list = []

        # list references to time variants
        self.time_variants = [self.dtheta_ia_list, self.dtheta_1ia_list, self.J_suc_i_list, self.N_suc_i_list,
                              self.N_suc_1i_list]

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
        self.tasks[priority] = task.copy()

    def update_tasks_dict(self, tasks_dict):
        for priority in tasks_dict:
            self.update_task(tasks_dict[priority], priority)

    def dtheta_i(self, i=1):

        # clear
        self.clear_time_variants()

        # BASE CASE: i=K
        if i == len(self.tasks):
            return self.dtheta_ia(i)

        # RECURSIVE CASE:
        return self.dtheta_ia(i) + np.dot(self.N_suc_i(i), self.dtheta_i(i+1))

    def dtheta_ia(self, i):

        # if the task is in joint space
        if not self.taskspace_flags[i]:
            self.dtheta_ia_list[i] = self.tasks[i][0]
            return self.dtheta_ia_list[i]

        # if we computed it earlier pull it out
        if self.dtheta_ia_list[i] is not None:
            return self.dtheta_ia_list[i]

        # otherwise compute and update dtheta_ia
        dtheta_ia = np.dot(npla.pinv(self.J_suc_i(i)), self.tasks[i][0] - np.dot(self.J_i(i), self.dtheta_1ia(i)))
        self.dtheta_ia_list[i] = dtheta_ia
        return dtheta_ia

    def N_suc_i(self, i):

        return self.null(self.J_suc_i(i))

    def N_suc_1i(self, i):
        product = np.identity(self.joint_dims)

        for j in range(1, i - 1):
            product = np.dot(product, self.N_suc_i(j))

        return product

    def dtheta_1ia(self, i):
        sum_product = 0
        for j in range(1, i - 1):
            sum_product += np.dot(self.N_suc_1i(j - 1), self.dtheta_ia(j))

        return sum_product

    def null(self, J):
        return np.identity(self.joint_dims) - np.dot(npla.pinv(J), J)

    def J_suc_i(self, i):
        if self.J_suc_i_list[i] is not None:
            return self.J_suc_i_list[i]

        J_suc_i = np.dot(self.J_i(i), self.N_suc_1i(i-1))
        self.J_suc_i_list[i] = J_suc_i
        return J_suc_i

    def J_i(self, i):
        return self.jacobian_getters[i]()

    def clear_time_variants(self):
        """
        Clears storage lists that depend on time
        :return: None
        """
        for time_variant in self.time_variants:
            for i in range(len(self.tasks)):
                time_variant[i] = None

    def setup_cascaded_pose_vel(self, pose_vel, jacobian_getter, starting_priority=1):
        self.get_full_jacobian = jacobian_getter

        self.add_task(pose_vel[0:3], self.slice_jacobian_position, True)
        self.add_task(pose_vel[3:6], self.slice_jacobian_rotation, True)

        return self.dtheta_i()

    def update_cascaded_pose_vel(self, pose_vel):
        self.update_task(pose_vel[0:3])
        self.update_task(pose_vel[3:6])
        return self.dtheta_i()

    def slice_jacobian_position(self):
        return self.get_full_jacobian[0:3, :]

    def slice_jacobian_rotation(self):
        return self.get_full_jacobian[3:6, :]
