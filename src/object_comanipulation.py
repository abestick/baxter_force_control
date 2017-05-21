#!/usr/bin/python

import numpy as np
import numpy.linalg as npla

import rospy
from geometry_msgs.msg import Twist
from pykdl_utils.kdl_kinematics import KDLKinematics
from scipy.optimize import minimize
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
from urdf_parser_py.urdf import URDF

from kinmodel.tools import colvec


class ObjectOptimizer(object):

    default_alpha = np.array([0.6, 0.2, 0, 0.2])

    def __init__(self, num_joints, get_object_human_jacobian, get_object_robot_jacobian, joint_angles_topic,
                 configuration_ref, intent, initial_alpha=None, get_initial_guess=None, k_p=1.0):
        self.num_joints = num_joints
        self.get_object_human_jacobian = get_object_human_jacobian
        self.get_object_robot_jacobian = get_object_robot_jacobian
        self.configuration_ref = colvec(configuration_ref)
        self.intent = colvec(intent)
        self.alpha = np.array(initial_alpha) if initial_alpha is not None else self.default_alpha.copy()
        self.k_p = k_p
        self.sub_config = None
        self.sub_intent = None
        self.pub_costs = rospy.Publisher('~costs', Float32MultiArray, queue_size=100)

        if get_initial_guess is None:
            self.get_initial_guess = self._get_config_ref
        else:
            self.get_initial_guess = get_initial_guess

        # block until we get out first joint update
        self.joints = colvec(rospy.wait_for_message(joint_angles_topic, JointState).position)

        # set up a subscriber to this joint topic
        self.sub_joints = rospy.Subscriber(joint_angles_topic, JointState, self._update_joints)

    def manipulability_cost(self, q):
        return npla.norm(np.dot(npla.pinv(self.get_object_human_jacobian(np.squeeze(q))[:3, :]), self.intent))

    def configuration_cost(self, q):
        return npla.norm(colvec(q)-self.configuration_ref)

    def joint_movement_cost(self, q):
        return npla.norm(q - self.joints)

    def effector_movement_cost(self, q):
        return npla.norm(self.get_object_robot_jacobian(np.squeeze(self.joints)).dot(colvec(q) - self.joints))

    def total_cost(self, q):
        cost_array = np.array([self.manipulability_cost(q), self.configuration_cost(q),
                               self.joint_movement_cost(q), self.effector_movement_cost(q)])

        if rospy.get_param('~publish_costs', False):
            self.pub_costs.publish(Float32MultiArray(data=cost_array))

        return np.sum(self.alpha * cost_array)

    def set_dynamic_configuration_ref(self, topic_name):
        self.sub_config = rospy.Subscriber(topic_name, JointState, self._update_configuration_ref)

    def set_dynamic_intent(self, topic_name):
        self.sub_config = rospy.Subscriber(topic_name, Twist, self._update_intent)

    def set_static_configuration_ref(self, configuration_ref=None):
        if self.sub_config is not None:
            self.sub_config.unregister()
            self.sub_config = None

        if configuration_ref is not None:
            self.configuration_ref = colvec(configuration_ref)

    def set_static_intent(self, intent=None):
        if self.sub_intent is not None:
            self.sub_intent.unregister()
            self.sub_intent = None

        if intent is not None:
            self.intent = colvec(intent)

    def _update_configuration_ref(self, msg):
        self.configuration_ref = colvec(msg.position)

    def _update_intent(self, msg):
        self.intent = colvec([msg.linear.x, msg.linear.y, msg.linear.z])

    def _update_joints(self, msg):
        # copy across angles into a column vector
        self.joints = colvec(msg.position)

    def _get_zeros(self):
        return np.zeros(self.num_joints)

    def _get_config_ref(self):
        return self.configuration_ref.flatten()

    def compute_optimal_joint_anlges(self, initial_guess=None):
        if initial_guess is None:
            initial_guess = self.get_initial_guess()
        return colvec(minimize(self.total_cost, initial_guess).x)

    def compute_optimal_joint_velocities(self, initial_guess=None):
        return self.k_p * (self.compute_optimal_joint_anlges(initial_guess) - self.joints)


class BoxOptimizer(ObjectOptimizer):

    def __init__(self, urdf, base_link='base_link', endpoint='human', initial_alpha=None):
        self.robot = URDF.from_xml_file(urdf)
        self.kdl_kin = KDLKinematics(self.robot, base_link, endpoint)
        num_joints = self.kdl_kin.num_joints
        super(BoxOptimizer, self).__init__(num_joints, self.kdl_kin.jacobian, self.kdl_kin.jacobian, 'joint_states',
                                           np.zeros(num_joints), np.random.random(3),initial_alpha,
                                           self.kdl_kin.random_joint_angles)

    def compute_optimal_robot_effector_position(self, initial_joints_guess=None):
        optimal_joints = self.compute_optimal_joint_anlges(initial_joints_guess)
        return self.kdl_kin.FK(optimal_joints)

    def compute_optimal_robot_effector_velocity(self, initial_joints_guess=None):
        optimal_joint_vels = self.compute_optimal_joint_velocities(initial_joints_guess)
        return self.kdl_kin.jacobian(np.squeeze(self.joints)).dot(np.squeeze(optimal_joint_vels))

    def timed_compute_effector_velocity(self, event):
        rospy.logdebug("last duration: %r" % event.last_duration)
        rospy.logdebug(self.compute_optimal_robot_effector_velocity())
        rospy.logdebug("---------------------------------------------")


if __name__ == '__main__':
    rospy.init_node('box_opt')
    rospy.set_param('~publish_costs', True)
    box_opt = BoxOptimizer("/home/pedge/catkin_ws/src/baxter_force_control/urdf/box_chain.urdf")
    timer = rospy.Timer(rospy.Duration(0.5), box_opt.timed_compute_effector_velocity)
    rospy.spin()