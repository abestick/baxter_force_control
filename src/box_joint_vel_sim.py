#!/usr/bin/python
import rospy, tf2_ros
import numpy as np
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped
from copy import copy
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Header
from baxter_core_msgs.msg import EndpointState, JointCommand
from pykdl_utils.kdl_kinematics import KDLKinematics
from urdf_parser_py.urdf import URDF


class LeftRightSubscriber(object):

    def __init__(self, prefix, suffix, message_type, cb):
        self.sub_l = rospy.Subscriber(prefix + '/left/' + suffix, message_type, cb)
        self.sub_r = rospy.Subscriber('/robot/limb/right/joint_command', message_type, cb)

    def unregister(self):
        self.sub_l.unregister()
        self.sub_r.unregister()

class JointVelocityRelay():

    name = ['head_pan', 'right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1',
                                  'right_w2', 'left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1',
                                  'left_w2', 'l_gripper_l_finger_joint', 'l_gripper_r_finger_joint',
                                  'r_gripper_l_finger_joint', 'r_gripper_r_finger_joint', 'joint_1', 'joint_2', 'joint_3']
    start_pos = JointState(position=np.array([0.0, -0.535688844727944, -0.0003125999999995521, 0.0,
                                              -0.00010840000000000155, 0.0,-6.462112780614149e-05, 0.0, 0.0,
                                              -0.0003125999999995521, 0.0,-0.00010840000000000155, -1.5708,
                                              -6.462112780614149e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                           velocity=np.array([0.0]*22),
                           effort=np.array([0.0]*22),
                           name=name)

    def __init__(self, rate=100):

        robot = URDF.from_xml_file("/home/pedge/catkin_ws/src/baxter_force_control/urdf/box.urdf")
        self.kdl_kin = KDLKinematics(robot, 'base_link', 'human')

        self.sub = None
        self.timer = None
        self.vel_control = rospy.get_param('~vel_control', True)
        rospy.set_param("~vel_control", self.vel_control)
        self.rate = rate
        self.current = copy(self.start_pos)
        self.pub = rospy.Publisher('robot/joint_states', JointState, queue_size=100)
        self.srv = rospy.Service('~switch', Empty, self.switch)
        self.left_endpoint_pub = rospy.Publisher('/robot/limb/left/endpoint_state', EndpointState, queue_size=100)
        self.right_endpoint_pub = rospy.Publisher('/robot/limb/right/endpoint_state', EndpointState, queue_size=100)
        self.box_sub = rospy.Subscriber('box/human_grip', PoseStamped, self.update_box_joints)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.endpoint_sub = rospy.Subscriber('tf', TFMessage, self.publish_endpoints)
        if self.vel_control:
            self.vel_init()
        else:
            self.pos_init()

        rospy.loginfo("Initialization complete.")

    def switch(self, req):
        self.shutdown()

        if self.vel_control:
            self.pos_init()

        else:
            self.vel_init()

        self.vel_control = not self.vel_control
        rospy.set_param("~vel_control", self.vel_control)

        return EmptyResponse()

    def vel_init(self):
        self.sub = LeftRightSubscriber('/robot/limb', 'joint_command', JointCommand, self.update_vel)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate), self.publish_vel)

    def vel_init_states(self):
        self.sub = rospy.Subscriber('~joint_states/vel', JointState, self.update_vel)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate), self.publish_vel)

    def pos_init(self):
        self.sub = rospy.Subscriber('~joint_states/pos', JointState, self.update_pos)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate), self.publish_pos)

    def shutdown(self):
        self.sub.unregister()
        self.timer.shutdown()

    def publish_vel(self, event):
        self.current.position += self.current.velocity * self.timer._period.to_sec()
        self.current.header = Header(stamp=rospy.Time.now())
        self.pub.publish(self.current)

    def update_vel(self, msg):
        for command, name in zip(msg.command, msg.names):
            self.current.velocity[self.current.name.index(name)] = command

    def update_vel_states(self, msg):
        self.current.velocity = np.array(msg.velocity)

    def publish_pos(self, event):
        self.current.header = Header(stamp=rospy.Time.now())
        self.pub.publish(self.current)

    def update_pos(self, msg):
        self.current.velocity[:19] = (self.current.position[:19] - np.array(msg.position)[:19]) / self.timer._period.to_sec()
        self.current.position[:19] = np.array(msg.position)[:19]

    def update_box_joints(self, msg):
        self.current.position[19:] = self.kdl_kin.inverse(msg, )

    def publish_endpoints(self, msg):
        try:
            left = self.tf_buffer.lookup_transform('world', 'l_gripper_l_finger_tip',
                                                   self.tf_buffer.get_latest_common_time('world', 'l_gripper_l_finger'))

            right = self.tf_buffer.lookup_transform('world', 'r_gripper_l_finger_tip',
                                                   self.tf_buffer.get_latest_common_time('world', 'r_gripper_l_finger'))

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException,
                tf2_ros.TransformException):
            return

        left_msg = EndpointState(header=left.header, pose=transform_to_pose(left.transform))
        right_msg = EndpointState(header=right.header, pose=transform_to_pose(right.transform))

        self.left_endpoint_pub.publish(left_msg)
        self.right_endpoint_pub.publish(right_msg)


def transform_to_pose(transform):
    pose = Pose()
    pose.position.x = transform.translation.x
    pose.position.y = transform.translation.y
    pose.position.z = transform.translation.z
    pose.orientation.x = transform.rotation.x
    pose.orientation.y = transform.rotation.y
    pose.orientation.z = transform.rotation.z
    pose.orientation.w = transform.rotation.w
    return pose


def main():
    rospy.init_node('box_joint_vel_sim')
    rate = rospy.get_param('~rate', 100)
    relay = JointVelocityRelay(rate)
    rospy.spin()

if __name__ == "__main__":
    main()
