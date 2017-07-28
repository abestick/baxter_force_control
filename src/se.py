import rospy
import tf
from geometry_msgs.msg import PointStamped, PoseStamped


class TransformRelay(object):

    def __init__(self, topic, new_topic, message_type, desired_frame, listener):
        self.listener = listener
        self.desired_frame = desired_frame
        self.sub = rospy.Subscriber(topic, message_type, self.callback)
        self.pub = rospy.Publisher(new_topic, message_type, queue_size=100)

    def callback(self, msg):
        try:
            (trans,rot) = self.listener.lookupTransform(msg.header.frame_id, self.desired_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

    def transform_point(self, point_msg):
        pass


if __name__ == '__main__':
    rospy.init_node('turtle_tf_listener')

    subscriber_2 = rospy.Subscriber('topic2', PoseStamped, transform)

    publisher_1 = rospy.Publisher('topic1_transformed', PointStamped)
    publisher_2 = rospy.Publisher('topic2_transformed', PoseStamped)

    listener = tf.TransformListener()


    while not rospy.is_shutdown():
        try:
            (trans,rot) = listener.lookupTransform('/turtle2', '/turtle1', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        angular = 4 * math.atan2(trans[1], trans[0])
        linear = 0.5 * math.sqrt(trans[0] ** 2 + trans[1] ** 2)
        cmd = geometry_msgs.msg.Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        turtle_vel.publish(cmd)

        rate.sleep()