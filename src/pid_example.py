#!/usr/bin/env python
import rospy
import baxter_interface
import argparse
from std_msgs.msg import Empty
import pid_controllers


class CSpaceImpedanceController(object):
    """
    Virtual Joint Springs class for torque example.

    @param limb: limb on which to run joint springs example
    @param reconfig_server: dynamic reconfigure server

    JointSprings class contains methods for the joint torque example allowing
    moving the limb to a neutral location, entering torque mode, and attaching
    virtual springs.
    """
    def __init__(self, limb):
        # create our limb instance
        self._limb = baxter_interface.Limb(limb)

        # initialize parameters
        self._springs = dict()
        self._damping = dict()
        self._start_angles = dict()
        self._rate = 1000.0
        self._missed_cmds = 20.0
        self._limb_name = limb

        # create cuff disable publisher
        cuff_ns = 'robot/limb/' + limb + '/suppress_cuff_interaction'
        self._pub_cuff_disable = rospy.Publisher(cuff_ns, Empty, queue_size=1)

        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable()
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        print("Running. Ctrl-c to quit")

    def _update_forces(self):
        """
        Calculates the current angular difference between the start position
        and the current joint positions applying the joint torque spring forces
        as defined on the dynamic reconfigure server.
        """

        # disable cuff interaction
        self._pub_cuff_disable.publish()

        # create our command dict
        cmd = dict()
        # record current angles/velocities
        cur_pos = self._limb.joint_angles()
        cur_vel = self._limb.joint_velocities()
        # calculate current forces
        for joint in self._controllers.keys():
            # spring portion
            cmd[joint] = self._controllers[joint].get_control_cmd(cur_pos[joint], cur_vel[joint])
        # command new joint torques
        self._limb.set_joint_torques(cmd)

    def move_to_neutral(self):
        """
        Moves the limb to neutral location.
        """
        self._limb.move_to_neutral()

    def attach_springs(self):
        """
        Switches to joint torque mode and attached joint springs to current
        joint positions.
        """
        # record initial joint angles
        self._start_angles = self._limb.joint_angles()

        # set control rate
        control_rate = rospy.Rate(self._rate)

        # for safety purposes, set the control rate command timeout.
        # if the specified number of command cycles are missed, the robot
        # will timeout and disable
        self._limb.set_command_timeout((1.0 / self._rate) * self._missed_cmds)

        # Create controllers
        self._controllers = dict()
        self._controllers[self._limb_name + '_s0'] = pid_controllers.PidController(20.0, 0.0, 10.0)
        self._controllers[self._limb_name + '_s1'] = pid_controllers.PidController(20.0, 0.0, 10.0)
        self._controllers[self._limb_name + '_e0'] = pid_controllers.PidController(10.0, 0.0, 5.0)
        self._controllers[self._limb_name + '_e1'] = pid_controllers.PidController(10.0, 0.0, 5.0)
        self._controllers[self._limb_name + '_w0'] = pid_controllers.PidController(5.0, 0.0, 2.0)
        self._controllers[self._limb_name + '_w1'] = pid_controllers.PidController(5.0, 0.0, 2.0)
        self._controllers[self._limb_name + '_w2'] = pid_controllers.PidController(1.0, 0.0, 1.0)

        # Set desired values
        for joint in self._controllers.keys():
            self._controllers[joint].set_desired_value(self._start_angles[joint])

        # loop at specified rate commanding new joint torques
        while not rospy.is_shutdown():
            if not self._rs.state().enabled:
                rospy.logerr("Joint torque example failed to meet "
                             "specified control rate timeout.")
                break
            self._update_forces()
            control_rate.sleep()

    def clean_shutdown(self):
        """
        Switches out of joint torque mode to exit cleanly
        """
        print("\nExiting example...")
        self._limb.exit_control_mode()
        if not self._init_state and self._rs.state().enabled:
            print("Disabling robot...")
            self._rs.disable()


def main():
    """RSDK Joint Torque Example: Joint Springs

    Moves the specified limb to a neutral location and enters
    torque control mode, attaching virtual springs (Hooke's Law)
    to each joint maintaining the start position.

    Run this example on the specified limb and interact by
    grabbing, pushing, and rotating each joint to feel the torques
    applied that represent the virtual springs attached.
    You can adjust the spring constant and damping coefficient
    for each joint using dynamic_reconfigure.
    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l', '--limb', dest='limb', required=True, choices=['left', 'right'],
        help='limb on which to attach joint springs'
    )
    args = parser.parse_args(rospy.myargv()[1:])

    print("Initializing node... ")
    rospy.init_node("rsdk_joint_torque_springs_%s" % (args.limb,))
    js = CSpaceImpedanceController(args.limb)
    # register shutdown callback
    rospy.on_shutdown(js.clean_shutdown)
    js.move_to_neutral()
    js.attach_springs()


if __name__ == "__main__":
    main()
