#!/usr/bin/python
import rospy
import baxter_interface
from baxter_pykdl import baxter_kinematics
import numpy as np
import numpy.linalg as npla
from math import sqrt

WRENCH = [10,0,0,0,0,0]
JOINTS = ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2']

class CartesianImpedanceController:
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


    def move_to_neutral(self):
        """
        Moves the limb to neutral location.
        """
        self._limb.move_to_neutral()

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
    rospy.init_node('baxter_kinematics')
    print '*** Baxter PyKDL Kinematics ***\n'
    kin = baxter_kinematics('right')
    limb = baxter_interface.Limb('right')

    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        print(kin.jacobian())
        jac = np.array(kin.jacobian())
        print(sqrt(npla.det(jac.dot(jac.T))))
        torques = jac.T.dot(WRENCH)
        torques_dict = {torque[0]:torque[1] for torque in zip(JOINTS, torques)}
        print(torques_dict)
        limb.set_joint_torques(torques_dict)
        rate.sleep()

def get_null_space_torques()

if __name__ == "__main__":
    main()
