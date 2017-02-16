#!/usr/bin/python
import rospy
import baxter_interface
from baxter_pykdl import baxter_kinematics
import numpy as np
import numpy.linalg as npla
from math import sqrt

WRENCH = [10,0,0,0,0,0]
JOINTS = ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2']


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


    

if __name__ == "__main__":
    main()
