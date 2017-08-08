#!/usr/bin/python
import baxter_force_control
from extra_baxter_tools.motion import EETrajectoryRecorder
from extra_baxter_tools.kinematics import ExtendedBaxterKinematics
from baxter_kinematics import baxter_kinematics
import rospy
from extra_baxter_tools.joint_trajectory_client import BaxterJointTrajectory, send_multi_dof_trajectory

rospy.init_node('recorder')

limb = 'right'
recorder = EETrajectoryRecorder(limb, send_multi_dof_trajectory)
ext_kin = ExtendedBaxterKinematics(limb)
joint_traj = BaxterJointTrajectory(limb)

recorder.record()
joint_traj_msg, real_traj, deviation = ext_kin.invert_trajectory_msg(recorder.draw(20.0))
print('Deviation: %s' % deviation)
joint_traj.set_trajectory(joint_traj_msg)
joint_traj.saturate_trajectory()
joint_traj.start()
joint_traj.wait()
print("Exiting - Joint Trajectory Action Test Complete")

while True:
	if raw_input('Play again?') == 'y':
		joint_traj.start()
		joint_traj.wait()
	else:
		break