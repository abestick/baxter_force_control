#!/usr/bin/python
import rospy
import baxter_interface
from baxter_pykdl import baxter_kinematics
import numpy as np
import numpy.linalg as npla
from math import sqrt

LIMB = 'left'
WRENCH = [10,0,0,0,0,0]
VEL_COMMAND = [0, 0, 0.1, 0, 0, 0]
MAX_JOINT_VEL = 0.5
MAX_NS_JOINT_VEL = 0.5

NEUTRAL = {'left': 
        {'left_w0': -0.09625729443980971,
         'left_w1': -0.7029466960484908,
         'left_w2': 0.03528155812136451,
         'left_e0': 0.057524279545703015,
         'left_e1': 1.131694326262464,
         'left_s0': -0.8736020587007431,
         'left_s1': -0.4670971499111085}}

def null(a, rtol=1e-5):
    u, s, v = npla.svd(a)
    rank = (s > rtol*s[0]).sum()
    return rank, v[rank:].T.copy()

def project_to_nullspace(nullspace, joint_order, joint_velocities):
    # Make a vector of joint velocities
    velocities = np.array([joint_velocities[joint] for joint in joint_order])

    # Compute the projection onto each nullspace vector
    proj_velocities = np.zeros_like(velocities)
    for n in range(nullspace.shape[1]):
        proj_velocities += nullspace[:,n].dot(velocities) * nullspace[:,n]

    # Convert back to a velocities dict
    return {joint: proj_velocities[i] for i, joint in enumerate(joint_order)}

def main():
    rospy.init_node('baxter_velocity_control')
    kin = baxter_kinematics(LIMB)
    limb = baxter_interface.Limb(LIMB)
    # JOINTS = limb.joint_names()
    # neutral_config = limb.joint_angles()
    # print(JOINTS)
    # print(neutral_config)

    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        # Neutral position return velocity
        # Compute the velocity command
        velocity_dict = {joint: NEUTRAL[LIMB][joint] - limb.joint_angle(joint) for joint in limb.joint_names()}

        #Clamp the velocity command so it's within limits for each joint
        vel_scale_factor = max([abs(velocity) for velocity in velocity_dict.values()]) / MAX_NS_JOINT_VEL
        if vel_scale_factor > 1:
            velocity_dict = {joint: velocity_dict[joint] / vel_scale_factor for joint in limb.joint_names()}

        #Command the velocities
        print(vel_scale_factor)
        print(velocity_dict)
        
        nullspace = null(np.array(kin.jacobian()))[1]
        velocity_dict = project_to_nullspace(nullspace, limb.joint_names(), velocity_dict)
        


        # End effector command velocity
        # Compute the joint velocity commands
        jacobian_pinv = np.array(kin.jacobian_pseudo_inverse())
        joint_velocity_command = jacobian_pinv.dot(VEL_COMMAND)
        command_velocity_dict = {joint: joint_velocity_command[i] for i, joint in enumerate(limb.joint_names())}

        #Clamp the EE velocity command so it's within limits for each joint
        ee_vel_scale_factor = max([abs(velocity) for velocity in command_velocity_dict.values()]) / MAX_JOINT_VEL
        if ee_vel_scale_factor > 1:
            command_velocity_dict = {joint: command_velocity_dict[joint] / ee_vel_scale_factor for joint in limb.joint_names()}

        
       
        # Command the velocities
        velocity_dict = {joint:command_velocity_dict[joint] + velocity_dict[joint] for joint in limb.joint_names()}
        print(velocity_dict)
        limb.set_joint_velocities(velocity_dict)

        rate.sleep()


    

if __name__ == "__main__":
    main()
