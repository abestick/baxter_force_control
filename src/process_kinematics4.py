#!/usr/bin/env python
import argparse
import warnings
import os
import rospy
import kinmodel
import numpy as np
import matplotlib.pyplot as plt
from kinmodel.track_mocap import KinematicTreeTracker, KinematicTreeExternalFrameTracker, MocapFrameTracker
from baxter_force_control.steppables import CostCalculator, BagReader, JointStateMsgReader, \
    TwistMsgReader, TFMsgFrameReader, JacobianCalculator, TwistTransformer, Vector3MsgReader, RegressorBase, \
    CostGradient, WeightedCostGradients, Merger, Magnitude, WeightedKinematicCostDescentEstimatorBases, \
    Joobie, WeightedKinematicCostDescentController
from rosbag import Bag
from baxter_force_control.system import ForwardBlockNode, ForwardSystem, ForwardRoot
from baxter_force_control.motion_costs import QuadraticDisplacementCost, WeightedCostCombination
from baxter_force_control.tools import *
from std_msgs.msg import Float32
from os.path import expanduser
import json


FRAMERATE = 50
GROUP_NAME = 'tree'
HOME = expanduser("~")

arm_joints = ['shoulder_0', 'shoulder_1', 'shoulder_2', 'elbow']
box_joints = ['joint_0', 'joint_1', 'joint_2', 'joint_3']


def get_time(trial_result):
    return trial_result[0][0]


def get_mocap_array(trial_result):
    return trial_result[3]


def process_kinematics():
    parser = argparse.ArgumentParser()
    parser.add_argument('participant', help='The name of the participant')
    args = parser.parse_args()

    folder_name = raw_input('Save to which folder?\n')

    box_json = HOME + '/experiment/box/new_box_opt.json'
    participant_json = HOME + '/experiment/%s/new_%s_opt.json' % (args.participant, args.participant)
    participant_ref_json = HOME + '/experiment/%s/%s_ergo_ref.json' % (args.participant, args.participant)
    participant_results_dir = HOME + '/experiment/results/%s' % args.participant
    block_diagram_path = participant_results_dir + '/%s_process.gv' % args.participant
    bag_path = participant_results_dir + '/kinematics_extracted2/'
    results_dir = participant_results_dir + '/processed/%s/' % folder_name

    create_dir(results_dir)

    # Load the calibration sequence
    bags = [f for f in os.listdir(bag_path) if f[-4:] == '.bag']
    bags = sorted(bags, key=lambda x: int(x[-5]))

    # Initialize the tree
    box_kin_tree = kinmodel.KinematicTree(json_filename=box_json)
    human_kin_tree = kinmodel.KinematicTree(json_filename=participant_json)

    config_reference_0 = {'joint_0': 0.0,
                          'joint_1': 0.0,
                          'joint_2': 0.0,
                          'joint_3': 0.0}

    # # Hard coded config reference from looking at the raw data
    # config_reference_2 = {'shoulder_0': 1.57,
    #                       'shoulder_1': 1.57,
    #                       'shoulder_2': -1.57,
    #                       'elbow': -1.57}


    # We only want to draw the block diagram once
    block_diag = None
    rospy.init_node('process_kinematics')
    mean_dt = 0.01875
    ergo_trial = Bag(bag_path + bags[0], 'r')
    if os.path.isfile(participant_ref_json):
        with open(participant_ref_json) as f:
            ergo_reference = json.load(f)
    else:
        ergo_reference = find_ergo_reference(ergo_trial, mean_dt)
        with open(participant_ref_json, 'w') as f:
            json.dump(ergo_reference, f)

    # Create a config cost and bases of manipulability costs
    config_cost_1 = QuadraticDisplacementCost('configuration_cost', config_reference_0, lambda x: x, config_reference_0.keys())
    config_cost_2 = QuadraticDisplacementCost('ergonomic_cost', ergo_reference, lambda x: x, ergo_reference.keys())

    for i, trial in enumerate(bags):
        trial_bag = Bag(bag_path + trial, 'r')
        # If its the first element, it is the combined data, change i to ALL and set to plot the block diagram
        if i == 0:
            block_diag = block_diagram_path

        print('Learning Trial: %s...\n' % trial)

        # Learn the system and draw the block diagram
        trajectories = process(trial_bag, box_kin_tree, human_kin_tree, mean_dt, config_cost_1, config_cost_2, block_diag)
        for name, traj in trajectories.items():
            traj_dir = results_dir + trial[:-4]
            create_dir(traj_dir)
            traj.to_pickle(traj_dir + '/' + name + '.p')
        block_diag = None


def mean_influence(array, index, lnorm=2):
    without = array[np.arange(len(array)) != index]
    ord = None if lnorm == 2 else lnorm
    return np.linalg.norm(array.mean() - without.mean(), ord=ord)


def variance_influence(array, index, lnorm=2):
    without = array[np.arange(len(array)) != index]
    ord = None if lnorm == 2 else lnorm
    return np.linalg.norm(array.var() - without.var(), ord=ord)


def find_ergo_reference(trial_bag, dt):
    # Create source
    bag_reader = BagReader(trial_bag)
    bag_reader.close()
    # bag_reader_node = ForwardBlockNode(bag_reader, 'Bag Reader', 'msgs')

    # # Initialize the frame tracker and attach frames of interest
    # human_frame_tracker = KinematicTreeExternalFrameTracker(human_kin_tree.copy())

    # chest_grip_twists = [get_twist_from_msgs(msgs, 'chest_grip_twist') for msgs in bag_reader.get_messages()]
    # mag_traj = np.array(map(twist_mag, chest_grip_twists))

    joint_states = np.array([get_joint_states_from_msgs(msgs, 'system_state', arm_joints)
                             for msgs in bag_reader.get_messages()])

    joint_vels = np.diff(joint_states, axis=0)
    joint_vels = np.vstack((np.zeros_like(arm_joints, dtype=float), joint_vels))
    # joint_vel_mag = np.linalg.norm(joint_vels, axis=1)

    # assert joint_vel_mag.shape == mag_traj.shape, '%s %s' % (str(joint_vel_mag.shape), str(mag_traj.shape))

    time = np.arange(0, dt*len(joint_vels), dt)
    relax_on = np.zeros_like(time, dtype=int)
    bent_on = np.zeros_like(time, dtype=int)
    time_after_relax = np.zeros_like(time)

    relax_bent = [
        (4.5, 9.5),
        (14.5, 19.5),
        (25, 30),
        (35.5, 40.5),
        (46, 51),
        (57.5, 62.5),
        (77.5, 82.5),
        (89.0, 94),
        (99.5, 104.5),
        (114, 119),
        (124.5, 129.5),
        (135, 140),
        (147.5, 152.5),
        (158, 163),
        (168.5, 173.5),
        (181, 186),
        (191.5, 196.5),
        (202.5, 207.5)
    ]
    relax_times, bent_times = zip(*relax_bent)
    for relax_start, bent_start in relax_bent:
        rest = time > relax_start
        time_after_relax[rest] = time[rest] - time[rest][0]
        time_after_relax[time > bent_start] = 0

    for start_relax, start_bent in zip(relax_times, bent_times):
        bent_on[time > start_relax] = 0
        relax_on[time > start_relax] = 1
        relax_on[time > start_bent] = 0
        bent_on[time > start_bent] = 1

    time_window = 1.0
    half_window = (int(time_window/dt) - 1) / 2
    # print('Window Size: %d' % (half_window*2 + 1))
    vel_thresh = 0.012
    # vel_used = mag_traj
    # vel_used = joint_vel_mag
    stationary = np.all([relax_on, np.all(abs(joint_vels) < vel_thresh, axis=1), time_after_relax>3], axis=0)
    # plt.scatter(time[stationary], np.zeros_like(time)[stationary], label='stat')

    steady = [np.all(stationary[max(0, i-half_window):min(len(stationary), i+1+half_window)]) for i in range(len(stationary))]
    usable = np.array([np.any(steady[max(0, i-half_window):min(len(stationary), i+1+half_window)]) for i in range(len(steady))])

    # these are the joints that are at rest for more than a second
    ergo_joints_raw = joint_states[usable, :]
    m_influence = np.zeros_like(ergo_joints_raw)
    v_influence = np.zeros_like(ergo_joints_raw)
    for j, joint_traj in enumerate(ergo_joints_raw.T):
        for t in range(len(joint_traj)):
            m_influence[t, j] = mean_influence(joint_traj, t, 2)
            # v_influence[t, j] = variance_influence(joint_traj, t, 2)

    ergo_joints = ergo_joints_raw.copy()
    ergo_joints[m_influence>0.00004] = np.nan
    mean_ergo_joints = np.nanmean(ergo_joints, axis=0)
    var_ergo_joints = np.nanvar(ergo_joints, axis=0)
    ptp_ergo_joints = abs(np.nanmax(ergo_joints, axis=0) - np.nanmin(ergo_joints, axis=0))
    print('Ergonomic Reference: %s' % mean_ergo_joints)
    print('Variance:            %s' % var_ergo_joints)
    print('Range at Rest:       %s' % ptp_ergo_joints)

    plottable_mean_joints = np.tile(mean_ergo_joints, (len(time), 1))

    # plt.plot(time, joint_vel_mag, label='joint_speeds speed')
    # plt.scatter(time[usable], mag_traj[usable], label='Usable speed', color='k')
    # plt.plot(time, relax_on, label='Relax ON')
    # plt.plot(time, bent_on, label='Bent ON')
    if True:
        for label, traj, mtraj, etraj, vtraj in zip(arm_joints, joint_states.T, plottable_mean_joints.T, ergo_joints.T, joint_vels.T):
            plt.plot(time, traj, label=label)
            plt.plot(time, vtraj, label='d_' + label)
            plt.plot(time, mtraj, label='mean_' + label)
            plt.scatter(time[usable][~np.isnan(etraj)], traj[usable][~np.isnan(etraj)], label='used_' + label, color='r')

            plt.legend()
            plt.show()

    # human_frame_tracker.set_split_joint('shoulder')
    # human_frame_tracker.attach_frame('elbow', 'forearm', tf_pub=False)
    # human_frame_tracker.attach_frame('base', 'chest', tf_pub=False)

    return dict(zip(arm_joints, mean_ergo_joints))


def attach_frames(bag_reader, box_kin_tree, human_kin_tree):
    input_features = box_kin_tree.get_joints()['joint_3'].children
    input_points = [np.array(p.primitive) for p in input_features]
    grip_location = grip_point(input_points)

    # Initialize the frame tracker and attach frames of interest
    box_frame_tracker = KinematicTreeExternalFrameTracker(box_kin_tree.copy())
    human_frame_tracker = KinematicTreeExternalFrameTracker(human_kin_tree.copy(), base_frame_name='chest')

    grip_forearm_transform = get_transform_from_msgs(bag_reader.get_messages()[0], 'grip', 'forearm')
    grip_forearm_transform._reference_frame = ''
    grip_forearm_transform._target = ''

    human_frame_tracker.set_split_joint('shoulder')
    human_frame_tracker.attach_frame('elbow', 'forearm', tf_pub=False)
    human_frame_tracker.attach_frame('base', 'chest', tf_pub=False)
    chest_forearm = human_frame_tracker.observe_frame('forearm')
    chest_grip_0 = chest_forearm * grip_forearm_transform.inv()
    human_frame_tracker.attach_frame('elbow', 'grip', tf_pub=False, pose=chest_grip_0)

    box_frame_tracker.attach_frame('joint_0', 'robot', tf_pub=False)
    box_frame_tracker.attach_frame('joint_3', 'grip', tf_pub=False, position=grip_location)
    return box_frame_tracker, human_frame_tracker


def process(trial_bag, box_kin_tree, human_kin_tree, dt, config_cost, ergo_cost, diagram_filename=None, pos_only=True):
    # Define the root (all source nodes)

    # Create source
    bag_reader = BagReader(trial_bag)
    bag_reader.step()
    bag_reader_node = ForwardBlockNode(bag_reader, 'Bag Reader', 'msgs')
    print('Number of data points: %d' % len(bag_reader))

    box_frame_tracker, human_frame_tracker = attach_frames(bag_reader, box_kin_tree, human_kin_tree)
    root = ForwardRoot([bag_reader_node]) #, zero_twist_node])

    # Create the system
    system = ForwardSystem(root, 1.0/dt)
    time = system.get_time

    x_node = ForwardBlockNode(JointStateMsgReader('system_state'), 'System State Reader', 'system_state')
    bag_reader_node.add_output(x_node, 'msgs')

    config_cost_node = ForwardBlockNode(CostCalculator(config_cost), "Config Cost Calculator", 'config_cost')
    ergo_cost_node = ForwardBlockNode(CostCalculator(ergo_cost), "Ergonomic Cost Calculator", 'ergonomic_cost')
    x_node.add_output(config_cost_node, 'states')
    x_node.add_output(ergo_cost_node, 'states')

    static_input_twist_reader = TwistMsgReader('grip_twist')
    static_input_twist_node = ForwardBlockNode(static_input_twist_reader, 'Static Input Twist Reader',
                                               'observed_W_G_W_G_twist')
    bag_reader_node.add_output(static_input_twist_node, 'msgs')

    robot_twist_reader = TwistMsgReader('robot_twist')
    robot_twist_node = ForwardBlockNode(robot_twist_reader, 'Robot Twist Reader', 'observed_W_G_W_G_twist')
    bag_reader_node.add_output(robot_twist_node, 'msgs')

    input_mag = Magnitude(element_wise=True, pos_only=True)
    input_mag_node = ForwardBlockNode(input_mag, 'Input Twist Magnitude Calculator', 'input_twist_mag')
    static_input_twist_node.add_output(input_mag_node, 'states')

    robot_mag = Magnitude(element_wise=True, pos_only=True)
    robot_mag_node = ForwardBlockNode(robot_mag, 'Robot Twist Magnitude Calculator', 'robot_twist_mag')
    robot_twist_node.add_output(robot_mag_node, 'states')

    robot_chest_twist_reader = TwistMsgReader('robot_chest_twist')
    R_H_v_R_H_node = ForwardBlockNode(robot_chest_twist_reader, 'Robot->Chest Twist Reader', 'R_C_R_C_twist')
    bag_reader_node.add_output(R_H_v_R_H_node, 'msgs')

    robot_chest_reader = TFMsgFrameReader('robot', 'chest')
    RH_node = ForwardBlockNode(robot_chest_reader, 'Robot Chest Frame Reader', 'robot_chest')
    bag_reader_node.add_output(RH_node, 'msgs')

    robot_world_reader = TFMsgFrameReader('world', 'robot', inv=True)
    RW_node = ForwardBlockNode(robot_world_reader, 'Robot World Frame Reader', 'robot_world')
    bag_reader_node.add_output(RW_node, 'msgs')

    robot_grip_chest_reader = Vector3MsgReader('robot_grip_chest_vec')
    robot_grip_chest_vec_node = ForwardBlockNode(robot_grip_chest_reader, 'Robot Grip->Chest Reader',
                                                 'robot_grip_chest_vec')
    bag_reader_node.add_output(robot_grip_chest_vec_node, 'msgs')

    robot_chest_translator = TwistTransformer(rot_trans='trans')
    R_G_v_R_H_node = ForwardBlockNode(robot_chest_translator, 'Twist Translator R_GC',
                                                   'R_G_R_C_twist')
    R_H_v_R_H_node.add_output(R_G_v_R_H_node, 'twist')
    robot_grip_chest_vec_node.add_output(R_G_v_R_H_node, 'translation')

    config_cost_descent = CostGradient(config_cost, neg=True)
    config_cost_descent_node = ForwardBlockNode(config_cost_descent, 'Config Cost Gradient', 'ideal_box_states')
    x_node.add_output(config_cost_descent_node, 'states')

    state_deriv_node = ForwardBlockNode(Differentiator(dt, 0.25), 'System State Differentiator', 'd_system_state')
    x_node.add_output(state_deriv_node, 'states')
    arm_joint_mag = Magnitude(element_wise=False, name='arm_joint_mag', keys=arm_joints)
    arm_joint_mag_node = ForwardBlockNode(arm_joint_mag, 'Arm Joint Magnitude Calculator', 'arm_joints_mag')
    state_deriv_node.add_output(arm_joint_mag_node, 'states')

    box_joint_mag = Magnitude(element_wise=False, name='box_joint_mag', keys=box_joints)
    box_joint_mag_node = ForwardBlockNode(box_joint_mag, 'Box Joint Magnitude Calculator', 'box_joints_mag')
    state_deriv_node.add_output(box_joint_mag_node, 'states')

    box_jac = JacobianCalculator(box_frame_tracker, 'robot', 'grip', position_only=pos_only)
    J_RG_node = ForwardBlockNode(box_jac, 'Robot to Grip Box Jacobian',
                                 'config_v_robot_grip' if pos_only else 'R_J_RG')
    x_node.add_output(J_RG_node, 'states')

    arm_jac = JacobianCalculator(human_frame_tracker, 'chest', 'grip', position_only=pos_only)
    H_J_HG_node = ForwardBlockNode(arm_jac, 'Chest to Grip Arm Jacobian', 'C_J_CG')
    x_node.add_output(H_J_HG_node, 'states')
    
    RH_rotator = Transformer()
    J_HG_node = ForwardBlockNode(RH_rotator, 'RH Rotator', 'R_J_RH')
    H_J_HG_node.add_output(J_HG_node, 'primitives')
    RH_node.add_output(J_HG_node, 'transform')

    weighted_cost = WeightedCostCombination('combo_costs', [ergo_cost, config_cost])
    cost_grads = WeightedCostGradients(weighted_cost, neg=True)
    dCdx_node = ForwardBlockNode(cost_grads, 'dC/dx Calculator', 'dCdx')
    x_node.add_output(dCdx_node, 'states')
    
    estimator = WeightedKinematicCostDescentEstimatorBases()
    relative_base_node = ForwardBlockNode(estimator, 'Relative Weight Estimator', 'weights_relative')
    
    estimator_static = WeightedKinematicCostDescentEstimatorBases(disturbance=False)
    static_base_node = ForwardBlockNode(estimator_static, 'Static Weight Estimator', 'weights_static')
    
    dCdx_node.add_output(relative_base_node, 'cost_descents')
    dCdx_node.add_output(static_base_node, 'cost_descents')
    
    state_deriv_node.add_output(relative_base_node, 'input')
    state_deriv_node.add_output(static_base_node, 'input')
    
    J_RG_node.add_output(relative_base_node, 'J_RG')
    J_RG_node.add_output(static_base_node, 'J_RG')
    J_HG_node.add_output(relative_base_node, 'J_HG')
    J_HG_node.add_output(static_base_node, 'J_HG')

    R_G_v_R_H_node.add_output(relative_base_node, 'RG_V_RH')
    R_G_v_R_H_node.add_output(static_base_node, 'RG_V_RH')

    ws = [1, 5, 10, 50, 100]
    relative_node = ForwardBlockNode(RegressorBase(ws, 'LS'), 'Rel Node', 'rel_weights')
    static_node = ForwardBlockNode(RegressorBase(ws, 'LS'), 'Stat Node', 'stat_weights')
    nnls_relative_node = ForwardBlockNode(RegressorBase(ws, 'NNLS'), 'NNLS Rel Node', 'nnls_rel_weights')
    nnls_static_node = ForwardBlockNode(RegressorBase(ws, 'NNLS'), 'NNLS Stat Node', 'nnls_stat_weights')

    relative_base_node.add_output(relative_node, 'bases')
    relative_base_node.add_output(nnls_relative_node, 'bases')

    static_base_node.add_output(static_node, 'bases')
    static_base_node.add_output(nnls_static_node, 'bases')

    merger = Merger(['_relative', '_static', '_relative_nnls', '_static_nnls'] + ['']*7)
    merger_node = ForwardBlockNode(merger, 'Output Merger', 'output')
    relative_node.add_output(merger_node, 'first')
    static_node.add_output(merger_node, 'second')
    nnls_relative_node.add_output(merger_node, 'third')
    nnls_static_node.add_output(merger_node, 'fourth')
    x_node.add_output(merger_node, 'fifth')
    input_mag_node.add_output(merger_node, 'sixth')
    arm_joint_mag_node.add_output(merger_node, 'seventh')
    box_joint_mag_node.add_output(merger_node, 'eighth')
    robot_mag_node.add_output(merger_node, 'ninth')
    ergo_cost_node.add_output(merger_node, 'tenth')
    config_cost_node.add_output(merger_node, 'eleventh')

    sinks = [merger_node.sink('Merger Sink')]

    weight_nodes = [relative_node, nnls_relative_node, nnls_static_node, static_node]
    base_nodes = [relative_base_node, relative_base_node, static_base_node, static_base_node]
    controller_nodes = [ForwardBlockNode(WeightedKinematicCostDescentController(),
                                         '%s Controller Estimator'% weight_node.get_name()[:-5], 'each_u')
                        for weight_node in weight_nodes]

    joobie_nodes = [ForwardBlockNode(Joobie(), '%s Velocity Estimator'% weight_node.get_name()[:-5], 'each_V')
                    for weight_node in weight_nodes]

    for controller_node, weight_node, base_node, joobie_node in zip(controller_nodes, weight_nodes, base_nodes, joobie_nodes):
        weight_node.add_output(controller_node, 'weights')
        base_node.add_output(controller_node, 'A_info')
        sinks.append(controller_node.sink('%s Sink' % controller_node.get_name()))

        H_J_HG_node.add_output(joobie_node, 'jacobian')
        controller_node.add_output(joobie_node, 'joint_vels')
        sinks.append(joobie_node.sink('%s Sink' % joobie_node.get_name()))


    # Draw the block diagram if requested
    if diagram_filename is not None:
        system.draw(filename=diagram_filename)

    # all the data for every timestep
    system.run(record=False, print_steps=50)

    return {sink.get_name().lower().replace(' ', '_'): sink.steppable().get_data() for sink in sinks}


def get_dict_to_float_msg(key):
    def constructor(dictionary):
        return Float32(data=dictionary[key])

    return constructor


def translation_mag(transform):
    return np.linalg.norm(transform.trans().q())


def twist_mag(twist):
    return np.linalg.norm(twist.nu())


def translation_mean(transforms):
    trans = [transform.trans().q() for transform in transforms]
    return np.mean(trans, axis=0)


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_vector(twist):
    return twist.trans()


if __name__ == '__main__':
    warnings.filterwarnings('error')
    process_kinematics()