#!/usr/bin/env python
import argparse
import warnings
import os
import rospy
import kinmodel
import numpy as np
import matplotlib.pyplot as plt
from kinmodel.track_mocap import KinematicTreeTracker, KinematicTreeExternalFrameTracker, MocapFrameTracker
from baxter_force_control.steppables import CostCalculator, EdgePublisher, BagReader, JointStateMsgReader, \
    TwistMsgReader, TFMsgFrameReader, JacobianOperator, TwistTransformer, RelativeTwist, Regressor, Vector3MsgReader, \
    CostGradient, Mux, Merger, Magnitude, Divider, Vector3Publisher
from rosbag import Bag
from baxter_force_control.system import ForwardBlockNode, ForwardSystem, ForwardRoot
from baxter_force_control.motion_costs import QuadraticDisplacementCost
from baxter_force_control.tools import *
from std_msgs.msg import Float32
from os.path import expanduser
import sys


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
    participant_json = HOME + '/experiment/%s/new_%s_opt_latest.json' % (args.participant, args.participant)
    participant_results_dir = HOME + '/experiment/results/%s' % args.participant
    block_diagram_path = participant_results_dir + '/%s_process.gv' % args.participant
    bag_path = participant_results_dir + '/kinematics_extracted/'
    new_bag_path = participant_results_dir + '/processed/%s/' % folder_name

    create_dir(new_bag_path)

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
    ergo_reference = find_ergo_reference(ergo_trial, mean_dt)

    # Create a config cost and bases of manipulability costs
    config_cost_1 = QuadraticDisplacementCost('configuration_cost', config_reference_0, lambda x: x, config_reference_0.keys())
    config_cost_2 = QuadraticDisplacementCost('ergonomic_cost', ergo_reference, lambda x: x, ergo_reference.keys())

    for i, trial in enumerate(bags):
        trial_bag = Bag(bag_path + trial, 'r')
        # If its the first element, it is the combined data, change i to ALL and set to plot the block diagram
        if i == 0:
            i = 'ALL'
            block_diag = block_diagram_path

        new_bag = Bag(new_bag_path + 'PROCESSED_%s' % trial, 'w')
        print('Saving to %s' % new_bag.filename)

        print('Learning Trial: %s...\n' % trial)

        # Learn the system and draw the block diagram
        trajectories = process(trial_bag, box_kin_tree, human_kin_tree, mean_dt, config_cost_1, config_cost_2, block_diag, new_bag)
        new_bag.close()
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

    # #remove
    # for label, traj, minf, vinf in zip(arm_joints, joint_states.T, m_influence.T, v_influence.T):
    #     # plt.plot(time, traj, label=label)
    #     plt.scatter(time[usable], minf, label='minf_' + label, color='r')
    #     # plt.scatter(time[usable], vinf, label='vinf_' + label)
    #     plt.scatter(time[usable][minf<0.00004], minf[minf<0.00004], label='inf_cut_' + label)
    #     plt.ylim(0, 0.0001)
    #     plt.legend()
    #     plt.show()

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
    if False:
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


def process(trial_bag, box_kin_tree, human_kin_tree, dt, config_cost, ergo_cost, diagram_filename=None, bag=None, pos_only=True):

    # Create source
    bag_reader = BagReader(trial_bag)
    bag_reader_node = ForwardBlockNode(bag_reader, 'Bag Reader', 'msgs')
    print('Number of data points: %d' % len(bag_reader))

    input_features = box_kin_tree.get_joints()['joint_3'].children
    input_points = [np.array(p.primitive) for p in input_features]
    grip_location = grip_point(input_points)

    # Initialize the frame tracker and attach frames of interest
    box_frame_tracker = KinematicTreeExternalFrameTracker(box_kin_tree.copy())
    human_frame_tracker = KinematicTreeExternalFrameTracker(human_kin_tree.copy())

    grip_forearm_transforms = [get_transform_from_msgs(msgs, 'grip', 'forearm') for msgs in bag_reader.get_messages()]
    mag_traj = map(translation_mag, grip_forearm_transforms)
    trans_mean = translation_mean(grip_forearm_transforms)
    print(trans_mean)
    # plt.plot(mag_traj)
    # plt.show()

    human_frame_tracker.set_split_joint('shoulder')
    human_frame_tracker.attach_frame('elbow', 'forearm', tf_pub=False)
    human_frame_tracker.attach_frame('base', 'chest', tf_pub=False)
    chest_forearm = human_frame_tracker.observe_frame('forearm')
    chest_grip_0 = chest_forearm * grip_forearm_transforms[0].inv()
    human_frame_tracker.attach_frame('elbow', 'grip', tf_pub=False, pose=chest_grip_0)

    box_frame_tracker.attach_frame('joint_0', 'robot', tf_pub=False)
    box_frame_tracker.attach_frame('joint_3', 'grip', tf_pub=False, position=grip_location)

    # Define the root (all source nodes)
    root = ForwardRoot([bag_reader_node]) #, zero_twist_node])

    # Create the system
    system = ForwardSystem(root, 1.0/dt)
    time = system.get_time

    system_state_node = ForwardBlockNode(JointStateMsgReader('system_state'), 'System State Reader', 'system_state')
    bag_reader_node.add_output(system_state_node, 'msgs')

    config_cost_node = ForwardBlockNode(CostCalculator(config_cost), "Config Cost Calculator", 'config_cost')
    ergo_cost_node = ForwardBlockNode(CostCalculator(ergo_cost), "Ergonomic Cost Calculator", 'ergonomic_cost')
    system_state_node.add_output(config_cost_node, 'states')
    system_state_node.add_output(ergo_cost_node, 'states')
    config_cost_node.add_raw_output(EdgePublisher('configuration_cost', Float32, get_dict_to_float_msg(config_cost.name), bag,
                                             system.get_time), 'Cost 1 Publisher', None, 'states')
    ergo_cost_node.add_raw_output(EdgePublisher('ergonomic_cost', Float32, get_dict_to_float_msg(ergo_cost.name), bag,
                                             system.get_time), 'Cost 2 Publisher', None, 'states')

    relative_input_twist_reader = TwistMsgReader('robot_grip_twist')
    relative_input_twist_node = ForwardBlockNode(relative_input_twist_reader, 'Relative Input Twist Reader', 'observed_R_G_R_G_twist')
    bag_reader_node.add_output(relative_input_twist_node, 'msgs')

    static_input_twist_reader = TwistMsgReader('grip_twist')
    static_input_twist_node = ForwardBlockNode(static_input_twist_reader, 'Static Input Twist Reader', 'observed_W_G_W_G_twist')
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
    robot_chest_twist_node = ForwardBlockNode(robot_chest_twist_reader, 'Robot->Chest Twist Reader', 'R_C_R_C_twist')
    bag_reader_node.add_output(robot_chest_twist_node, 'msgs')

    robot_chest_reader = TFMsgFrameReader('robot', 'chest')
    robot_chest_node = ForwardBlockNode(robot_chest_reader, 'Robot Chest Frame Reader', 'robot_chest')
    bag_reader_node.add_output(robot_chest_node, 'msgs')

    robot_world_reader = TFMsgFrameReader('world', 'robot', inv=True)
    robot_world_node = ForwardBlockNode(robot_world_reader, 'Robot World Frame Reader', 'robot_world')
    bag_reader_node.add_output(robot_world_node, 'msgs')

    robot_grip_chest_reader = Vector3MsgReader('robot_grip_chest_vec')
    robot_grip_chest_vec_node = ForwardBlockNode(robot_grip_chest_reader, 'Robot Grip->Chest Reader',
                                                 'robot_grip_chest_vec')
    bag_reader_node.add_output(robot_grip_chest_vec_node, 'msgs')

    robot_chest_translator = TwistTransformer(rot_trans='trans')
    robot_chest_translator_node = ForwardBlockNode(robot_chest_translator, 'Twist Translator R_GC',
                                                   'R_G_R_C_twist')
    robot_chest_twist_node.add_output(robot_chest_translator_node, 'twist')
    robot_grip_chest_vec_node.add_output(robot_chest_translator_node, 'translation')

    config_cost_descent = CostGradient(config_cost, neg=True)
    config_cost_descent_node = ForwardBlockNode(config_cost_descent, 'Config Cost Gradient', 'ideal_box_states')
    system_state_node.add_output(config_cost_descent_node, 'states')

    state_deriv_node = ForwardBlockNode(Differentiator(dt), 'System State Differentiator', 'd_system_state')
    system_state_node.add_output(state_deriv_node, 'states')
    arm_joint_mag = Magnitude(element_wise=False, name='arm_joint_mag', keys=arm_joints)
    arm_joint_mag_node = ForwardBlockNode(arm_joint_mag, 'Arm Joint Magnitude Calculator', 'arm_joints_mag')
    state_deriv_node.add_output(arm_joint_mag_node, 'states')

    box_joint_mag = Magnitude(element_wise=False, name='box_joint_mag', keys=box_joints)
    box_joint_mag_node = ForwardBlockNode(box_joint_mag, 'Box Joint Magnitude Calculator', 'box_joints_mag')
    state_deriv_node.add_output(box_joint_mag_node, 'states')

    ergo_cost_descent = CostGradient(ergo_cost, neg=True)
    ergo_cost_descent_node = ForwardBlockNode(ergo_cost_descent, 'Ergo Cost Gradient', 'ideal_arm_states')
    system_state_node.add_output(ergo_cost_descent_node, 'states')

    box_jac = JacobianOperator(box_frame_tracker, 'robot', 'grip', position_only=pos_only)
    config_predictor_node = ForwardBlockNode(box_jac, 'Robot to Grip Box Jacobian',
                                             'config_v_robot_grip' if pos_only else 'config_R_G_R_G_twist')
    system_state_node.add_output(config_predictor_node, 'states')
    config_cost_descent_node.add_output(config_predictor_node, 'velocities')
    config_predictor_node.add_raw_output(Vector3Publisher('config_cost_descent', bag, time),
                                         'Config Predictor Publisher', None, 'states')


    arm_jac = JacobianOperator(human_frame_tracker, 'chest', 'grip')
    arm_jac_node = ForwardBlockNode(arm_jac, 'Chest to Grip Arm Jacobian', 'C_G_C_G_twist')
    system_state_node.add_output(arm_jac_node, 'states')
    ergo_cost_descent_node.add_output(arm_jac_node, 'velocities')

    chest_grip_rotator = TwistTransformer(rot_trans='rot')
    chest_grip_rotator_node = ForwardBlockNode(chest_grip_rotator, 'Twist Rotator RC', 'R_G_C_G_twist')
    arm_jac_node.add_output(chest_grip_rotator_node, 'twist')
    robot_chest_node.add_output(chest_grip_rotator_node, 'rotation')

    world_grip_rotator = TwistTransformer(rot_trans='rot')
    world_grip_rotator_node = ForwardBlockNode(world_grip_rotator, 'Twist Rotator RW', 'R_G_W_G_twist')
    static_input_twist_node.add_output(world_grip_rotator_node, 'twist')
    robot_world_node.add_output(world_grip_rotator_node, 'rotation')

    relative_twist_node = ForwardBlockNode(RelativeTwist(), 'Relative Twist Sum', 'ergo_R_G_R_G_twist')
    robot_chest_translator_node.add_output(relative_twist_node, 'first')
    chest_grip_rotator_node.add_output(relative_twist_node, 'second')

    if pos_only:
        ergo_predictor_node = ForwardBlockNode(Modifier(get_vector), 'Robot Robot->Grip Vector Ergo Extractor', 'ergo_v_robot_grip')
        relative_twist_node.add_output(ergo_predictor_node, 'states')
        ergo_predictor_node.add_raw_output(Vector3Publisher('ergo_cost_descent', bag, time),
                                           'Ergo Predictor Publisher', None, 'states')
    else:
        ergo_predictor_node = relative_twist_node

    predictors_node = ForwardBlockNode(Mux('ergonomic', 'configuration'), 'Predictors Mux', 'predictors')
    ergo_predictor_node.add_output(predictors_node, 'first')
    config_predictor_node.add_output(predictors_node, 'second')

    if pos_only:
        static_observation_node = ForwardBlockNode(Modifier(get_vector), 'Robot World->Grip Vector Obs Extractor', 'obs_v_world_grip')
        relative_observation_node = ForwardBlockNode(Modifier(get_vector), 'Robot Robot->Grip Vector Obs Extractor', 'obs_v_robot_grip')
        world_grip_rotator_node.add_output(static_observation_node, 'states')
        relative_input_twist_node.add_output(relative_observation_node, 'states')
        relative_observation_node.add_raw_output(Vector3Publisher('observed_input', bag, time),
                                                 'Observed Input Publisher', None, 'states')

    else:
        static_observation_node = world_grip_rotator_node
        relative_observation_node = relative_input_twist_node

    window_sizes = [1, 10, 50, int(0.8*len(bag_reader))]

    static_regressor = Regressor(window_sizes, method='NNLS')
    static_regressor_node = ForwardBlockNode(static_regressor, 'Static Weight Regressor', 'weights_static')
    static_observation_node.add_output(static_regressor_node, 'observation')
    predictors_node.add_output(static_regressor_node, 'predictors')

    relative_regressor = Regressor(window_sizes, method='NNLS')
    relative_regressor_node = ForwardBlockNode(relative_regressor, 'Relative Weight Regressor', 'weights_relative')
    relative_observation_node.add_output(relative_regressor_node, 'observation')
    predictors_node.add_output(relative_regressor_node, 'predictors')

    static_regressor_node.add_raw_output(JointPublisher('weights_static', bag, get_time=time),
                                         'Static Weights Publisher', None, 'states')
    relative_regressor_node.add_raw_output(JointPublisher('weights_relative', bag, get_time=time),
                                           'Relative Weights Publisher', None, 'states')

    for n in window_sizes:
        static_divider = Divider('ergonomic_%d' % n, 'configuration_%d' % n)
        static_divider_node = ForwardBlockNode(static_divider, 'Static Weight Divider %d' % n, 'static_ratio_%d' % n)
        static_regressor_node.add_output(static_divider_node, 'states')

        relative_divider = Divider('ergonomic_%d' % n, 'configuration_%d' % n)
        relative_divider_node = ForwardBlockNode(relative_divider, 'Static Weight Divider %d' % n, 'relative_ratio_%d' % n)
        relative_regressor_node.add_output(relative_divider_node, 'states')

    vector_to_point_pub(config_predictor_node, 'config_descent', 'world')
    vector_to_point_pub(ergo_predictor_node, 'ergo_descent', 'world')
    vector_to_point_pub(relative_observation_node, 'relative_obs', 'world')
    vector_to_point_pub(static_observation_node, 'static_obs', 'world')

    merger = Merger(['_relative', '_static'] + ['']*8)
    merger_node = ForwardBlockNode(merger, 'Output Merger', 'output')
    relative_regressor_node.add_output(merger_node, 'first')
    static_regressor_node.add_output(merger_node, 'second')
    config_cost_node.add_output(merger_node, 'third')
    ergo_cost_node.add_output(merger_node, 'fourth')
    system_state_node.add_output(merger_node, 'fifth')
    input_mag_node.add_output(merger_node, 'sixth')
    arm_joint_mag_node.add_output(merger_node, 'seventh')
    box_joint_mag_node.add_output(merger_node, 'eighth')
    robot_mag_node.add_output(merger_node, 'ninth')
    state_deriv_node.add_output(merger_node, 'tenth')

    merger_node.add_raw_output(JointPublisher('all_output', bag, get_time=time), 'Output Publisher', None, 'states')

    # Draw the block diagram if requested
    if diagram_filename is not None:
        system.draw(filename=diagram_filename)

    # all the data for every timestep
    all_data = system.run(record=False, print_steps=50)

    return all_data


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