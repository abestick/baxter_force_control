#!/usr/bin/env python
import argparse
import pickle
import rospy
import kinmodel
import os
import numpy as np
import phasespace.load_mocap as load_mocap
from kinmodel.track_mocap import KinematicTreeTracker, KinematicTreeExternalFrameTracker, MocapFrameTracker
from baxter_force_control.steppables import MocapMeasurement, TFBagger, Vector3Publisher, Modifier, \
    Subtractor, Iterator, JointCommandPublisher, ExponentialFilter
from rosbag import Bag
from baxter_force_control.system import ForwardBlockNode, ForwardSystem, ForwardRoot
from baxter_force_control.tools import get_frame_tracker_node, get_system_state_node, differentiate_node, \
    relative_transform_node, grip_point
from std_msgs.msg import Float32
from os.path import expanduser
import warnings


HOME = expanduser("~")


def get_time(trial_result):
    return trial_result[4]


def get_start_time(trial_result):
    return trial_result[0][0]


def get_mocap_array(trial_result):
    return trial_result[3]


def get_baxter_joints(trial_result):
    return trial_result[0][1]


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def extract_kinematics():
    parser = argparse.ArgumentParser()
    parser.add_argument('participant', help='The name of the participant')
    args = parser.parse_args()

    box_json = HOME + '/experiment/box/new_box_opt.json'
    participant_json = HOME + '/experiment/%s/new_%s_opt.json' % (args.participant, args.participant)
    results_pickle = HOME + '/experiment/results/%s_results.p' % args.participant
    participant_results_dir = HOME + '/experiment/results/%s' % args.participant
    block_diagram_path = participant_results_dir + '/%s_extract.gv' % args.participant
    bag_path = participant_results_dir + '/profile/' #kinematics_extracted2

    create_dir(participant_results_dir)
    create_dir(bag_path)

    # Load the calibration sequence
    trials = pickle.load(open(results_pickle, 'rb'))

    # Initialize the tree
    box_kin_tree = kinmodel.KinematicTree(json_filename=box_json)
    human_kin_tree = kinmodel.KinematicTree(json_filename=participant_json)

    # We only want to draw the block diagram once
    block_diag = None
    rospy.init_node('trial_publisher')

    for i, (key, trial) in enumerate(trials.items()):
        # If its the first element, it is the combined data, change i to ALL and set to plot the block diagram
        if i == 0:
            block_diag = None #block_diagram_path

        bag = Bag(bag_path + '%s_%s.bag' % (args.participant, key), 'w')
        print('Saving to %s' % bag.filename)

        # Learn the system and draw the block diagram
        mocap_data = get_mocap_array(trial)
        robot_data = get_baxter_joints(trial)
        time = get_time(trial)
        dt = np.diff(time)
        dt = dt[dt > 0.01]
        mean_dt = dt.mean()
        print('Mean dt: %f\n'
              'Std dt:  %f\n'
              'Min dt:  %f\n'
              'Max dt:  %f' % (mean_dt, dt.std(), dt.min(), dt.max()))

        trajectories = extract(mocap_data, robot_data, box_kin_tree, human_kin_tree, mean_dt, block_diag, bag)
        if bag is not None:
            bag.close()
        block_diag = None


def extract(trial, robot_data, box_kin_tree, human_kin_tree, dt, diagram_filename=None, bag=None):

    # Put into a MocapArray
    mocap_array = load_mocap.ArrayMocapSource(trial, 1.0/dt)

    print('Number of data points: %d' % len(mocap_array))

    input_features = box_kin_tree.get_joints()['joint_3'].children
    input_points = [np.array(p.primitive) for p in input_features]
    grip_location = grip_point(input_points)

    # Initialize the frame tracker and attach frames of interest
    box_frame_tracker = KinematicTreeExternalFrameTracker(box_kin_tree.copy())
    human_frame_tracker = KinematicTreeExternalFrameTracker(human_kin_tree.copy())

    # Define the measurement block
    mocap_measurement = MocapMeasurement(mocap_array, 'mocap_measurement')
    robot_joint_iterator = Iterator(robot_data, 'baxter_joint_commands')

    #############################
    # Build up the system nodes #
    #############################
    measurement_node = ForwardBlockNode(mocap_measurement, 'Mocap Measurement', 'raw_mocap')
    command_node = ForwardBlockNode(robot_joint_iterator, 'Baxter Commands', 'baxter_joint_commands')

    # Define the root (all source nodes)
    root = ForwardRoot([measurement_node]) #, command_node]) #, zero_twist_node])

    # Create the system
    system = ForwardSystem(root, 1.0/dt)
    time = system.get_time

    grip_node = get_frame_tracker_node(box_frame_tracker, 'grip', 'joint_3', get_time=time, position=grip_location)
    robot_node = get_frame_tracker_node(box_frame_tracker, 'robot', 'joint_0', get_time=time)
    chest_node = get_frame_tracker_node(human_frame_tracker, 'chest', 'base', get_time=time)
    forearm_node = get_frame_tracker_node(human_frame_tracker, 'forearm', 'elbow', get_time=time)
    grip_forearm_node = relative_transform_node(grip_node, forearm_node, get_time=time)
    chest_grip_node = relative_transform_node(chest_node, grip_node, get_time=time)
    robot_chest_node = relative_transform_node(robot_node, chest_node, get_time=time)
    robot_grip_node = relative_transform_node(robot_node, grip_node, get_time=time)
    # chest_forearm = relative_transform_node(chest_node, forearm_node, get_time=time)

    robot_chest_vec_node = ForwardBlockNode(Modifier(get_vector), 'Robot->Chest Vector Extractor', 'p_robot_chest')
    robot_grip_vec_node = ForwardBlockNode(Modifier(get_vector), 'Robot->Grip Vector Extractor', 'p_robot_grip')
    robot_grip_chest_vec_node = ForwardBlockNode(Subtractor(), 'Grip->Chest Subtractor', 'p_grip_chest')

    robot_chest_node.add_output(robot_chest_vec_node, 'states')
    robot_grip_node.add_output(robot_grip_vec_node, 'states')
    robot_chest_vec_node.add_output(robot_grip_chest_vec_node, 'first')
    robot_grip_vec_node.add_output(robot_grip_chest_vec_node, 'second')
    robot_grip_chest_vec_node.add_raw_output(Vector3Publisher('robot_grip_chest_vec', bag, time),
                                             'Grip->Chest Publisher', None, 'states')

    system_state_node, filtered_state_node = get_system_state_node([box_kin_tree, human_kin_tree], 'system_state',
                                                                   True, bag, time)


    command_node.add_raw_output(JointCommandPublisher('baxter_joint_commands', bag, get_time=time), 'Command Publisher',
                                None, 'states')

    measurement_node.add_output(grip_node, 'measurement')
    measurement_node.add_output(robot_node, 'measurement')
    measurement_node.add_output(chest_node, 'measurement')
    measurement_node.add_output(forearm_node, 'measurement')
    measurement_node.add_output(system_state_node, 'measurement')

    grip_twist_node = differentiate_node(grip_node, dt, True, bag, time)
    robot_twist_node = differentiate_node(robot_node, dt, True, bag, time)
    forearm_twist_node = differentiate_node(forearm_node, dt, True, bag, time)
    grip_forearm_twist_node = differentiate_node(grip_forearm_node, dt, True, bag, time)
    chest_grip_twist_node = differentiate_node(chest_grip_node, dt, True, bag, time)
    robot_chest_twist_node = differentiate_node(robot_chest_node, dt, True, bag, time, filter=True)
    robot_grip_twist_node = differentiate_node(robot_grip_node, dt, True, bag, time)

    tf_bagger_node = ForwardBlockNode(TFBagger(bag, time, dummy_arg=True), 'TF Bagger', None)
    measurement_node.add_output(tf_bagger_node, 'dummy')

    # Draw the block diagram if requested
    if diagram_filename is not None:
        system.draw(filename=diagram_filename)

    # all the data for every timestep
    all_data = system.run(record=False, print_steps=50)

    return all_data


def twist_properties(twist_dict):
    (_, twist),  = twist_dict.items()
    print(np.linalg.norm(twist.nu()))
    angle = np.linalg.norm(twist.omega())
    print(twist.omega() / angle)
    print(angle)


def get_dict_to_float_msg(key):
    def constructor(dictionary):
        return Float32(data=dictionary[key])

    return constructor


def get_vector(transform):
    return transform.trans()


if __name__ == '__main__':
    warnings.filterwarnings("error")
    extract_kinematics()