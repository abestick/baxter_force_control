#!/usr/bin/env python
import argparse
import numpy as np
import pickle
import rospy
import kinmodel
import phasespace.load_mocap as load_mocap
import os
from kinmodel.track_mocap import KinematicTreeTracker, KinematicTreeExternalFrameTracker, MocapFrameTracker
from baxter_force_control.steppables import MocapMeasurement, TFBagger
from baxter_force_control.tools import get_system_state_node, get_frame_tracker_node, \
    differentiate_node_for_visualization, grip_point, relative_transform_node
from rosbag import Bag
from baxter_force_control.system import ForwardBlockNode, ForwardSystem, ForwardRoot
from std_msgs.msg import Float32
from os.path import expanduser
import matplotlib.pyplot as plt


GROUP_NAME = 'tree'
HOME = expanduser("~")


def get_time(trial_result):
    return trial_result[4]


def get_start_time(trial_result):
    return trial_result[0][0]


def get_mocap_array(trial_result):
    return trial_result[3]


def visualize_experiment(dry_run=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('participant', help='The name of the participant')
    parser.add_argument('-b', '--bagfile', help='output bagfile to store ROS messages')
    args = parser.parse_args()

    box_json = HOME + '/experiment/box/box_opt.json'
    participant_json = HOME + '/experiment/%s/%s_opt.json' % (args.participant, args.participant)
    results_pickle = HOME + '/experiment/results/%s_results.p' % args.participant
    block_diagram_path = HOME + '/experiment/results/%s_viz.gv' % args.participant

    # Load the calibration sequence
    trials = pickle.load(open(results_pickle, 'rb'))

    # Initialize the tree
    box_kin_tree = kinmodel.KinematicTree(json_filename=box_json)
    human_kin_tree = kinmodel.KinematicTree(json_filename=participant_json)

    # We only want to draw the block diagram once
    block_diag = None
    rospy.init_node('trial_publisher')
    dry_runs = []

    for i, (key, trial) in enumerate(trials.items()):
        # If its the first element, it is the combined data, change i to ALL and set to plot the block diagram
        if i == 0:
            block_diag = block_diagram_path

        if args.bagfile is not None:
            bag = Bag('%s_%s.bag' % (args.bagfile, key), 'w')
            print('Saving to %s' % bag.filename)

        elif dry_run:
            bag_path = HOME + '/experiment/dry_runs/%s_%s_dry_run.bag' % (args.participant, key)
            dry_runs.append(bag_path)

            if os.path.isfile(bag_path):
                continue
            bag = Bag(bag_path, 'w')
            print('Saving to %s' % bag.filename)

        else:
            bag = None

        # Learn the system and draw the block diagram
        mocap_data = get_mocap_array(trial)

        time = get_time(trial)
        dt = np.diff(time)
        dt = dt[dt>0.01]
        mean_dt = dt.mean()
        print('Mean dt: %f\n'
              'Std dt:  %f\n'
              'Min dt:  %f\n'
              'Max dt:  %f' % (mean_dt, dt.std(), dt.min(), dt.max()))

        if not dry_run:
            if raw_input('Press ENTER to begin trial %s [%d] or s to skip' % (key, i)) == 's':
                continue

        else:
            print('Beginning trial %s [%d]' % (key, i))

        trajectories = visualize(mocap_data, box_kin_tree, human_kin_tree, mean_dt, block_diag, bag, dry_run)

        if bag is not None:
            bag.close()

        block_diag = None

    if dry_run:
        for bag in dry_runs:

            if raw_input('Press ENTER to begin trial %s or s to skip' % bag) == 's':
                continue

            os.system('rosbag play %s --clock' % bag)


def visualize(trial, box_kin_tree, human_kin_tree, dt, diagram_filename=None, bag=None, dry_run=False):

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

    #############################
    # Build up the system nodes #
    #############################
    measurement_node = ForwardBlockNode(mocap_measurement, 'Mocap Measurement', 'raw_mocap')

    # Define the root (all source nodes)
    root = ForwardRoot([measurement_node]) #, zero_twist_node])

    # Create the system
    system = ForwardSystem(root, 1.0/dt)
    time = system.get_time if dry_run else None

    input_flap_node = get_frame_tracker_node(box_frame_tracker, 'input_flap', 'joint_3', get_time=time)
    grip_node = get_frame_tracker_node(box_frame_tracker, 'grip', 'joint_3', get_time=time, position=grip_location)
    robot_flap_node = get_frame_tracker_node(box_frame_tracker, 'robot_flap', 'joint_0', get_time=time)
    chest_node = get_frame_tracker_node(human_frame_tracker, 'chest', 'base', get_time=time)
    forearm_node = get_frame_tracker_node(human_frame_tracker, 'forearm', 'elbow', get_time=time)
    input_grip_node = relative_transform_node(grip_node, forearm_node, get_time=time)

    system_state_node = get_system_state_node([box_kin_tree, human_kin_tree], 'system_state', True, bag, time)

    measurement_node.add_output(input_flap_node, 'measurement')
    measurement_node.add_output(robot_flap_node, 'measurement')
    measurement_node.add_output(chest_node, 'measurement')
    measurement_node.add_output(forearm_node, 'measurement')
    measurement_node.add_output(grip_node, 'measurement')
    measurement_node.add_output(system_state_node, 'measurement')

    input_twist_node, _ = differentiate_node_for_visualization(input_flap_node, dt, bag, time)
    robot_twist_node, modifer_node = differentiate_node_for_visualization(robot_flap_node, dt, bag, time)

    if bag is not None:
        tf_bagger_node = ForwardBlockNode(TFBagger(bag, time, dummy_arg=True), 'TF Bagger', None)
        measurement_node.add_output(tf_bagger_node, 'dummy')

    # Draw the block diagram if requested
    if diagram_filename is not None:
        system.draw(filename=diagram_filename)

    # all the data for every timestep
    if dry_run:
        all_data = system.run(record=False, print_steps=50)
    else:
        all_data = system.run_timed(1.0/dt, record=False, print_steps=5)

    return all_data


def get_dict_to_float_msg(key):
    def constructor(dictionary):
        return Float32(data=dictionary[key])

    return constructor


if __name__ == '__main__':
    visualize_experiment(dry_run=False)
