#!/usr/bin/env python
import argparse
import numpy as np
import pickle
import rospy
import kinmodel
import phasespace.load_mocap as load_mocap
from kinmodel import Twist
from kinmodel.track_mocap import KinematicTreeTracker, KinematicTreeExternalFrameTracker, MocapFrameTracker
from baxter_force_control.steppables import MocapFrameEstimator, MocapSystemState, MocapMeasurement, Differentiator, \
    TwistPublisher, TFPublisher, PointCloudPublisher, JointPublisher, CostCalculator, EdgePublisher, \
    WeightedKinematicCostDescentEstimator, TFBagger, Constant, WrenchPublisher, Modifier
from rosbag import Bag
from baxter_force_control.system import ForwardBlockNode, ForwardSystem, ForwardRoot
from baxter_force_control.motion_costs import QuadraticDisplacementCost, WeightedCostCombination
from baxter_force_control.control_law import WeightedKinematicCostDescent
from std_msgs.msg import Float32
from os.path import expanduser


FRAMERATE = 50
GROUP_NAME = 'tree'
HOME = expanduser("~")


def get_time(trial_result):
    return trial_result[0][0]


def get_mocap_array(trial_result):
    return trial_result[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('participant', help='The name of the participant')
    parser.add_argument('-b', '--bagfile', help='output bagfile to store ROS messages')
    args = parser.parse_args()

    box_json = HOME + '/experiment/box/box_opt.json'
    participant_json = HOME + '/experiment/%s/%s_opt.json' % (args.participant, args.participant)
    results_pickle = HOME + '/experiment/results/%s_results.p' % args.participant
    block_diagram_path = HOME + '/experiment/results/%s.gv' % args.participant


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
            block_diag = block_diagram_path

        if args.bagfile is not None:
            bag = Bag('%s_%s.bag' % (args.bagfile, str(i)), 'w')
            print('Saving to %s' % bag.filename)

        else:
            bag = None

        if raw_input('Press ENTER to begin trial %s [%d] or s to skip' % (key, i)) == 's':
            continue

        # Learn the system and draw the block diagram
        mocap_data = get_mocap_array(trial)
        trajectories = learn(mocap_data, box_kin_tree, human_kin_tree, block_diag, bag)
        if bag is not None:
            bag.close()
        block_diag = None


def learn(trial, box_kin_tree, human_kin_tree, diagram_filename=None, bag=None):
    # Put into a MocapArray
    mocap_array = load_mocap.ArrayMocapSource(trial, FRAMERATE)

    print('Number of data points: %d' % len(mocap_array))

    # Initialize the frame tracker and attach frames of interest
    frame_tracker = KinematicTreeExternalFrameTracker(box_kin_tree.copy())
    input_indices, input_points = frame_tracker.attach_frame('joint_3', 'input_flap')
    disturbance_indices, disturbance_points = frame_tracker.attach_frame('joint_0', 'disturbance_flap')

    # Define the measurement block
    mocap_measurement = MocapMeasurement(mocap_array, 'mocap_measurement')

    input_frame_tracker = MocapFrameTracker('input_tracker', input_indices, input_points)
    input_frame_estimator = MocapFrameEstimator(input_frame_tracker, 'input_flap')
    input_estimator = Differentiator(fixed_step=1.0/FRAMERATE)
    input_pos_modifier = Modifier(pos_only_transform)

    disturbance_frame_tracker = MocapFrameTracker('disturbance_tracker', disturbance_indices, disturbance_points)
    disturbance_frame_estimator = MocapFrameEstimator(disturbance_frame_tracker, 'disturbance_flap')
    disturbance_estimator = Differentiator(fixed_step=1.0/FRAMERATE)
    disturbance_pos_modifier = Modifier(pos_only_transform)

    # Define the system state tracker used to estimate the system state
    box_tree_tracker = KinematicTreeTracker('tree_tracker', box_kin_tree)
    box_system_state = MocapSystemState([box_tree_tracker])
    human_tree_tracker = KinematicTreeTracker('tree_tracker', human_kin_tree)
    human_system_state = MocapSystemState([human_tree_tracker])

    ##############################
    ## Build up the system nodes##
    ##############################
    measurement_node = ForwardBlockNode(mocap_measurement, 'Mocap Measurement', 'raw_mocap')
    # zero_twist_node = ForwardBlockNode(Constant({'zero_twist': Twist(xi=np.zeros(6))}), 'Zero Twist Constant',
    #                                    'zero_twist')
    # Define the root (all source nodes)
    root = ForwardRoot([measurement_node]) #, zero_twist_node])

    # Create the system
    system = ForwardSystem(root)

    measurement_node.add_raw_output(PointCloudPublisher('mocap_point_cloud', 'world', bag, system.get_time),
                                    'PointCloud Publisher', None, 'states')

    input_frame_node = ForwardBlockNode(input_frame_estimator, 'Input Frame Estimator', 'input_flap')
    input_modifier_node = ForwardBlockNode(input_pos_modifier, 'Input Pos Modifier', 'input_pos')
    input_frame_node.add_output(input_modifier_node, 'states')
    input_frame_node.add_raw_output(TFPublisher('world', 'input_flap'), 'Input Frame Publisher', None, 'transform')
    input_modifier_node.add_raw_output(TFPublisher('world', 'input_flap_pos'), 'Input Pos Publisher', None, 'transform')
    measurement_node.add_output(input_frame_node, 'measurement')

    disturbance_frame_node = ForwardBlockNode(disturbance_frame_estimator, 'Disturbance Frame Estimator', 'disturbance_flap')
    disturbance_modifier_node = ForwardBlockNode(disturbance_pos_modifier, 'Disturbance Pos Modifier', 'disturbance_pos')
    disturbance_frame_node.add_output(disturbance_modifier_node, 'states')
    disturbance_frame_node.add_raw_output(TFPublisher('world', 'disturbance_flap'), 'Disturbance Frame Publisher', None, 'transform')
    disturbance_modifier_node.add_raw_output(TFPublisher('world', 'disturbance_flap_pos'), 'Disturbance Pos Publisher', None, 'transform')
    measurement_node.add_output(disturbance_frame_node, 'measurement')

    box_system_state_node = ForwardBlockNode(box_system_state, 'Box State Estimator', 'object_joints')
    box_system_state_node.add_raw_output(JointPublisher('box_joint_states', bag, system.get_time),
                                     'Object Joint Publisher', None, 'states')
    measurement_node.add_output(box_system_state_node, 'measurement')

    human_system_state_node = ForwardBlockNode(human_system_state, 'Human State Estimator', 'human_joints')
    human_system_state_node.add_raw_output(JointPublisher('human_joint_states', bag, system.get_time),
                                           'Box Joint Publisher', None, 'states')
    measurement_node.add_output(human_system_state_node, 'measurement')

    input_differentiator_node = ForwardBlockNode(input_estimator, 'Input Differentiator', 'd_disturbance_input_flap')
    # input_differentiator_node.add_raw_output(TwistPublisher('input_twist', 'world', bag, system.get_time),
    #                                          'Input Twist Publisher', None, 'states')
    input_differentiator_node.add_raw_output(WrenchPublisher('input_twist', 'input_flap_pos', bag, system.get_time),
                                             'Input Wrench Publisher', None, 'states')
    input_frame_node.add_output(input_differentiator_node, 'states')

    disturbance_differentiator_node = ForwardBlockNode(disturbance_estimator, 'Disturbance Differentiator',
                                                       'd_world_disturbance_flap')
    # disturbance_differentiator_node.add_raw_output(TwistPublisher('disturbance_twist', 'world', bag, system.get_time),
    #                                                'Disturbance Twist Publisher', None, 'states')
    disturbance_differentiator_node.add_raw_output(WrenchPublisher('disturbance_twist', 'disturbance_flap_pos', bag, system.get_time),
                                                   'Disturbance Wrench Publisher', None, 'states')
    disturbance_frame_node.add_output(disturbance_differentiator_node, 'states')

    tf_bagger_node = ForwardBlockNode(TFBagger(bag, system.get_time, dummy_arg=True), 'TF Bagger', None)
    # .add_output(tf_bagger_node, 'dummy')

    # Draw the block diagram if requested
    if diagram_filename is not None:
        system.draw(filename=diagram_filename)

    # all the data for every timestep
    all_data = system.run_timed(FRAMERATE, record=False, print_steps=50)

    return all_data


def get_dict_to_float_msg(key):
    def constructor(dictionary):
        return Float32(data=dictionary[key])

    return constructor


def pos_only_transform(transform):
    return transform.trans(as_transform=True)

if __name__ == '__main__':
    main()