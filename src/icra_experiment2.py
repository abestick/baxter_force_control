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
    parser.add_argument('results_pickle', help='The name of the pickle object containing the results')
    parser.add_argument('block_diagram', help='The file name of the drawn block diagram of the system')
    parser.add_argument('-b', '--bagfile', help='output bagfile to store ROS messages')
    args = parser.parse_args()

    box_json = HOME + '/experiment/box/box_opt.json'
    participant_json = HOME + '/experiment/%s/%s_opt.json' % (args.participant, args.participant)
    results_pickle = HOME + '/experiment/results/%s' % args.results_pickle

    # Load the calibration sequence
    trials = pickle.load(open(results_pickle, 'rb'))

    # Initialize the tree
    box_kin_tree = kinmodel.KinematicTree(json_filename=box_json)
    human_kin_tree = kinmodel.KinematicTree(json_filename=participant_json)



    config_reference_1 = {'joint_0': 0.0,
                          'joint_1': 0.0,
                          'joint_2': 0.0,
                          'joint_3': 0.0}


    # Hard coded config reference from looking at the raw data
    config_reference_2 = {'joint_0': 1.57,
                          'joint_1': 1.57,
                          'joint_2': -1.57,
                          'joint_3': -1.57}

    # Create a config cost and bases of manipulability costs
    config_cost_1 = QuadraticDisplacementCost('config_1', config_reference_1, lambda x: x, config_reference_1.keys())
    config_cost_2 = QuadraticDisplacementCost('config_2', config_reference_2, lambda x: x, config_reference_2.keys())

    # Put these into a weighted cost and put the weighted cost into the system cost dictionary
    costs = [config_cost_1, config_cost_2]

    # We only want to draw the block diagram once
    block_diag = None
    rospy.init_node('trial_publisher')

    for i, trial in enumerate(mocap_data):
        # If its the first element, it is the combined data, change i to ALL and set to plot the block diagram
        if i == 0:
            i = 'ALL'
            block_diag = args.block_diagram

        if args.bagfile is not None:
            bag = Bag('%s_%s.bag' % (args.bagfile, str(i)), 'w')
            print('Saving to %s' % bag.filename)

        else:
            bag = None

        print('Press ENTER to learn trial ' + str(i) + '...')

        # Learn the system and draw the block diagram
        trajectories = learn(trial, kin_tree, costs, block_diag, bag)
        bag.close()
        block_diag = None


def learn(trial, kin_tree, costs, diagram_filename=None, bag=None):
    # Put into a MocapArray
    mocap_array = load_mocap.ArrayMocapSource(trial, FRAMERATE)

    print('Number of data points: %d' % len(mocap_array))

    # Initialize the frame tracker and attach frames of interest
    frame_tracker = KinematicTreeExternalFrameTracker(kin_tree.copy())
    input_indices, input_points = frame_tracker.attach_frame('joint_3', 'input_flap')
    disturbance_indices, disturbance_points = frame_tracker.attach_frame('joint_0', 'disturbance_flap')

    # Define the measurement block
    mocap_measurement = MocapMeasurement(mocap_array, 'mocap_measurement')

    input_frame_tracker = MocapFrameTracker('input_tracker', input_indices, input_points)
    input_frame_estimator = MocapFrameEstimator(input_frame_tracker, 'input_flap')
    input_estimator = Differentiator(fixed_step=1.0/FRAMERATE)
    input_pos_modifier = Modifier(pos_only_transform)

    disturbance_frame_tracker = MocapFrameTracker('disturbance_tracker', disturbance_indices)#, disturbance_points)
    disturbance_frame_estimator = MocapFrameEstimator(disturbance_frame_tracker, 'disturbance_flap')
    disturbance_estimator = Differentiator(fixed_step=1.0/FRAMERATE)
    disturbance_pos_modifier = Modifier(pos_only_transform)

    # Define the system state tracker used to estimate the system state
    # kin_tree_tracker = KinematicTreeTracker('tree_tracker', kin_tree)
    # system_state = MocapSystemState([kin_tree_tracker])
    #
    # weighted_config_cost = WeightedCostCombination('config_cost', costs)
    # weighted_descent = WeightedKinematicCostDescent(weighted_config_cost, frame_tracker, 'input_flap',
    #                                                 'disturbance_flap')
    # weighted_descent_estimator = WeightedKinematicCostDescentEstimator(weighted_descent, 'world', 200, False)


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
    input_frame_node.add_raw_output(input_pos_modifier, 'Input Pos Modifier', 'input_pos', 'states')
    input_frame_node.add_raw_output(TFPublisher('world', 'input_flap'), 'Input Frame Publisher', None, 'transform')
    input_frame_node.add_raw_output(TFPublisher('world', 'input_flap_pos'), 'Input Pos Publisher', None, 'transform')
    measurement_node.add_output(input_frame_node, 'measurement')

    disturbance_frame_node = ForwardBlockNode(disturbance_frame_estimator, 'Disturbance Frame Estimator', 'disturbance_flap')
    disturbance_frame_node.add_raw_output(input_pos_modifier, 'Disturbance Pos Modifier', 'disturbance_pos', 'states')
    disturbance_frame_node.add_raw_output(TFPublisher('world', 'disturbance_flap'), 'Disturbance Frame Publisher', None, 'transform')
    disturbance_frame_node.add_raw_output(TFPublisher('world', 'disturbance_flap_pos'), 'Disturbance Pos Publisher', None, 'transform')
    measurement_node.add_output(disturbance_frame_node, 'measurement')

    # system_state_node = ForwardBlockNode(system_state, 'State Estimator', 'object_joints')
    # system_state_node.add_raw_output(JointPublisher('joint_states', bag, system.get_time),
    #                                  'Object Joint Publisher', None, 'states')
    # measurement_node.add_output(system_state_node, 'measurement')

    # cost_1_node = ForwardBlockNode(CostCalculator(costs[0]), "Cost 1 Calculator", 'cost_1')
    # cost_2_node = ForwardBlockNode(CostCalculator(costs[1]), "Cost 2 Calculator", 'cost_2')
    # system_state_node.add_output(cost_1_node, 'states')
    # system_state_node.add_output(cost_2_node, 'states')
    # cost_1_node.add_raw_output(EdgePublisher('cost_1', Float32, get_dict_to_float_msg(costs[0].name), bag,
    #                                          system.get_time), 'Cost 1 Publisher', None, 'states')
    # cost_2_node.add_raw_output(EdgePublisher('cost_2', Float32, get_dict_to_float_msg(costs[1].name), bag,
    #                                          system.get_time), 'Cost 2 Publisher', None, 'states')

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

    # estimator_node = ForwardBlockNode(weighted_descent_estimator, 'Weight Estimator', 'weights')
    # weight_publisher_node = ForwardBlockNode(JointPublisher('weights', bag, system.get_time), 'Weight Publisher', None)
    # estimator_node.add_output(weight_publisher_node, 'states')
    # input_differentiator_node.add_output(estimator_node, 'input_twist')
    # system_state_node.add_output(estimator_node, 'states')
    # disturbance_differentiator_node.add_output(estimator_node, 'base_twist')
    # zero_twist_node.add_output(estimator_node, 'base_twist')
    # disturbance_frame_node.add_output(estimator_node, 'base_transform')

    tf_bagger_node = ForwardBlockNode(TFBagger(bag, system.get_time, dummy_arg=True), 'TF Bagger', None)
    # weight_publisher_node.add_output(tf_bagger_node, 'dummy')

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