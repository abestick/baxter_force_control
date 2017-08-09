#!/usr/bin/env python
import argparse
import numpy as np
import rospy
import kinmodel
import phasespace.load_mocap as load_mocap
from kinmodel import Twist
from kinmodel.track_mocap import KinematicTreeTracker, KinematicTreeExternalFrameTracker, MocapFrameTracker
from baxter_force_control.steppables import MocapFrameEstimator, MocapSystemState, MocapMeasurement, Differentiator, \
    TwistPublisher, TFPublisher, PointCloudPublisher, JointPublisher, CostCalculator, EdgePublisher, \
    WeightedKinematicCostDescentEstimator, TFBagger, Constant
from rosbag import Bag
from baxter_force_control.system import ForwardBlockNode, ForwardSystem, ForwardRoot
from baxter_force_control.motion_costs import QuadraticDisplacementCost, WeightedCostCombination
from baxter_force_control.control_law import WeightedKinematicCostDescent
from std_msgs.msg import Float32

FRAMERATE = 50
GROUP_NAME = 'tree'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kinmodel_json_optimized', help='The kinematic model JSON file')
    parser.add_argument('task_npz')
    parser.add_argument('block_diagram', help='The file name of the drawn block diagram of the system')
    parser.add_argument('-b', '--bagfile', help='output bagfile to store ROS messages')
    args = parser.parse_args()

    # Load the calibration sequence
    data = np.load(args.task_npz)
    trials = 0

    # Stack all the trials
    while 'full_sequence_' + str(trials) in data.keys():
        trials += 1
    print('Number of trials: %d' % trials)
    mocap_data = [data['full_sequence_' + str(trial)] for trial in range(trials)]
    mocap_data.insert(0, np.concatenate(mocap_data, axis=2))

    # Initialize the tree
    kin_tree = kinmodel.KinematicTree(json_filename=args.kinmodel_json_optimized)
    # kin_tree = kinmodel.KinematicTree(json_filename='/home/pedge/shoulder_data/shoulder_elbow_opt.json')
    config_reference_1 = KinematicTreeTracker('tree_tracker', kin_tree).process_frame(data['goal'][:, :, None])

    # Hard coded config reference from looking at the raw data
    config_reference_2 = {'joint2': 1.57,
                          'joint3': 1.57,
                          'joint4': -1.57,
                          'joint5': -1.57}

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

    input_flap_joint_num = 4
    input_flap_joint = 'joint' + str(input_flap_joint_num)
    disturbance_flap_joint = 'joint' + str(4 + int(input_flap_joint_num == 4))

    # Initialize the frame tracker and attach frames of interest
    frame_tracker = KinematicTreeExternalFrameTracker(kin_tree.copy())
    base_indices, base_points = frame_tracker.attach_frame('joint1', 'base')
    input_indices, input_points = frame_tracker.attach_frame(input_flap_joint, 'input_flap')
    disturbance_indices, disturbance_points = frame_tracker.attach_frame(disturbance_flap_joint, 'disturbance_flap')

    # Define the measurement block
    mocap_measurement = MocapMeasurement(mocap_array, 'mocap_measurement')

    input_frame_tracker = MocapFrameTracker('input_tracker', input_indices, input_points)
    input_frame_estimator = MocapFrameEstimator(input_frame_tracker, 'input_flap')
    input_estimator = Differentiator(fixed_step=1.0/FRAMERATE)

    disturbance_frame_tracker = MocapFrameTracker('disturbance_tracker', disturbance_indices)#, disturbance_points)
    disturbance_frame_estimator = MocapFrameEstimator(disturbance_frame_tracker, 'disturbance_flap')
    disturbance_estimator = Differentiator(fixed_step=1.0/FRAMERATE)

    base_frame_tracker = MocapFrameTracker('base_tracker', base_indices)#, base_points)
    base_frame_estimator = MocapFrameEstimator(base_frame_tracker, 'base_flap')

    # Define the system state tracker used to estimate the system state
    kin_tree_tracker = KinematicTreeTracker('tree_tracker', kin_tree)
    system_state = MocapSystemState([kin_tree_tracker])

    weighted_config_cost = WeightedCostCombination('config_cost', costs)
    weighted_descent = WeightedKinematicCostDescent(weighted_config_cost, frame_tracker, 'input_flap',
                                                    'disturbance_flap')
    weighted_descent_estimator = WeightedKinematicCostDescentEstimator(weighted_descent, 'map', 200, False)


    ##############################
    ## Build up the system nodes##
    ##############################
    measurement_node = ForwardBlockNode(mocap_measurement, 'Mocap Measurement', 'raw_mocap')
    zero_twist_node = ForwardBlockNode(Constant({'zero_twist': Twist(xi=np.zeros(6))}), 'Zero Twist Constant',
                                       'zero_twist')
    # Define the root (all source nodes)
    root = ForwardRoot([measurement_node, zero_twist_node])

    # Create the system
    system = ForwardSystem(root)

    measurement_node.add_raw_output(PointCloudPublisher('points', 'map', bag, system.get_time),
                                    'PointCloud Publisher', None, 'states')

    input_frame_node = ForwardBlockNode(input_frame_estimator, 'Input Frame Estimator', 'input_flap')
    input_frame_node.add_raw_output(TFPublisher('map', 'input_flap'), 'Input Frame Publisher', None, 'transform')
    measurement_node.add_output(input_frame_node, 'measurement')

    disturbance_frame_node = ForwardBlockNode(disturbance_frame_estimator, 'Disturbance Frame Estimator', 'disturbance_flap')
    disturbance_frame_node.add_raw_output(TFPublisher('map', 'disturbance_flap'), 'Disturbance Frame Publisher', None, 'transform')
    measurement_node.add_output(disturbance_frame_node, 'measurement')

    base_frame_node = ForwardBlockNode(base_frame_estimator, 'Base Frame Estimator', 'base_flap')
    base_frame_node.add_raw_output(TFPublisher('map', 'base_flap'), 'Base Frame Publisher', None, 'transform')
    measurement_node.add_output(base_frame_node, 'measurement')

    system_state_node = ForwardBlockNode(system_state, 'State Estimator', 'object_joints')
    system_state_node.add_raw_output(JointPublisher('joint_states', bag, system.get_time),
                                     'Object Joint Publisher', None, 'states')
    measurement_node.add_output(system_state_node, 'measurement')

    cost_1_node = ForwardBlockNode(CostCalculator(costs[0]), "Cost 1 Calculator", 'cost_1')
    cost_2_node = ForwardBlockNode(CostCalculator(costs[1]), "Cost 2 Calculator", 'cost_2')
    system_state_node.add_output(cost_1_node, 'states')
    system_state_node.add_output(cost_2_node, 'states')
    cost_1_node.add_raw_output(EdgePublisher('cost_1', Float32, get_dict_to_float_msg(costs[0].name), bag,
                                             system.get_time), 'Cost 1 Publisher', None, 'states')
    cost_2_node.add_raw_output(EdgePublisher('cost_2', Float32, get_dict_to_float_msg(costs[1].name), bag,
                                             system.get_time), 'Cost 2 Publisher', None, 'states')

    input_differentiator_node = ForwardBlockNode(input_estimator, 'Input Differentiator', 'd_disturbance_input_flap')
    input_differentiator_node.add_raw_output(TwistPublisher('input_twist', 'map', bag, system.get_time),
                                             'Input Twist Publisher', None, 'states')
    input_frame_node.add_output(input_differentiator_node, 'states')

    disturbance_differentiator_node = ForwardBlockNode(disturbance_estimator, 'Disturbance Differentiator', 'd_map_disturbance_flap')
    disturbance_differentiator_node.add_raw_output(TwistPublisher('disturbance_twist', 'map', bag, system.get_time),
                                                   'Disturbance Twist Publisher', None, 'states')
    disturbance_frame_node.add_output(disturbance_differentiator_node, 'states')

    estimator_node = ForwardBlockNode(weighted_descent_estimator, 'Weight Estimator', 'weights')
    weight_publisher_node = ForwardBlockNode(JointPublisher('weights', bag, system.get_time), 'Weight Publisher', None)
    estimator_node.add_output(weight_publisher_node, 'states')
    input_differentiator_node.add_output(estimator_node, 'input_twist')
    system_state_node.add_output(estimator_node, 'states')
    disturbance_differentiator_node.add_output(estimator_node, 'base_twist')
    # zero_twist_node.add_output(estimator_node, 'base_twist')
    disturbance_frame_node.add_output(estimator_node, 'base_transform')

    tf_bagger_node = ForwardBlockNode(TFBagger(bag, system.get_time, dummy_arg=True), 'TF Bagger', None)
    weight_publisher_node.add_output(tf_bagger_node, 'dummy')

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

if __name__ == '__main__':
    main()
