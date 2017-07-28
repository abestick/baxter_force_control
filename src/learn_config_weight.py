#!/usr/bin/env python
import argparse
import numpy as np
import pickle

import kinmodel
import phasespace.load_mocap as load_mocap
from kinmodel.track_mocap import KinematicTreeTracker, KinematicTreeExternalFrameTracker, MocapFrameTracker
from baxter_force_control.motion_costs import WeightedCostCombination, BasisManipulabilityCost, QuadraticDisplacementCost
from baxter_force_control.steppables import MocapFrameEstimator, WeightedKinematicCostDescentEstimator, MocapSystemState, MocapMeasurement, \
    Differentiator, Transformer, DynamicController, Selector, Averager
from baxter_force_control.system import ForwardBlockNode, ForwardSystem, ForwardRoot

from baxter_force_control.control_law import WeightedKinematicCostDescent, KinematicCostDescent

FRAMERATE = 50
GROUP_NAME = 'tree'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kinmodel_json_optimized', help='The kinematic model JSON file')
    parser.add_argument('task_npz')
    parser.add_argument('block_diagram', help='The file name of the drawn block diagram of the system')
    parser.add_argument('system_history', help='The system output pkl file')
    args = parser.parse_args()

    # Perform weight estimation over different window sizes
    window_sizes = (10, 50, 100)

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
    config_reference = KinematicTreeTracker('tree_tracker', kin_tree).process_frame(data['goal'][:, :, None])

    # We only want to draw the block diagram once
    block_diag = None
    histories = {}

    for i, trial in enumerate(mocap_data):
        # If its the first element, it is the combined data, change i to ALL and set to plot the block diagram
        if i == 0:
            i = 'ALL'
            block_diag = args.block_diagram

        print('Learning trial ' + str(i) + '...')

        # Learn the system and draw the block diagram
        trajectories = learn(trial, kin_tree, window_sizes, config_reference, block_diag)

        # Set back to None so we don't re draw all the other times
        block_diag = None

        # append the trial number to the key to avoid clashing
        histories[str(i)] = trajectories

    # save to an npz file
    filename = args.system_history
    save(filename, histories)


def save(filename, data):
    # store the trajectory of every element in the system

    output = open(filename, 'wb')

    # Pickle dictionary using protocol 0.
    pickle.dump(data, output)

    # with open(filename, 'w') as output_file:
    #     np.savez_compressed(output_file, **data)
    print('System history saved to ' + filename)


def learn(trial, kin_tree, window_sizes, config_reference, diagram_filename=None):
    # Put into a MocapArray
    mocap_array = load_mocap.ArrayMocapSource(trial, FRAMERATE)

    # Append window_sizes to include a total estimation for all data
    window_sizes = window_sizes + (len(mocap_array),)

    print('Number of data points: %d' % len(mocap_array))

    # Initialize the frame tracker and attach frames of interest
    frame_tracker = KinematicTreeExternalFrameTracker(kin_tree.copy())
    frame_tracker.attach_frame('joint4', 'flap1')
    frame_tracker.attach_frame('joint5', 'flap2')

    # Create a function that takes a joint angles dict and returns the jacobian for flap1
    def get_flap1_jacobian(object_joint_angles):
        return frame_tracker.compute_jacobian('flap2', 'flap1', object_joint_angles)

    # Create a config cost and bases of manipulability costs
    config_cost = QuadraticDisplacementCost('config', config_reference, lambda x: x, config_reference.keys())

    # Put these into a weighted cost and put the weighted cost into the system cost dictionary
    weighted_config_cost = WeightedCostCombination('config_cost', [config_cost])
    types = ['Config', 'NNLS_Config']*len(window_sizes)

    # Define the control laws we will estimate
    weighted_descent = WeightedKinematicCostDescent(weighted_config_cost, frame_tracker, 'flap1')

    # And their respective estimators (the first len(window_size) elements are the full estimators)
    weighted_descent_estimators = [WeightedKinematicCostDescentEstimator(weighted_descent, window_size=i,
                                                                         non_negative=non_neg)
                                   for i in window_sizes for non_neg in [False, True] ]

    # Define the control laws descending each cost separately to check orthogonality
    config_descent = KinematicCostDescent(1.0, config_cost, frame_tracker, 'flap1', twist_control=True)

    # And their respective controllers
    config_controllers = []
    config_selectors = []
    input_averagers = []

    for window_size in window_sizes:
        config_controllers.append(DynamicController(config_descent.copy(), persistent_control=False))
        config_controllers.append(DynamicController(config_descent.copy(), persistent_control=False))
        # And a selector to extract the config weight as the descent rate
        config_selectors.append(Selector({'config':'rate'}))
        config_selectors.append(Selector({'config':'rate'}))
        input_averagers.append(Averager(window_size))

    # Define the measurement block
    mocap_measurement = MocapMeasurement(mocap_array, 'mocap_measurement')

    # Define the frame tracker used for the estimator of the input
    joints = kin_tree.get_joints()
    flap_point_strings = kin_tree.get_features(joints['joint4']).keys()
    flap_points = [int(s.split('_')[1]) for s in flap_point_strings]

    input_frame_tracker = MocapFrameTracker('input_tracker', flap_points)
    input_frame_estimator = MocapFrameEstimator(input_frame_tracker, 'flap1')
    input_estimator = Differentiator(fixed_step=1.0/FRAMERATE)

    # Define the system state tracker used to estimate the system state
    kin_tree_tracker = KinematicTreeTracker('tree_tracker', kin_tree)
    system_state = MocapSystemState([kin_tree_tracker])

    # Build up the system nodes
    measurement_node = ForwardBlockNode(mocap_measurement, 'Mocap Measurement', 'raw_mocap')
    input_frame_node = ForwardBlockNode(input_frame_estimator, 'Input Frame Estimator', 'input_flap')
    system_state_node = ForwardBlockNode(system_state, 'State Estimator', 'object_joints')
    differentiator_node = ForwardBlockNode(input_estimator, 'Differentiator', 'd_base_input_flap')
    estimator_nodes = [ForwardBlockNode(weighted_descent_estimator,'%s Weighted Cost Descent Estimator (%d)'
                                     % (typ, weighted_descent_estimator.get_window_size()),
                                     '%s_weights_%d' % (typ.lower(), weighted_descent_estimator.get_window_size()))
                    for weighted_descent_estimator, typ in zip(weighted_descent_estimators, types)]

    controller_selector_nodes = []

    for i, window_size in enumerate(window_sizes):
        config_controller_node = ForwardBlockNode(config_controllers[2*i], 'Config Controller (%d)' % window_size,
                                                  'input_config_controller_%d' % window_size)

        selector_node = ForwardBlockNode(config_selectors[2*i], 'Config Selector (%d)' % window_size, 'config_select_%d'
                                         % window_size)

        input_averager_node = ForwardBlockNode(input_averagers[i], 'Input Averager (%d)' % window_size,
                                               'input_average_%d' % window_size)

        nnls_config_controller_node = ForwardBlockNode(config_controllers[2*i+1], 'NNLS Config Controller (%d)' % window_size,
                                                  'nnls_input_config_controller_%d' % window_size)

        nnls_selector_node = ForwardBlockNode(config_selectors[2*i+1], 'NNLS Config Selector (%d)' % window_size,
                                              'nnls_config_select_%d' % window_size)


        estimator_nodes[2*i].add_output(selector_node, 'states')
        estimator_nodes[2*i+1].add_output(nnls_selector_node, 'states')
        selector_node.add_output(config_controller_node, 'parameters')
        nnls_selector_node.add_output(nnls_config_controller_node, 'parameters')

        system_state_node.add_output(config_controller_node, 'states')
        system_state_node.add_output(nnls_config_controller_node, 'states')

        differentiator_node.add_output(input_averager_node, 'states')

        controller_selector_nodes.append((config_controller_node, nnls_config_controller_node, selector_node,
                                          nnls_selector_node, input_averager_node))

    measurement_node.add_output(input_frame_node, 'measurement')
    measurement_node.add_output(system_state_node, 'measurement')
    input_frame_node.add_output(differentiator_node, 'states')

    for estimator_node in estimator_nodes:
        differentiator_node.add_output(estimator_node, 'inputs')
        system_state_node.add_output(estimator_node, 'states')

    # Define the root (all source nodes)
    root = ForwardRoot([measurement_node])

    # Create the system
    system = ForwardSystem(root)

    # Draw the block diagram if requested
    if diagram_filename is not None:
        system.draw(filename=diagram_filename)

    # all the data for every timestep
    all_data = system.run(record=True, print_steps=50)

    return all_data


if __name__ == '__main__':
    main()
