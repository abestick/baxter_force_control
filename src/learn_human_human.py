#!/usr/bin/env python
import argparse
import numpy as np
from kinmodel.track_mocap import KinematicTreeTracker, KinematicTreeExternalFrameTracker, MocapFrameTracker
from system import ForwardBlockNode, ForwardSystem, ForwardRoot
from control_law import WeightedKinematicCostDescent
from steppables import MocapFrameEstimator, WeightedKinematicCostDescentEstimator, MocapSystemState, MocapMeasurement, \
    Differentiator, Transformer
from motion_costs import WeightedCostCombination, BasisManipulabilityCost, QuadraticDisplacementCost
import phasespace.load_mocap as load_mocap
import kinmodel


FRAMERATE = 50
GROUP_NAME = 'tree'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kinmodel_json_optimized', help='The kinematic model JSON file')
    parser.add_argument('task_npz')
    parser.add_argument('block_diagram', help='The file name of the drawn block diagram of the system')
    parser.add_argument('system_history', help='The system output npz file')
    args = parser.parse_args()

    window_sizes = (10, 50, 100)

    # Load the calibration sequence
    data = np.load(args.task_npz)
    trials = 0
    # Stack all the trials
    while 'full_sequence_' + str(trials) in data.keys():
        trials += 1
    print('Number of trials: %d' % trials)
    mocap_data = [data['full_sequence_' + str(trial)] for trial in range(trials)]
    mocap_data.append(np.concatenate(mocap_data, axis=2))
    all_trials = {}

    # Initialize the tree
    kin_tree = kinmodel.KinematicTree(json_filename=args.kinmodel_json_optimized)

    block_diag = None

    for i, trial in enumerate(mocap_data):
        if i == len(mocap_data)-1:
            i = 'ALL'
            block_diag = args.block_diagram

        print('Learning trial ' + str(i) + '...')
        times, trajectories = learn(trial, kin_tree, window_sizes, block_diag)
        block_diag = None

        append_keys(trajectories, '_' + str(i))
        all_trials.update(trajectories)
        all_trials['time_' + str(i)] = times

    all_trials['num_trials'] = len(mocap_data)

    filename = args.system_history

    save(filename, all_trials)


def append_keys(dictionary, suffix):
    keys = dictionary.keys()
    for key in keys:
        dictionary[key+suffix] = dictionary.pop(key)



def save(filename, data):
    # store the trajectory of every element in the system
    with open(filename, 'w') as output_file:
        np.savez_compressed(output_file, **data)
        print('System history saved to ' + filename)


def learn(trial, kin_tree, window_sizes, diagram_filename=None):
    # Put into a MocapArray
    mocap_array = load_mocap.ArrayMocapSource(trial, FRAMERATE)

    # Append window_sizes to include a total estimation for all data
    window_sizes = window_sizes + (len(mocap_array),)

    print('Number of data points: %d' % len(mocap_array))

    # Initialize the frame tracker and attach frames of interest
    frame_tracker = KinematicTreeExternalFrameTracker(kin_tree.copy())
    frame_tracker.attach_frame('joint1', 'base')
    frame_tracker.attach_frame('joint4', 'flap1')
    frame_tracker.attach_frame('joint5', 'flap2')

    # Create a function that takes a joint angles dict and returns the jacobian for flap1
    def get_flap1_jacobian(object_joint_angles):
        return frame_tracker.compute_jacobian('base', 'flap1', object_joint_angles)

    # Hard coded config reference from looking at the raw data
    config_reference = {'joint2': 0.066,
                        'joint3': 0.032,
                        'joint4': 0.017,
                        'joint5': 0.027}


    # Create a config cost and bases of manipulability costs
    config_cost = QuadraticDisplacementCost('config', config_reference, lambda x: x, config_reference.keys())
    manip_cost_x = BasisManipulabilityCost('manip_x', config_reference.keys(), get_flap1_jacobian, 'flap1_x')
    manip_cost_y = BasisManipulabilityCost('manip_y', config_reference.keys(), get_flap1_jacobian, 'flap1_y')
    manip_cost_z = BasisManipulabilityCost('manip_z', config_reference.keys(), get_flap1_jacobian, 'flap1_z')

    # Put these into a weighted cost and put the weighted cost into the system cost dictionary
    weighted_cost = WeightedCostCombination('object_costs', [config_cost, manip_cost_x, manip_cost_y, manip_cost_z])

    # Define the control law we will estimate and an estimator
    weighted_descent = WeightedKinematicCostDescent(1.0, weighted_cost, frame_tracker, 'flap1')
    weighted_descent_estimators = [WeightedKinematicCostDescentEstimator(weighted_descent, window_size=i) for i in window_sizes]

    # Define the measurement block
    mocap_measurement = MocapMeasurement(mocap_array, 'mocap_measurement')

    # Define the frame tracker used for the estimator of the input
    joints = kin_tree.get_joints()
    flap_point_strings = kin_tree.get_features(joints['joint3']).keys()
    flap_points = [int(s.split('_')[1]) for s in flap_point_strings]
    base_point_strings = kin_tree.get_features(joints['joint1']).keys()
    base_points = [int(s.split('_')[1]) for s in base_point_strings]

    input_frame_tracker = MocapFrameTracker('input_tracker', flap_points)
    base_frame_tracker = MocapFrameTracker('base_tracker', base_points)
    input_frame_estimator = MocapFrameEstimator(input_frame_tracker, 'flap1')
    input_estimator = Differentiator()
    base_estimator = MocapFrameEstimator(base_frame_tracker, 'base')
    input_transformer = Transformer()

    # Define the system state tracker used to estimate the system state
    kin_tree_tracker = KinematicTreeTracker('tree_tracker', kin_tree)
    system_state = MocapSystemState([kin_tree_tracker])

    # Build up the system nodes
    measurement_node = ForwardBlockNode(mocap_measurement, 'Mocap Measurement', 'raw_mocap')
    input_frame_node = ForwardBlockNode(input_frame_estimator, 'Input Frame Estimator', 'input_flap')
    system_state_node = ForwardBlockNode(system_state, 'State Estimator', 'object_joints')
    differentiator_node = ForwardBlockNode(input_estimator, 'Differentiator', 'd_base_input_flap')
    base_frame_node = ForwardBlockNode(base_estimator, 'Base Frame Estimator', 'base_transform')
    transformer_node = ForwardBlockNode(input_transformer, 'Base Frame Transformer', 'base_input_flap')
    output_nodes = [ForwardBlockNode(weighted_descent_estimator,'Weighted Cost Descent Estimator (%d)'
                                     % weighted_descent_estimator.get_window_size(),
                                     'weights_%d' % weighted_descent_estimator.get_window_size())
                    for weighted_descent_estimator in weighted_descent_estimators]

    measurement_node.add_output(input_frame_node, 'measurement')
    measurement_node.add_output(system_state_node, 'measurement')
    measurement_node.add_output(base_frame_node, 'measurement')
    input_frame_node.add_output(transformer_node, 'primitives')
    base_frame_node.add_output(transformer_node, 'transform')
    transformer_node.add_output(differentiator_node, 'states')
    for output_node in output_nodes:
        differentiator_node.add_output(output_node, 'inputs')
        system_state_node.add_output(output_node, 'states')

    root = ForwardRoot([measurement_node])

    system = ForwardSystem(root)

    if diagram_filename is not None:
        system.draw(filename=diagram_filename)

    # all the data for every timestep
    all_times, all_data = zip(*system.run(record=True, print_steps=50))
    # convert list of dicts to a dict of lists
    trajectories = {element: [] for element in all_data[-1]}
    for time_slice in all_data:
        for element in trajectories:
            trajectories[element].append(time_slice[element])

    # convert dict of lists to dict of numpy arrays
    for element in trajectories:
        trajectories[element] = np.array(trajectories[element])

    return all_times, trajectories


if __name__ == '__main__':
    # cProfile.run('main()', 'fit_kinmodel.profile')
    main()