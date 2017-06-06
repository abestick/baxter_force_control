#!/usr/bin/env python
import argparse
import numpy as np
from kinmodel.track_mocap import KinematicTreeTracker, WristTracker, KinematicTreeExternalFrameTracker
from system_state import SystemState
from system import BlockNode, System
from control_law import WeightedKinematicCostDescent
from input_source import MocapInputTracker, WeightedKinematicCostDescentEstimator
from motion_costs import WeightedCostCombination, BasisManipulabilityCost, QuadraticDisplacementCost
import phasespace.load_mocap as load_mocap
import kinmodel
import json


FRAMERATE = 50
GROUP_NAME = 'tree'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kinmodel_json_optimized', help='The kinematic model JSON file')
    parser.add_argument('task_npz')
    parser.add_argument('block_diagram', help='The file name of the drawn block diagram of the system')
    parser.add_argument('system_history', help='The system output npz file')
    args = parser.parse_args()

    # Load the calibration sequence
    data = np.load(args.task_npz)
    trials = 0
    # Stack all the trials
    while 'full_sequence_' + str(trials) in data.keys():
        trials += 1
    print('Number of Trials: %d' % trials)
    mocap_data = [data['full_sequence_' + str(trial)] for trial in range(trials)]
    all_trials = np.concatenate(mocap_data, axis=2)

    # Put into a MocapArray
    ukf_mocap = load_mocap.ArrayMocapSource(all_trials, FRAMERATE)

    # Initialize the tree
    kin_tree = kinmodel.KinematicTree(json_filename=args.kinmodel_json_optimized)

    # Initialize the trackers
    object_tracker = KinematicTreeTracker('object', kin_tree, ukf_mocap)

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
    weighted_descent_estimator = WeightedKinematicCostDescentEstimator(weighted_descent, window_size=40)

    # Define the system state tracker and input tracker
    system_state = SystemState([object_tracker])

    # TODO
    input_source = MocapInputTracker(object_tracker)

    # Build up the system nodes
    output_node = BlockNode(weighted_descent_estimator, 'Weighted Cost Descent Estimator')
    output_node.add_raw_input(system_state, "System State Tracker", 'states')
    output_node.add_raw_input(input_source, "Hand Pose Velocity Tracker", 'inputs')

    # An example of how we might want to pipe system output to something when running online
    def some_function_that_uses_the_weights(w):
        pass

    system = System(output_node, output_function=some_function_that_uses_the_weights, output_name='weights')
    system.draw(filename=args.block_diagram)

    # all the data for every timestep
    all_data = system.run(record=True)

    # convert list of dicts to a dict of lists
    trajectories = {element: [] for element in all_data[-1]}
    for time_slice in all_data:
        for element in trajectories:
            trajectories[element].append(time_slice[element])

    # convert dict of lists to dict of numpy arrays
    for element in trajectories:
        trajectories[element] = np.array(trajectories[element])

    # store the trajectory of every element in the system
    with open(args.system_history, 'w') as output_file:
        np.savez_compressed(output_file, **trajectories)
        print('System history saved to ' + args.output_data_npz)


if __name__ == '__main__':
    # cProfile.run('main()', 'fit_kinmodel.profile')
    main()