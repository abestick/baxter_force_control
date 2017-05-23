#!/usr/bin/env python
import argparse
import numpy as np
from kinmodel.track_mocap import KinematicTreeTracker, WristTracker, KinematicTreeExternalFrameTracker
from system_state import OfflineSystem
from motion_costs import WeightedCostCombination, BasisManipulabilityCost, QuadraticDisplacementCost
import phasespace.load_mocap as load_mocap
import kinmodel
from phasespace.mocap_definitions import MocapWrist


FRAMERATE = 50
GROUP_NAME = 'tree'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kinmodel_json_optimized', help='The kinematic model JSON file')
    parser.add_argument('task_npz')
    #parser.add_argument('trials', help='The number of trials')
    args = parser.parse_args()

    # Load the calibration sequence
    data = np.load(args.task_npz)
    #mocap_data = [data['full_sequence_' + str(trial)] for trial in range(int(args.trials))]
    trials = 0
    while 'full_sequence_' + str(trials) in data.keys():
        trials += 1
    print(trials)
    mocap_data = [data['full_sequence_' + str(trial)] for trial in range(trials)]
    print([d.shape[:2] for d in mocap_data])
    all_trials = np.concatenate(mocap_data, axis=2)
    ukf_mocap = load_mocap.MocapArray(all_trials, FRAMERATE)

    tracker_kin_tree = kinmodel.KinematicTree(json_filename=args.kinmodel_json_optimized)
    kin_tree = kinmodel.KinematicTree(json_filename=args.kinmodel_json_optimized)

    object_tracker = KinematicTreeTracker('object', tracker_kin_tree, ukf_mocap)

    ref = 0
    while np.isnan(all_trials[:, :, ref]).any():
        ref += 1
    reference_frame = all_trials[:, :, ref]
    print("Reference Frame is frame %d" % ref)
    marker_indices_1 = {name: index + 16 for index, name in enumerate(MocapWrist.names[::-1])}
    marker_indices_2 = {name: index + 24 for index, name in enumerate(MocapWrist.names[::-1])}
    wrist_tracker_1 = WristTracker('wrist1', ukf_mocap, marker_indices_1, reference_frame)
    wrist_tracker_2 = WristTracker('wrist2', ukf_mocap, marker_indices_2, reference_frame)

    # config_reference = mocap_to_joint_angles(goal)
    config_reference = {object_tracker.name+'_'+'joint2': 0.066,
                        object_tracker.name+'_'+'joint3': 0.032,
                        object_tracker.name+'_'+'joint4': 0.017,
                        object_tracker.name+'_'+'joint5': 0.027}

    wrist_1_zero = {name: 0 for name in wrist_tracker_1.get_state_names()}
    wrist_2_zero = {name: 0 for name in wrist_tracker_2.get_state_names()}

    config_cost = QuadraticDisplacementCost(config_reference)
    wrist_1_cost = QuadraticDisplacementCost(wrist_1_zero)
    wrist_2_cost = QuadraticDisplacementCost(wrist_2_zero)

    costs = {"error_costs": WeightedCostCombination([config_cost, wrist_1_cost, wrist_2_cost])}

    system = OfflineSystem([object_tracker, wrist_tracker_1, wrist_tracker_2], costs)
    print(system.learn_weights())




if __name__ == '__main__':
    # cProfile.run('main()', 'fit_kinmodel.profile')
    main()