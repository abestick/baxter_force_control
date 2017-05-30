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
    args = parser.parse_args()

    # Load the calibration sequence
    data = np.load(args.task_npz)
    trials = 0

    # Stack all the trials
    while 'full_sequence_' + str(trials) in data.keys():
        trials += 1
    print(trials)
    mocap_data = [data['full_sequence_' + str(trial)] for trial in range(trials)]
    print([d.shape[:2] for d in mocap_data])
    all_trials = np.concatenate(mocap_data, axis=2)

    # Put into a MocapArray
    ukf_mocap = load_mocap.MocapArray(all_trials, FRAMERATE)

    # Initialize the tree
    kin_tree = kinmodel.KinematicTree(json_filename=args.kinmodel_json_optimized)

    # Initialize the trackers
    object_tracker = KinematicTreeTracker('object', kin_tree, ukf_mocap)

    # Initialize the frame tracker and attach frames of interest
    frame_tracker = KinematicTreeExternalFrameTracker(kin_tree.copy())
    frame_tracker.connect_tracker(object_tracker)
    frame_tracker.attach_frame('joint1', 'base')
    frame_tracker.attach_frame('joint4', 'flap1')
    frame_tracker.attach_frame('joint5', 'flap2')

    # Create a function that takes a joint angles dict and returns the jacobian for flap1
    def get_flap1_jacobian(object_joint_angles):
        return frame_tracker.compute_jacobian('base', 'flap1', object_joint_angles)

    # Wrist stuff
    # ref = 0
    # while np.isnan(all_trials[:, :, ref]).any():
    #     ref += 1
    # reference_frame = all_trials[:, :, ref]
    # print("Reference Frame is frame %d" % ref)
    # marker_indices_1 = {name: index + 16 for index, name in enumerate(MocapWrist.names[::-1])}
    # marker_indices_2 = {name: index + 24 for index, name in enumerate(MocapWrist.names[::-1])}
    # # wrist_tracker_1 = WristTracker('wrist1', ukf_mocap, marker_indices_1, reference_frame)
    # wrist_tracker_2 = WristTracker('wrist2', ukf_mocap, marker_indices_2, reference_frame)

    # Hard coded config reference from looking at the raw data
    config_reference = {object_tracker.name+'_'+'joint2': 0.066,
                        object_tracker.name+'_'+'joint3': 0.032,
                        object_tracker.name+'_'+'joint4': 0.017,
                        object_tracker.name+'_'+'joint5': 0.027}

    # wrist_1_zero = {name: 0 for name in wrist_tracker_1.get_state_names()}
    # wrist_2_zero = {name: 0 for name in wrist_tracker_2.get_state_names()}

    # Create a config cost and bases of manipulability costs
    config_cost = QuadraticDisplacementCost('config', config_reference)
    manip_cost_x = BasisManipulabilityCost('manip_x', config_reference.keys(), get_flap1_jacobian, 'x')
    manip_cost_y = BasisManipulabilityCost('manip_y', config_reference.keys(), get_flap1_jacobian, 'y')
    manip_cost_z = BasisManipulabilityCost('manip_z', config_reference.keys(), get_flap1_jacobian, 'z')

    # Put these into a weighted cost and put the weighted cost into the system cost dictionary
    costs = {'object_costs': WeightedCostCombination('object_costs',
                                                     [config_cost, manip_cost_x, manip_cost_y, manip_cost_z])}

    # Create the offline system and learn the weights of the Weighted Cost
    system = OfflineSystem([object_tracker], frame_tracker, costs)
    print(system.learn_weights(['flap1_' + element for element in kinmodel.POSE_NAMES]))


if __name__ == '__main__':
    # cProfile.run('main()', 'fit_kinmodel.profile')
    main()