#!/usr/bin/env python
import rospy
from motion_costs import WeightedCostCombination
import numpy as np
import numpy.linalg as npla


class SystemState(object):

    def __init__(self, state_trackers, cost_functions):
        """
        Constructor
        :param list state_trackers: a list of trackers which each contain a step function that return a dict of states
        :param dict cost_functions: a dict of StateCost objects which each contain a cost function that takes states
        """
        self.state_trackers = state_trackers
        self.cost_functions = cost_functions
        self.costs = {}
        self.states = {}

    def step(self):
        """
        Performs one cycle, getting the full state estimate from all the trackers and computing all the costs
        :return: 
        """
        for i in range(len(self.state_trackers)):
            self.states.update(self.state_trackers[i].step())

        for cost_name in self.cost_functions:
            self.costs[cost_name] = self.cost_functions[cost_name].cost(self.states)


class OnlineSystem(SystemState):

    def __init__(self, rate, state_trackers, cost_function):
        """
        
        :param rate: if positive, the cycle rate in hertz, if negative, the index of the tracker to be the master whose
        rate all other trackers run at
        """

        super(OnlineSystem, self).__init__(state_trackers, cost_function)

        if rate > 0:
            self.timer = rospy.Timer(rospy.Duration(1.0/rate), self._step_wrapper)

        elif rate <= 0:
            master_idx = abs(rate)
            for tracker in self.state_trackers:
                tracker.register_callback(self._new_states_callback)

            slaves = list(self.state_trackers)
            slaves.pop(master_idx)
            self.master_tracker = self.state_trackers[master_idx]
            self.master_tracker.set_master(True)
            self.master_tracker.enslave(slaves)
            self.master_tracker.register_callback(self._update_costs_callback)

    def _new_states_callback(self, i, new_states, *args):
        self.states.update(new_states)

    def _update_costs_callback(self, *args):

        for cost_name in self.cost_functions:
            self.costs[cost_name] = self.cost_functions[cost_name].cost(self.states)

    def _step_wrapper(self, event):
        self.step()


class OfflineSystem(SystemState):

    def run_through(self, basis_vectors=False):
        """
        Runs through the whole system
        :return: 
        """

        # List where we will store each trackers estimations across all time
        all_tracker_estimations = []

        # append the list with each trackers estimations over time
        for tracker in self.state_trackers:
            tracker_estimations, _, _ = tracker.run(record=True)
            all_tracker_estimations.append(tracker_estimations)

        # List where we will store each full estimation per timestep
        all_merged_estimations = []

        all_costs = []

        basis_vectors = {cost_name: [] for cost_name in self.cost_functions}

        # timestep_estimation is a tuple of dicts across all trackers on one timestep
        for timestep_estimations in zip(*all_tracker_estimations):

            # initialize the dict that will store all the merged state estimates
            timestep_merged_estimations = {}

            costs = {}

            # iterate through each trackers estimate for this timestep adding its estimates to the merged dict
            for timestep_tracker_estimation in timestep_estimations:
                timestep_merged_estimations.update(timestep_tracker_estimation)

            # append this timesteps full estimate
            all_merged_estimations.append(timestep_merged_estimations)

            for cost_name in self.cost_functions:
                costs[cost_name] = self.cost_functions[cost_name].cost(self.states)

                if basis_vectors:
                    if isinstance(self.cost_functions[cost_name], WeightedCostCombination):
                        basis_vectors[cost_name].append(self.cost_functions[cost_name].cost_basis_vector(timestep_merged_estimations))

            all_costs.append(costs)

            for cost_name in basis_vectors:
                if len(basis_vectors[cost_name])==0:
                    basis_vectors.pop(cost_name)

                else:
                    basis_vectors[cost_name] = np.vstack(basis_vectors[cost_name])

        return all_merged_estimations, all_costs, basis_vectors

    def learn_weights(self):

        weights = {}

        _, _, basis_vectors = self.run_through(basis_vectors=True)

        for cost_name in basis_vectors:
            weights[cost_name] = npla.lstsq(basis_vectors[cost_name], np.zeros(len(basis_vectors[cost_name])))[0]

        return weights