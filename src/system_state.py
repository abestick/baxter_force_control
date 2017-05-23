#!/usr/bin/env python
import rospy
from motion_costs import WeightedCostCombination
import numpy as np
from scipy.optimize import minimize


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

        self.iterations = 0

    def step(self):
        """
        Performs one cycle, getting the full state estimate from all the trackers and computing all the costs
        :return: 
        """
        for i in range(len(self.state_trackers)):
            self.states.update(self.state_trackers[i].step())

        for cost_name in self.cost_functions:
            self.costs[cost_name] = self.cost_functions[cost_name].cost(self.states)

        self.iterations += 1


class OnlineSystem(SystemState):
    """
    Sets up a system state to track an online system. At initialization there is the option to fix the rate the system
    will run at or to choose one of the trackers to run as fast as possible, and time the others of this "master". 
    """

    def __init__(self, rate, state_trackers, cost_function):
        """
        Constructor
        :param rate: if positive, the cycle rate in hertz, if negative, the index of the tracker to be the master whose
        rate all other trackers run at
        """

        super(OnlineSystem, self).__init__(state_trackers, cost_function)

        if rate > 0:
            self.timer = rospy.Timer(rospy.Duration(1.0/rate), self._step_wrapper)

        elif rate <= 0:
            # get the index of the master tracker
            master_idx = abs(rate)

            # set all trackers to update the system states at the end of each frame
            for tracker in self.state_trackers:
                tracker.register_callback(self._new_states_callback)

            # create a copy of the trackers without the master
            slaves = list(self.state_trackers)
            slaves.pop(master_idx)

            # save a reference to the tracker to be master
            self.master_tracker = self.state_trackers[master_idx]

            # set this tracker as the master and register the others as slaves, changing them to slave if needed
            self.master_tracker.set_master(True)
            self.master_tracker.enslave(slaves)

            # since the master callbacks are called last and in the order they are registered, this callback is called
            # once all the states have been updated so that we can update the costs and complete the cycle
            self.master_tracker.register_callback(self._update_costs_callback)

    def _new_states_callback(self, i, new_states, *args):
        """This will be called by each tracker and passed the dicks of the states they estimate"""
        self.states.update(new_states)

    def _update_costs_callback(self, *args):
        """This will be called by the master once all states are updated and will update the cost and finish the loop"""
        for cost_name in self.cost_functions:
            self.costs[cost_name] = self.cost_functions[cost_name].cost(self.states)

        self.iterations += 1

    def _step_wrapper(self, event):
        """A wrapper for the step function since timer callbacks are passed a timer event object"""
        self.step()


class OfflineSystem(SystemState):

    def run_through(self, jacobian_bases=False):
        """
        Runs through the whole system
        :return: 
        """

        # List where we will store each trackers estimations across all time
        all_tracker_estimations = []

        # append the list with each trackers estimations over time
        for tracker in self.state_trackers:
            print(tracker.name)
            tracker_estimations, _, _ = tracker.run(record=True)
            all_tracker_estimations.append(tracker_estimations)

        # List where we will store each full estimation per timestep
        all_merged_estimations = []

        all_costs = []

        # only WeightedCostCombination objects have a cost_basis_vector function
        jacobian_bases_dict = {cost_name: [] for cost_name in self.cost_functions
                               if isinstance(self.cost_functions[cost_name],
                                             WeightedCostCombination) and jacobian_bases}

        row_names = None
        column_names = None

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

            # calculate each of the costs
            for cost_name in self.cost_functions:
                costs[cost_name] = self.cost_functions[cost_name].cost(timestep_merged_estimations)

                # if we are calculating basis vectors, do so for compatible costs and append to the list across time
                if cost_name in jacobian_bases_dict:
                        jacobian_bases_dict[cost_name].append(self.cost_functions[cost_name].jacobian_bases(
                            timestep_merged_estimations))

            # add this timesteps costs to the list across time
            all_costs.append(costs)

        # stack the
        for cost_name in jacobian_bases_dict:
            jacobian_bases_dict[cost_name] = np.vstack(jacobian_bases_dict[cost_name])

        return all_merged_estimations, all_costs, jacobian_bases_dict

    def learn_weights(self, input_states):

        # empty dict for the weights of each cost
        weights = {}

        # calculate the basis vector of costs for each time step for each cost function
        all_states, _, all_jacobian_bases = self.run_through(jacobian_bases=True)
        print(all_jacobian_bases)

        # Initialize row order and column order to None so that the weighted cost automatically sets it
        row_names = None
        columns_names = None

        y =

        # For every weighted cost function
        # costname maps to a list of dicts across time
        for cost_name in all_jacobian_bases:

            # For every time instance for this cost1
            # jacobian_bases_dict is a dict of dicts, each key mapping to a jacobian for a cost basis
            for jacobian_bases_dict in all_jacobian_bases[cost_name]:
                # convert the dict of dicts into a matrix whereby each cost basis has a column and each state has a row
                jacobian_bases_matrix, row_names, columns_names = \
                    self.cost_functions[cost_name].jacobian_bases_matrix(jacobian_bases_dict)



        return weights