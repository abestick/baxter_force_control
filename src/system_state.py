#!/usr/bin/env python
import rospy
from motion_costs import WeightedCostCombination
import numpy as np
from collections import OrderedDict


class SystemState(object):

    def __init__(self, state_trackers, kinematic_system_tracker, cost_functions):
        """
        Constructor
        :param list state_trackers: a list of trackers which each contain a step function that return a dict of states
        :param dict cost_functions: a dict of StateCost objects which each contain a cost function that takes states
        """
        self.state_trackers = state_trackers
        self.cost_functions = cost_functions
        self.kinematic_system_tracker = kinematic_system_tracker

        self.state_sources = {state_name: tracker.name for tracker in self.state_trackers
                              for state_name in tracker.get_state_names()}

        self.costs = dict.fromkeys(self.cost_functions.keys())
        self.states = dict.fromkeys(self.state_sources.keys())
        self.observables = dict.fromkeys(self.kinematic_system_tracker.get_observables().keys())
        self.system_states = self.states.copy().update(self.observables)

        self.iterations = 0
        self.jacobian_groups = self.kinematic_system_tracker.jacobian_groups()

    def step(self):
        """
        Performs one cycle, getting the full state estimate from all the trackers and computing all the costs
        :return: 
        """
        for i in range(len(self.state_trackers)):
            self.states.update(self.state_trackers[i].step())

        self.observables.update(self.kinematic_system_tracker.get_observables(self.states))
        self.system_states.update(self.states)
        self.system_states.update(self.observables)

        for cost_name in self.cost_functions:
            self.costs[cost_name] = self.cost_functions[cost_name].cost(self.states)

        self.iterations += 1

    def old_partial_derivative(self, function_output, function_input, states=None):
        """
        
        :param function_output: A list of strings which is the output vector of the function
        :param function_input: A list of strings which is the input vector to the function
        :param states: the system states, if None will default to the current state
        :return: 
        """

        if states is None:
            states = self.states

        # Create sets of each of the vectors
        output_set = set(function_output)
        input_set = set(function_input)

        # Check for duplicates within a single vector
        assert len(output_set) == len(function_output), "Duplicates found in function_output!"
        assert len(input_set) == len(function_input), "Duplicates found in function_input!"

        # We will create a dictionary of groups that these states pertain to
        output_groups = {}
        input_groups = {}

        for group_name in self.jacobian_groups:
            group_set = set(self.jacobian_groups[group_name])

            #  if there are some elements that are part of this group, add the group
            if len(output_set & group_set) > 0:
                output_groups[group_name] = output_set & group_set

            if len(input_set & group_set) > 0:
                input_groups[group_name] = output_set & group_set

        redundant_full_jacobian = None

        # Now we know all the jacobians we will need subsets of to compute the partial derivative
        for output_group_name in output_groups:
            column_block = None
            for input_group_name in input_groups:
                # partial is a Jacobian object
                partial = self.kinematic_system_tracker.partial_derivative(output_group_name, input_group_name, states)

                # Stack this Jacobian ontop of the others. If column_block is None, it will return partial
                column_block = partial.append_vertically(column_block)

            redundant_full_jacobian = column_block.hstack(redundant_full_jacobian)

        # Now we have a big jacobian with all the states needed and maybe some unnecessary ones
        trimmed_jacobian = redundant_full_jacobian.subset(row_names=list(function_output),
                                                          column_names=list(function_input))

        return trimmed_jacobian

    def partial_derivative(self, coordinate_frame, function_output, function_input, states=None):
        """
        
        :param function_output: A list of strings which is the output vector of the function
        :param function_input: A list of strings which is the input vector to the function
        :param states: the system states, if None will default to the current state
        :return: 
        """

        if states is None:
            states = self.states

        full_partial_derivative = self.kinematic_system_tracker.full_partial_derivative(coordinate_frame, states)
        return full_partial_derivative.subset(row_names=list(function_output),
                                              column_names=list(function_input))

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
            tracker_estimations, _, _ = tracker.run(record=True)
            all_tracker_estimations.append(tracker_estimations)

        # List where we will store each full estimation per timestep
        all_merged_estimations = []

        all_costs = []

        # only WeightedCostCombination objects have a cost_basis_vector function
        jacobian_bases_dict = {cost_name: [] for cost_name in self.cost_functions
                               if isinstance(self.cost_functions[cost_name],
                                             WeightedCostCombination) and jacobian_bases}

        # timestep_estimation is a tuple of dicts across all trackers on one timestep
        for timestep_estimations in zip(*all_tracker_estimations):

            # initialize the dict that will store all the merged state estimates
            timestep_merged_estimations = {}

            costs = {}

            # iterate through each trackers estimate for this timestep adding its estimates to the merged dict
            for timestep_tracker_estimation in timestep_estimations:
                timestep_merged_estimations.update(timestep_tracker_estimation)

            timestep_merged_estimations.update(self.kinematic_system_tracker.get_observables(timestep_merged_estimations))

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

        return all_merged_estimations, all_costs, jacobian_bases_dict

    def learn_weights(self, input_states_names, input_states_frame):

        # make sure input_states is ordered
        input_states_names = list(input_states_names)

        # empty dict for the weights of each cost
        weights = {}

        # calculate the basis vector of costs for each time step for each cost function
        all_states, _, all_jacobian_bases = self.run_through(jacobian_bases=True)

        y = np.array([state_estimate[input_state] for state_estimate in all_states
                      for input_state in input_states_names])

        # For every weighted cost function
        # costname maps to a list of dicts across time
        for cost_name in all_jacobian_bases:

            A_columns = OrderedDict()

            # Each key of all_jacobian_bases is a list of dicts, each dict containing the jacobians for each cost basis
            # for that timestep
            for timestep_jacobian, state_estimate in zip(all_jacobian_bases[cost_name], all_states):

                # Each timestep_jacobian is a dict of basis jacobians
                for cost_basis_name in timestep_jacobian:

                    # the jacobian of cost as a function of its relevant states
                    # this is a (1, Xc) jacobian
                    cost_coststates_jacobian = timestep_jacobian[cost_basis_name]

                    # get the states the cost depends on
                    coststates = {state_name: state_estimate[state_name]
                                  for state_name in cost_coststates_jacobian.column_names()}

                    input_states = {state_name: state_estimate[state_name]
                                    for state_name in input_states_names}

                    # get the jacobian between these cost states and the input states
                    # this is a (Xc, U) jacobian
                    coststates_input_jacobian = self.partial_derivative(input_states_frame,
                                                                        coststates, input_states, state_estimate)

                    # compute the jacobian for the cost as a function of input via the chain rule
                    # produces a (1, U) jacobian which we stack vertically to match y
                    cost_input_jacobian = cost_coststates_jacobian * coststates_input_jacobian

                    # if this is the first timestep, get will return an empty column matrix
                    A_columns[cost_basis_name] = np.concatenate([A_columns.get(cost_basis_name, np.empty((0,1))),
                                                                 cost_input_jacobian.J().T], axis=0)

            # stack the columns of the A matrix horizontally giving y=Aw for all timesteps
            A = np.concatenate(A_columns.values(), axis=1)

            # learn the weights
            w = np.linalg.lstsq(A, y)

            weights[cost_name] = {cost_basis_name: w[0][i] for i, cost_basis_name in enumerate(A_columns.keys())}

        return weights
