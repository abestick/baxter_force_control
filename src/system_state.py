#!/usr/bin/env python
import rospy
from motion_costs import WeightedCostCombination
import numpy as np
from collections import OrderedDict
from system import Steppable
import time
from abc import ABCMeta, abstractmethod


class SystemState(Steppable):

    __metaclass__ = ABCMeta

    def __init__(self):
        self._states = {}

    def step(self):
        """
        Performs one cycle, getting the full state estimate from all the trackers and computing all the costs
        :return: 
        """

        self._step_states()
        return self._states.copy()

    def get_state_names(self):
        return self._states.keys()

    @abstractmethod
    def _step_states(self, **kwargs):
        pass


class MocapSystemState(SystemState):

    def __init__(self, mocap_trackers):
        self._mocap_trackers = mocap_trackers
        super(MocapSystemState, self).__init__()

    def _step_states(self, frame):
        for mocap_tracker in self._mocap_trackers:
            self._states.update(mocap_tracker.process_frame(frame))


class Differentiator(Steppable):

    def __init__(self):
        self.last_states = None
        self.last_time = None

    def step(self, states):
        now = time.time()
        if self.last_states is not None:
            derivs = {state_name: (states[state_name] - self.last_states[state_name]) /
                                (now - self.last_time) for state_name in states}

        else:
            derivs = {state_name: 0.0 for state_name in states}

        self.last_states = states.copy()
        self.last_time = now

        return derivs


# TODO: Check if there's any useful bits of code before deleting all below
class OnlineSystemState(SystemState):
    """
    Sets up a system state to track an online system. At initialization there is the option to fix the rate the system
    will run at or to choose one of the trackers to run as fast as possible, and time the others off this "master".
    """

    def __init__(self, rate, state_trackers):
        """
        Constructor
        :param rate: if positive, the cycle rate in hertz, if negative, the index of the tracker to be the master whose
        rate all other trackers run at
        """

        super(OnlineSystemState, self).__init__(state_trackers)

        self.timer = None
        self._timer_args = None
        self.master_tracker = None

        if rate > 0:
            self._timer_args = (rospy.Duration(1.0/rate), self._step_timer_callback)
            self.start = self._start_timer
            self.stop = self._stop_timer

        elif rate <= 0:
            # get the index of the master tracker
            master_idx = abs(rate)

            # set all trackers to update the system states at the end of each frame
            for tracker in self._state_trackers:
                tracker.register_callback(self._update_states_callback)

            # create a copy of the trackers without the master
            slaves = list(self._state_trackers)
            slaves.pop(master_idx)

            # save a reference to the tracker to be master
            self.master_tracker = self._state_trackers[master_idx]

            # set this tracker as the master and register the others as slaves, changing them to slave if needed
            self.master_tracker.set_master(True)
            self.master_tracker.enslave(slaves)

            self.start = self._start_master
            self.stop = self._stop_master

    def _update_states_callback(self, i, new_states, *args):
        """This will be called by each tracker and passed the dicks of the states they estimate"""
        self._states.update(new_states)

    def _step_timer_callback(self, event):
        """A wrapper for the step function since timer callbacks are passed a timer event object"""
        self.step()

    def _start_timer(self):
        self.timer = rospy.Timer(*self._timer_args)

    def _stop_timer(self):
        self.timer.shutdown()

    def _start_master(self):
        self.master_tracker.start()

    def _stop_master(self):
        self.master_tracker.stop()



class OfflineSystem(SystemState):

    def run_through(self, jacobian_bases=False):
        """
        Runs through the whole system
        :return: 
        """

        # List where we will store each trackers estimations across all time
        all_tracker_estimations = []

        # append the list with each trackers estimations over time
        for tracker in self._state_trackers:
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

            timestep_merged_estimations.update(self._frame_tracker.get_observables(timestep_merged_estimations))

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
