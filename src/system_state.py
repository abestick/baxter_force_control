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

    def run_through(self, basis_vectors=False):
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
        basis_vectors_dict = {cost_name: [] for cost_name in self.cost_functions
                              if isinstance(self.cost_functions[cost_name], WeightedCostCombination) and basis_vectors}

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
                if cost_name in basis_vectors_dict:
                        basis_vectors_dict[cost_name].append(self.cost_functions[cost_name].cost_basis_vector(
                            timestep_merged_estimations))

            # add this timesteps costs to the list across time
            all_costs.append(costs)

        # stack the
        for cost_name in basis_vectors_dict:
            basis_vectors_dict[cost_name] = np.vstack(basis_vectors_dict[cost_name])

        return all_merged_estimations, all_costs, basis_vectors_dict

    def learn_weights(self):

        # empty dict for the weights of each cost
        weights = {}

        # calculate the basis vector of costs for each time step for each cost function
        _, _, basis_vectors = self.run_through(basis_vectors=True)
        print(basis_vectors)

        def total_cost(x, bases):
            return np.sum(bases.dot(x))

        equality_constraint = [{'type': 'eq',
                                'fun': lambda x: np.array(np.sum(x) - 1.0),
                                'jac': lambda x: np.ones_like(x)}]

        # find the weights that minimze cost across time for each cost function
        for cost_name in basis_vectors:

            # Make the initial guess weighting each cost inversely proportional to their total accumulated cost
            inverse_totals = 1.0 / np.sum(basis_vectors[cost_name], axis=0)
            x0 = inverse_totals / np.sum(inverse_totals)
            print(x0)
            '''
            Is x0 actually our solution? The below minimization will always fully weight the lowest cost and apply no
            weight to the others. This makes sense since we do not know the constraints under which the human is
            minimizing and therefore we do not know the set of possible cost basis vectors from which the measured one
            is the weighted minimum.
            '''

            # the jacobians of the inequality constraints for each ith element of x will simply be a zero vector with a
            # 1 at the ith index. This corresponds to the ith row of the identity.
            inequality_jacs = np.identity(len(x0))
            inequality_constraints = [{'type': 'ineq',
                                       'fun': lambda x, i: np.array([x[i]]),
                                       'jac': lambda x, i: inequality_jacs[i,:],
                                       'args': (i,)} for i in range(len(x0))]

            constraints = equality_constraint + inequality_constraints

            weights[cost_name] = minimize(total_cost, x0, args=(basis_vectors[cost_name],), constraints=constraints)

        return weights