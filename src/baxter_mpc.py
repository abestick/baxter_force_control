#!/usr/bin/python

import rospy
import baxter_interface
from baxter_pykdl import baxter_kinematics
import numpy as np
import numpy.linalg as npla
from math import sqrt
import threading
import tf.transformations
import tf
from inspect import getargspec, ismethod

MAX_JOINT_VEL = 1.5 # Max commanded joint velocity
MAX_NS_JOINT_VEL = 0.5 # Max nullspace joint velocity
TIME_STEP = 0.05 # time delta 

# Arm neutral configs
NEUTRAL = {'left': 
        {'left_w0': 0.0,
         'left_w1': -0.55,
         'left_w2': 0.0,
         'left_e0': 0.75,
         'left_e1': 0.0,
         'left_s0': 1.26,
         'left_s1': 0.0},
         'right': 
        {'right_w0': 0.0,
         'right_w1': -0.55,
         'right_w2': 0.0,
         'right_e0': 0.75,
         'right_e1': 0.0,
         'right_s0': 1.26,
         'right_s1': 0.0}}


class DiscreteNLS():
    """
    A class describing a discrete nonlinear system
    It takes the form:
    x(k+1) = x(k) + f(x(k),u(k))
    y(k) = g(x(k), u(k))

    """

    def __init__(self, f, g, x_0, time_step=1.0, t_0=0.0):

        if not check_arg_count(f, 2):
            raise ValueError("The system function f must take 2 arguments")

        if not check_arg_count(g, 2):
            raise ValueError("The system function g must take 2 arguments")

        self.f = f
        self.g = g
        self.x_0 = np.array(x_0)
        self.x = self.x_0.copy()
        self.time_step = time_step
        self.t_0 = t_0
        self.t = self.t_0
        self.k = 0
        self.y = None

    def reset(self):
        """
        Resets the system to initial conditions
        """

        self.t = self.t_0
        self.k = 0
        self.x = self.x_0.copy()
    
    def step(self, u):
        """
        Steps the system by one time step
        """
        
        self.x += self.f(self.x, u)
        self.y = self.g(self.x, u)
        self.k += 1
        self.t += self.time_step
        return self.x, self.y


class LinearQuadraticCost():

    def __init__(self, Q, R, T):
        """
        
        :param Q: the state cost 
        :param R: the input cost
        :param T: the terminal cost
        """


class NMPC():
    """
    A Nonlinear Model Predictive Controller class

    Args:
    limb: 'left'|'right'
    """

    def __init__(self, model, cost_function):
        self.model = model
        self.cost_function = cost_function


class ArmNMPC():
    """
    A container class for an NMPC controller specific to the Baxter Arm
    Args:
    limb: 'left'|'right'
    """

    def __init__(self, limb, time_step):
        self._neutral_config = NEUTRAL[limb]
        self._kin = baxter_kinematics(limb)
        self._limb = baxter_interface.Limb(limb)
        self.time_step = time_step
        
        # Choose the endpoint's current pose as the starting setpoint
        x_0 = np.array(self._limb.endpoint_pose()['position'] + 
                self._limb.endpoint_pose()['orientation'])

        # Define discrete NLS system
        self.system = DiscreteNLS(self.f, self.g, x_0, time_step=TIME_STEP)

    def f(self, x, u):
        """
        The state increment function of the discrete NLS
        """

        return self.time_step * u
    
    def g(self, x, u):
        """
        The output function of the discrete NLS. This should be the inverse kinematics
        """

        return np.dot(self._kin.jacobian(), x)

    def set_neutral_config(self, neutral_config_dict):
        """Sets the arm's neutral configuration to something other than the default value.

        Args:
        neutral_config_dict: dict - {'left_s1':value, ..., 'left_w2':value}
        """
        self._neutral_config = neutral_config_dict


def check_arg_count(func, args_to_pass):
    
    # get the specs of the function
    argspec = getargspec(func)

    # check for args or kwargs in which case we can pass any number of args
    if argspec.varargs is not None or argspec.keywords is not None:
        return True

    # otherwise find range of acceptable arguments
    max_args = len(argspec.args) - int(ismethod(func))
    min_args = max_args - len(argspec.defaults)

    # return true if passing args_to_pass many args would work
    return  min_args <= args_to_pass <= max_args


def main():
    rospy.init_node('baxter_mpc')

    rospy.spin()

if __name__ == "__main__":
    main()
