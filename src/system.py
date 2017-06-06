#!/usr/bin/env python
from abc import ABCMeta, abstractmethod
from inspect import getargspec
from graphviz import Digraph
import time


class Steppable(object):
    """An abstract base class for all objects that can be stepped through iteratively within a System"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def step(self, **kwargs):
        pass

    def get_step_inputs(self):
        """
        Returns the names of the inputs to the step function
        :return: a list of strings which are the names of the inputs to the step function
        """
        return getargspec(self.step)[0][1:]


class BlockNode(object):
    """
    A class which contains a Steppable, and connects other BlockNode's Steppables to it's own Steppable's step function
    """

    def __init__(self, steppable, name):
        """
        Constructor
        :param Steppable steppable: a Steppable object
        :param name: a name for the object
        """
        assert isinstance(steppable, Steppable)
        self.steppable = steppable
        self.name = name
        self.step_arg_names = steppable.get_step_inputs()
        self.inputs = {}

    def add_input(self, other_node, step_arg_name, replace=True):
        """
        Connects another node's output to one of the input arguments of it's step function
        :param other_node: the other node to be connected
        :param step_arg_name: the argument name of step which other node's output will be passed as
        :param replace: if another node is already connected under that name, this will replace it if true
        :return: None
        """
        assert step_arg_name not in self.inputs or replace, "input '%s' already exists and replace is set to False" % \
                                                            step_arg_name

        assert step_arg_name in self.step_arg_names, "'%s' is not an input argument of %s. Input arguments are: %s" % \
                                                     (step_arg_name, type(self.steppable), self.step_arg_names)

        assert isinstance(other_node, BlockNode), "other_node must be another BlockNode object."

        self.inputs[step_arg_name] = other_node

    def is_locally_ready(self):
        """
        Checks if all the inputs of the step function have been connected to another node
        :return:
        """
        if set(self.inputs) == set(self.step_arg_names):
            return True
        else:
            print('The following inputs of %s are not connected: %s' %
                  (self.name, set(self.step_arg_names) - set(self.inputs)))
            return False

    def is_ready(self):
        """
        Checks if all inputs are satisfied recursively
        :return:
        """
        return self.is_locally_ready() and all(input_node.is_ready() for input_node in self.inputs.items())

    def step(self, save_dict=None):
        """
        Calls its Steppable's step function, sourcing the inputs recursively
        :param save_dict: a dictionary with which to save the inputs
        :return: the output of the step call
        """
        # store the input value as the output of the input node's step function
        inputs = {input_name: self.inputs[input_name].step(save_dict) for input_name in self.inputs}

        # if a save_dict was passed, update it with the inputs
        if save_dict is not None:
            save_dict.update(inputs)

        # return the result of this Steppable's step call
        return self.steppable.step(**inputs)

    def is_source(self):
        """Checks if this node requires inputs"""
        return len(self.step_arg_names) == 0

    def get_name(self):
        """Returns the name of the node"""
        return self.name

    def get_input_nodes(self):
        """Returns the dict of input nodes connected"""
        return self.inputs


class System(object):

    def __init__(self, output_node, output_function=None, output_name='output'):
        """
        Constructor
        :param output_node: the last node in the system pipeline
        :param output_function: a function that will process the output
        :param output_name: a name to assign the output of the last node
        """
        assert isinstance(output_node, BlockNode), 'output_node must be a BlockNode object'
        assert output_node.is_ready(), 'output_node is not fully connected'

        self.output_node = output_node
        self.output_function = output_function
        self.output_name = output_name
        self.history = []
        self.start_time=0

    def step(self, record):
        """
        Runs through one iteration of the system
        :param record: if True, records the edges in the history
        :return: None
        """
        edges = {}
        edges[self.output_name] = self.output_node.step(edges)
        if record:
            self.history.append((time.time()-self.start_time, edges))

        if self.output_function is not None:
            self.output_function(edges[self.output_name])

    def clear(self):
        """Clears the history"""
        self.history = []

    def run(self, record=True):
        """
        Runs the system
        :param record: If true, the system edges will be saved each step
        :return: a list of the system edges for each timestep
        """
        self.start_time = time.time()
        while True:
            try:
                self.step(record)
            except (EOFError, KeyboardInterrupt):
                break

        return list(self.history)

    def draw(self, filename='block_diagram.gv'):
        """
        Draws a block diagram of the system using graphviz
        :param filename: the filename with which to save the output diagram
        :return: None
        """
        dot = Digraph("System Block Diagram", graph_attr={'rankdir': 'LR'})
        dot.node(self.output_name, shape='plain_text')
        shape = 'ellipse' if self.output_node.is_source() else 'box'
        dot.node(self.output_node.get_name(), shape=shape)
        dot.edge(self.output_node.get_name(), self.output_name)
        self._draw_inputs(dot, self.output_node)
        dot.render(filename, view=True)

    def _draw_inputs(self, dot, block_node):
        """
        Recursively draws block diagram from BlockNodes.
        Assumes that block_node has already been added to the dot graph under the name returned by block_node.get_name()
        :param Digraph dot:
        :param BlockNode block_node:
        :param str parent_node_name:
        :return: None
        """

        # BASE CASE: If the block_node is a source it has no inputs, return
        if block_node.is_source():
            return

        # RECURSIVE CASE: Otherwise draw each of it's inputs, and call _draw_inputs on each of them
        else:
            for input_name, input_node in block_node.get_input_nodes().items():
                # We will draw sources as ellipses and process blocks as boxes
                shape = 'ellipse' if input_node.is_source() else 'box'

                # Add the node to the graph using it's name
                dot.node(input_node.get_name(), shape=shape)

                # Add an edge between the input node and the current node, labelling it with its input_name
                # NOTE: The input_node name is different to the input_name. E.g a Controller might have an input named
                # 'states' which is produced by a Tracker named 'State Tracker'. 'states' is the input_name and
                # 'State Tracker' is the input_node.get_name().
                dot.edge(input_node.get_name(), block_node.get_name(), label=input_name)

                # draw this nodes inputs
                self._draw_inputs(dot, input_node)