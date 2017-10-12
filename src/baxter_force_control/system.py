#!/usr/bin/env python
from __future__ import print_function
from abc import ABCMeta, abstractmethod
from graphviz import Digraph
import time
import pandas as pd
import rospy
from steppables import Steppable, PandaSink


class Node(object):
    """
    An abstract base class for system nodes. A node will have a step function which performs an iteration of its part of
    the system, and an is_connected function which asserts that the node graph is properly connected from this node
    onward.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def step(self, save_dict=None):
        pass

    @abstractmethod
    def is_connected(self):
        pass


class BlockNode(Node):
    """
    A class which contains a Steppable, and connects other BlockNode's Steppables to it's own Steppable's step function.
    It may be implemented as a Forward or Backward BlockNode, depending on the desired direction of node dependencies in
    the graph.
    """

    __metaclass__ = ABCMeta

    def __init__(self, steppable, name):
        """
        Constructor
        :param Steppable steppable: a Steppable object
        :param name: a name for the object
        """
        assert isinstance(steppable, Steppable)
        self._steppable = steppable
        self._name = name
        self._step_arg_names = steppable.get_step_inputs()

    def get_step_arg_names(self):
        """Returns the arguments required for the Steppable's step function"""
        return self._step_arg_names

    def steppable_type(self):
        """Returns the type of the Steppable"""
        return type(self._steppable)

    def get_name(self):
        """Returns the BlockNode's name"""
        return self._name

    def is_source(self):
        """Checks if this node requires inputs"""
        return len(self._step_arg_names) == 0


class ForwardBlockNode(BlockNode):
    """
    An implementation of BlockNode for which an iteration of the graph is initiated by sources then propogated forward
    to outputs. The Forward implementation is slightly more complicated and slower, but does not run into issues when
    two nodes share the same input.
    """

    def __init__(self, steppable, name, output_name):
        super(ForwardBlockNode, self).__init__(steppable, name)
        self._output_nodes = {}
        self._inputs = {}
        self._output_name = output_name
        self._connected_inputs = {arg_name: False for arg_name in self._step_arg_names}

    def _connect_input(self, step_arg_name):
        """
        Private method for updating the satisfied dependencies of the step function
        :param step_arg_name: the name of the argument being connected
        :return: a bool, True indicating that argument was not already satisfied and has now been marked as connected
        """
        if self._connected_inputs[step_arg_name]:
            return False

        else:
            self._connected_inputs[step_arg_name] = True
            return True

    def _disconnect_input(self, step_arg_name):
        """
        Marks an input as no longer connected
        :param step_arg_name: the name of the argument being connected
        :return:
        """
        self._connected_inputs[step_arg_name] = False

    def add_output(self, other_node, step_arg_name):
        """
        Connects this node's output to one of the input arguments of another node's step function
        :param other_node: the other node to be connected
        :param step_arg_name: the argument name of the other node's step function which this node will provide
        :param replace: if another node is already connected under that name, this will replace it if true
        :return: None
        """
        assert isinstance(other_node, ForwardBlockNode), "other_node must be another ForwardBlockNode object."

        assert step_arg_name in other_node.get_step_arg_names(), "'%s' is not an input argument of %s. " \
                                                                 "Input arguments are: %s" % \
                                                                 (step_arg_name, other_node.steppable_type(),
                                                                  other_node.get_step_arg_names())

        if other_node._connect_input(step_arg_name):
            self._output_nodes[other_node.get_name()] = (other_node, step_arg_name)

        else:
            raise ValueError("The arg_name %s of node %s is already the output of some other node." %
                             (step_arg_name, other_node.get_name()))

    def add_raw_output(self, other_steppable, block_name, output_name, step_arg_name):
        other_node = ForwardBlockNode(other_steppable, block_name, output_name)
        self.add_output(other_node, step_arg_name)
        return other_node

    def remove_output(self, node_name):
        node, step_arg_name = self._output_nodes.pop(node_name)
        node._disconnect_input(step_arg_name)

    def update_ready(self, input_name, input_val):
        self._inputs[input_name] = input_val
        return set(self._inputs) == set(self._step_arg_names)

    def step(self, save_dict=None):
        local_output = self._steppable.step(**self._inputs)
        self._inputs = {}

        if save_dict is not None and not self.is_sink():
            save_dict[self._output_name] = local_output

        if self.is_output():
            if self.is_sink():
                return {}
            else:
                return {self._output_name: local_output}

        else:
            output = {}

            for output_node, step_arg_name in self._output_nodes.values():
                if output_node.update_ready(step_arg_name, local_output):
                    output.update(output_node.step(save_dict))

            return output

    def is_locally_connected(self):
        if all(self._connected_inputs.values()):
            return True

        else:
            print('The following inputs of %s are not connected: %s' %
                  (self._name, set(self._step_arg_names) -
                   set(name for name, value in self._connected_inputs.items() if value)))
            return False

    def is_connected(self):
        return self.is_locally_connected() and all(output_node.is_connected()
                                                   for output_node, _ in self._output_nodes.values())

    def get_output_name(self):
        return self._output_name

    def is_output(self):
        return len(self._output_nodes) == 0

    def is_sink(self):
        return self._output_name is None

    def get_output_nodes(self):
        return self._output_nodes

    def __str__(self):
        return '[ForwardBlockNode "%s"]' % self._name

    def sink(self, block_name):
        return self.add_raw_output(PandaSink(), block_name, None, 'states')

    def steppable(self):
        return self._steppable


class ForwardRoot(Node):
    """
    A node which is the root of the graph, it contains all the source nodes of the systems (which have no input)
    """
    def __init__(self, source_nodes):
        self._source_nodes = source_nodes

    def step(self, save_dict=None):

        output = {}

        for source_node in self._source_nodes:
            output.update(source_node.step(save_dict))

        return output

    def is_connected(self):
        return all(source_node.is_connected() for source_node in self._source_nodes)

    def get_source_nodes(self):
        return self._source_nodes


class BackwardBlockNode(BlockNode):

    def __init__(self, steppable, name):
        super(BackwardBlockNode, self).__init__(steppable, name)
        self._input_nodes = {}

    def add_input(self, other_node, step_arg_name, replace=True):
        """
        Connects another node's output to one of the input arguments of it's step function
        :param other_node: the other node to be connected
        :param step_arg_name: the argument name of step which other node's output will be passed as
        :param replace: if another node is already connected under that name, this will replace it if true
        :return: None
        """
        assert step_arg_name not in self._input_nodes or replace, "input '%s' already exists and replace is set to False" % \
                                                                  step_arg_name

        assert step_arg_name in self._step_arg_names, "'%s' is not an input argument of %s. Input arguments are: %s" % \
                                                      (step_arg_name, type(self._steppable), self._step_arg_names)

        assert isinstance(other_node, BlockNode), "other_node must be another BlockNode object."

        self._input_nodes[step_arg_name] = other_node

    def add_raw_input(self, other_steppable, block_name, step_arg_name, replace=True):
        other_node = BackwardBlockNode(other_steppable, block_name)
        self.add_input(other_node, step_arg_name, replace)
        return other_node

    def is_locally_connected(self):
        """
        Checks if all the inputs of the step function have been connected to another node
        :return:
        """
        if set(self._input_nodes) == set(self._step_arg_names):
            return True
        else:
            print('The following inputs of %s are not connected: %s' %
                  (self._name, set(self._step_arg_names) - set(self._input_nodes)))
            return False

    def is_connected(self):
        """
        Checks if all inputs are satisfied recursively
        :return:
        """
        return self.is_locally_connected() and all(input_node.is_connected() for input_node in self._input_nodes.values())

    def step(self, save_dict=None):
        """
        Calls its Steppable's step function, sourcing the inputs recursively
        :param save_dict: a dictionary with which to save the inputs
        :return: the output of the step call
        """
        # store the input value as the output of the input node's step function
        inputs = {input_name: self._input_nodes[input_name].step(save_dict) for input_name in self._input_nodes}

        # if a save_dict was passed, update it with the inputs
        if save_dict is not None:
            save_dict.update(inputs)

        # return the result of this Steppable's step call
        return self._steppable.step(**inputs)

    def get_name(self):
        """Returns the name of the node"""
        return self._name

    def get_input_nodes(self):
        """Returns the dict of input nodes connected"""
        return self._input_nodes


class BackwardRoot(Node):
    """
    A node which is the root of the graph, it contains the single output node
    """
    def __init__(self, output_node, output_name):
        self._output_node = output_node
        self._output_name = output_name

    def step(self, save_dict=None):
        output_dict = {self._output_name: self._output_node.step()}
        save_dict.update(output_dict)
        return output_dict

    def is_connected(self):
        return self._output_node.is_connected()

    def get_output_name(self):
        return self._output_name

    def get_name(self):
        return self._output_node.get_name()

    def get_output_node(self):
        return self._output_node


class System(object):
    """
    A class which contains a graph of nodes
    """
    __metaclass__ = ABCMeta

    def __init__(self, root_node, framerate=None):
        """
        Constructor
        :param root_node: the last node in the system pipeline
        :param output_function: a function that will process the output
        """
        assert root_node.is_connected(), 'root_node is not fully connected'

        self.root_node = root_node
        self.history = pd.DataFrame()
        self.steps = 0
        self._all_edges = {}
        self.start_time = 0
        self.framerate = framerate
        self.get_time = self._get_real_time if framerate is None else self._get_sim_time
        self.pubs = {}

    def _get_sim_time(self):
        return 1.0* self.steps / self.framerate

    def _get_real_time(self):
        return time.time() - self.start_time

    def step(self, record):
        """
        Runs through one iteration of the system
        :param record: if True, records the edges in the history
        :return: None
        """
        edges = {}
        output = self.root_node.step(edges)
        if record:
            edges['time'] = {'time': self.get_time()}
            self._all_edges.update(edges)
            self.history = self.history.append([{'%s_%s'%(g, k):v for g, d in edges.items() for k,v in d.items()}], ignore_index=True)

    def clear(self):
        """Clears the history"""
        self._all_edges = {}
        self.history = pd.DataFrame()

    def run(self, record=True, print_steps=-1):
        """
        Runs the system
        :param record: If true, the system edges will be saved each step
        :return: a list of the system edges for each timestep
        """

        self.start_time = time.time()
        self.steps = 0
        while True:
            try:
                self.step(record)
                if print_steps >= 0 and self.steps % print_steps == 0:
                    print('%d: %f' % (self.steps, self.steps/self.framerate))
                self.steps += 1

            except (EOFError, KeyboardInterrupt, StopIteration):
                break

        groups = {group: self._all_edges[group].keys() for group in self._all_edges}
        return self.history, self.history.groupby(groups)

    def dynamic_publish(self, topic, message_type, message, parser):
        publisher = self.pubs.get(topic, rospy.Publisher(topic, message_type, queue_size=100))
        publisher.publish(message_type(parser(message)))

    def run_timed(self, rate, record=True, print_steps=-1):
        rate = rospy.Rate(rate)
        self.start_time = time.time()

        while True:
            try:
                self.step(record)
                if print_steps >= 0 and self.steps % print_steps == 0:
                    print('%d: %f' % (self.steps, self.steps/self.framerate))
                self.steps += 1

            except (EOFError, KeyboardInterrupt, StopIteration):
                break

            rate.sleep()

        groups = {group: self._all_edges[group].keys() for group in self._all_edges}
        return self.history, self.history.groupby(groups)


class ForwardSystem(System):

    def __init__(self, forward_root, output_function=None):
        assert isinstance(forward_root, ForwardRoot), 'forward_root must be a ForwardRoot object'

        super(ForwardSystem, self).__init__(forward_root, output_function)

    def draw(self, filename='block_diagram.gv'):
        """
        Draws a block diagram of the system using graphviz
        :param filename: the filename with which to save the output diagram
        :return: None
        """
        dot = Digraph("System Block Diagram", graph_attr={'rankdir': 'LR'}, strict=True)

        for source_node in self.root_node.get_source_nodes():
            dot.node(source_node.get_name(), shape='ellipse')
            self._draw_outputs(dot, source_node)

        dot.render(filename, view=True)

    def _draw_outputs(self, dot, block_node):
        for output_node, _ in block_node.get_output_nodes().values():
            dot.node(output_node.get_name(), shape='box')
            dot.edge(block_node.get_name(), output_node.get_name(), label=block_node.get_output_name())

            if output_node.is_output():
                if not output_node.is_sink():
                    dot.node(output_node.get_output_name(), shape='none')
                    dot.edge(output_node.get_name(), output_node.get_output_name())

            else:
                self._draw_outputs(dot, output_node)


class BackwardSystem(System):

    def __init__(self, backward_root, output_function=None):
        """
        Constructor
        :param output_node: the last node in the system pipeline
        :param output_function: a function that will process the output
        :param output_name: a name to assign the output of the last node
        """
        assert isinstance(backward_root, BackwardRoot), 'root_node must be a BackwardRoot object'
        super(BackwardSystem, self).__init__(backward_root, output_function)

    def draw(self, filename='block_diagram.gv'):
        """
        Draws a block diagram of the system using graphviz
        :param filename: the filename with which to save the output diagram
        :return: None
        """
        dot = Digraph("System Block Diagram", graph_attr={'rankdir': 'LR'})

        dot.node(self.root_node.get_output_name(), shape='none')
        shape = 'ellipse' if self.root_node.is_source() else 'box'
        dot.node(self.root_node.get_name(), shape=shape)
        dot.edge(self.root_node.get_name(), self.root_node.get_output_name())
        self._draw_inputs(dot, self.root_node.get_output_node())
        dot.render(filename, view=True)

    def _draw_inputs(self, dot, block_node):
        """
        Recursively draws block diagram from BlockNodes.
        Assumes that block_node has already been added to the dot graph under the name returned by block_node.get_name()
        :param Digraph dot:
        :param BackwardBlockNode block_node:
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
