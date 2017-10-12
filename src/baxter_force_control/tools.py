from kinmodel.track_mocap import KinematicTreeTracker, KinematicTreeExternalFrameTracker, MocapFrameTracker
from steppables import MocapFrameEstimator, MocapSystemState, TFPublisher, JointPublisher, \
    TFBagger, Differentiator, TwistPublisher, WrenchPublisher, Modifier, Transformer, transform_msg_to_transform, \
    twist_msg_to_twist, PointPublisher, ExponentialFilter
from system import ForwardBlockNode
import numpy as np


def get_frame_tracker_node(frame_tracker, frame_name, joint_name, publish=True, get_time=None,
                           **presets):
    frame_indices, frame_points = frame_tracker.attach_frame(joint_name, frame_name, tf_pub=False, **presets)
    mocap_frame_tracker = MocapFrameTracker(frame_name, frame_indices, frame_points)
    frame_estimator = MocapFrameEstimator(mocap_frame_tracker, frame_name)
    nice_name = frame_name.replace('_', ' ').title()
    frame_node = ForwardBlockNode(frame_estimator, '%s Frame Estimator' % nice_name, frame_name)
    if publish:
        frame_node.add_raw_output(TFPublisher(get_time), '%s Frame Publisher' % nice_name, None,
                                  'transform')
    return frame_node


def relative_transform_node(world_parent_node, world_child_node, publish=True, get_time=None):
    child_frame = world_child_node.get_output_name()
    parent_frame = world_parent_node.get_output_name()
    nice_child_name = child_frame.replace('_', ' ').title()
    nice_parent_name = parent_frame.replace('_', ' ').title()

    transformer = Transformer(inv=True)
    transformer_node = ForwardBlockNode(transformer, '%s %s Frame Transformer' % (nice_parent_name, nice_child_name),
                                        '%s_%s' % (parent_frame, child_frame))
    world_parent_node.add_output(transformer_node, 'transform')
    world_child_node.add_output(transformer_node, 'primitives')
    if publish:
        transformer_node.add_raw_output(TFPublisher(get_time), '%s %s Frame Publisher'
                                        % (nice_parent_name, nice_child_name), None, 'transform')
    return transformer_node


def get_system_state_node(kin_trees, states_name, publish=True, bag=None, get_time=None):
    kin_tree_trackers = [KinematicTreeTracker('tree_tracker', kin_tree, scalar_states=True) for kin_tree in kin_trees]
    system_state = MocapSystemState(kin_tree_trackers)
    system_state_node = ForwardBlockNode(system_state, 'System State Estimator', 'raw_' + states_name)
    exp_filter = ExponentialFilter(alpha=0.15)
    filter_node = ForwardBlockNode(exp_filter, 'State Filter', states_name)
    system_state_node.add_output(filter_node, 'states')
    if publish:
        system_state_node.add_raw_output(JointPublisher('raw_' + states_name, bag, get_time),
                                         'System State Publisher', None, 'states')
        filter_node.add_raw_output(JointPublisher(states_name, bag, get_time),
                                         'Filtered State Publisher', None, 'states')
    return system_state_node, filter_node


def differentiate_node(target_node, fixed_step, publish=True, bag=None, get_time=None, as_wrench=False, filter=False):

    output_name = 'd_' + target_node.get_output_name()
    node_name = ' '.join(target_node.get_name().split(' ')[:-2]) + ' Differentiator'
    differentiator = Differentiator(fixed_step=fixed_step)

    prefix = 'raw_' if filter else ''
    differentiator_node = ForwardBlockNode(differentiator, node_name, prefix+output_name)

    if filter:
        exp_filter = ExponentialFilter(alpha=0.1)
        filter_node = ForwardBlockNode(exp_filter, 'State Filter', output_name)
        differentiator_node.add_output(filter_node, 'states')

    if publish:
        topic_name = target_node.get_output_name() + '_twist'
        publisher_name = ' '.join(target_node.get_name().split(' ')[:-2]) + ' Twist Publisher'
        publisher_type = WrenchPublisher if as_wrench else TwistPublisher
        differentiator_node.add_raw_output(publisher_type(prefix+topic_name, bag, get_time), publisher_name,
                                           None, 'states')
        if filter:
            filter_node.add_raw_output(publisher_type(topic_name, bag, get_time), publisher_name,
                                           None, 'states')

    target_node.add_output(differentiator_node, 'states')

    return differentiator_node


def differentiate_node_for_visualization(target_node, fixed_step, bag=None, get_time=None, parent_frame='world'):
    pos_node_name = 'Pos Only ' + target_node.get_name()
    output_name = 'pos_only_' + target_node.get_output_name()
    publisher_name = 'Pos Only ' + ' '.join(target_node.get_name().split(' ')[:-1]) + ' Publisher'
    modifier_node = ForwardBlockNode(Modifier(pos_only_transform), pos_node_name, output_name)
    target_node.add_output(modifier_node, 'states')
    modifier_node.add_raw_output(TFPublisher(get_time), publisher_name, None, 'transform')
    return differentiate_node(target_node, fixed_step, True, bag, get_time, as_wrench=True), modifier_node


def transform_node(target_node, transform_node, publish=True, bag=None, get_time=None, publisher_type=None,
                   all_frames=True):
    transformer = Transformer()
    parent_frame = transform_node.get_output_name()
    out_name = target_node.get_output_name()
    nice_parent_name = parent_frame.replace('_', ' ').title()
    nice_out_name = out_name.replace('_', ' ').title()

    transformer_node = ForwardBlockNode(transformer, '%s %s Transformer' % (nice_parent_name, nice_out_name),
                                        '%s_%s' % (parent_frame, out_name))

    transform_node.add_output(transformer_node, 'transform')
    target_node.add_output(transformer_node, 'primitives')
    if publish:
        if publisher_type is None:
            steppable_type = target_node.steppable_type()
            if steppable_type == MocapFrameEstimator:
                publisher_type = TFPublisher
            else:
                raise NotImplementedError()

        if publisher_type == TFPublisher:
            transformer_node.add_raw_output(TFPublisher(get_time),
                                            '%s %s Frame Publisher' % (nice_parent_name, nice_out_name),
                                            None, 'transform')

        else:
            topic_name = transformer_node.get_output_name()
            transformer_node.add_raw_output(publisher_type(topic_name, bag, get_time, all_frames),
                                            '%s %s Publisher' % (nice_parent_name, nice_out_name), None, 'states')

    return transformer_node


def pos_only_transform(transform):
    return transform.trans_only()


def grip_point(points):
    distances = [None] * len(points)
    midpoints = [None] * len(points)
    for i in range(len(points)):
        other_points = points[:]
        other_points.pop(i)
        diff = np.diff(other_points, axis=0).squeeze()
        midpoints[i] = np.mean(other_points, axis=0).squeeze()
        distances[i] = np.linalg.norm(diff)

    return midpoints[np.argmax(distances)]


def get_twist_from_msgs(msgs, topic):
    twist_msg = msgs[topic]

    return twist_msg_to_twist(twist_msg)


def get_joint_states_from_msgs(msgs, topic, names):
    joint_msg = msgs[topic]
    return [joint_msg.position[joint_msg.name.index(name)] for name in names]


def get_transform_from_msgs(msgs, parent, child):
    tf_msg = msgs['tf']

    transform_msg = next(transform for transform in tf_msg.transforms
                         if transform.header.frame_id == parent and transform.child_frame_id == child)

    return transform_msg_to_transform(transform_msg)


def get_transform_from_bag_reader(t, bag_reader, parent, child):
    msgs = bag_reader[t]
    return get_transform_from_msgs(bag_reader, parent, child)


def vector_to_point_pub(vector_node, topic_name, reference_frame):
    name = vector_node.get_name() + ' Publisher'
    vector_node.add_raw_output(PointPublisher(topic_name, reference_frame), name, None, 'states')