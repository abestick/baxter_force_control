from definitions import BaxterDefs as bd
from trajectory_msgs.msg import JointTrajectory, MultiDOFJointTrajectory
from baxter_core_msgs.msg import EndpointState
from conversions import trajectory_to_arrays, arrays_to_joint_trajectory_msg, cartesian_to_array, \
    arrays_to_multi_dof_trajectory_msg, pose_to_array, quaternion_to_array
import numpy as np
import rospy


def saturate_trajectory_pos(joint_trajectory, uniform=False):
    """

    :param JointTrajectory joint_trajectory:
    :param bool uniform:
    :return:
    """
    positions, time = trajectory_to_arrays(joint_trajectory)
    new_positions = positions.copy()
    min_scale = 1

    for i, pos in enumerate(positions):
        # normalize the joint array so that now they must be less than or equal to 1
        abs_normalized_pos = np.abs((pos - bd.arm.neutrals) / (0.5 * bd.arm.ranges))

        if np.any(abs_normalized_pos > 1):
            scale_factor = 1.0 / np.max(abs_normalized_pos)
            min_scale = min(min_scale, scale_factor)
            new_positions[i, :] = scale_factor * pos + (1 - scale_factor) * bd.arm.neutrals

    if uniform:
        new_positions = min_scale * positions + (1 - min_scale) * bd.arm.neutrals

    return arrays_to_joint_trajectory_msg(new_positions, time, names=joint_trajectory.joint_names)


def saturate_trajectory_vel(joint_trajectory, uniform=False):
    """

    :param JointTrajectory joint_trajectory:
    :param bool uniform:
    :return:
    """
    positions, time = trajectory_to_arrays(joint_trajectory)
    time_deltas = np.diff(time)
    new_deltas = time_deltas.copy()

    for i, time_delta in enumerate(time_deltas):
        diff = positions[i+1] - positions[i]
        if sum(diff) == 0:
            continue
        if time_delta == 0:
            time_delta = 0.0001
        overshoot = np.abs(diff / time_delta) - bd.arm.vel_limits
        if np.any(overshoot > 0):
            new_deltas[i] = np.max(diff) / bd.arm.vel_limits[np.argmax(diff)]

    if uniform:
        max_scale = np.max(new_deltas / time_deltas)
        new_deltas = time_deltas * max_scale

    new_time = np.insert(np.cumsum(new_deltas)+time[0], 0, time[0])

    return arrays_to_joint_trajectory_msg(positions, new_time, names=joint_trajectory.joint_names)


def get_normal_to_points(p1, p2, p3):
    p1, p2, p3 = map(np.array, [p1, p2, p3])
    return np.cross(p2 - p1, p3 - p1)


class Segment(object):
    
    def draw(self, rate):
        pass


class Circle(Segment):

    def __init__(self, start_point, far_point, normal, period, travel=2*np.pi):
        self.start_point = start_point
        self.far_point = far_point
        self.normal = np.array(normal)
        self.period = period
        self.travel = travel

    def draw(self, rate, start_time, hold_orientation=True):
        
        start = cartesian_to_array(self.start_point.position)
        center = (cartesian_to_array(self.far_point.position) + start) / 2
        center = center.reshape((1, 3))
        a = start - center
        b = np.cross(self.normal, a)
        b = b * np.linalg.norm(a) / np.linalg.norm(b)
        total_time = self.period * self.travel / (2.0*np.pi)
        time = np.arange(start_time, start_time + total_time, 1.0 / rate)
        theta = np.linspace(0, self.travel, num=len(time)).reshape((-1, 1))
        pos = center + a*np.cos(theta) + b*np.sin(theta)

        if hold_orientation:
            quat = np.ones((pos.shape[0], 4)) * quaternion_to_array(self.start_point.orientation)

        return arrays_to_multi_dof_trajectory_msg(np.hstack((pos, quat)), time)


class Polygon(Segment):

    def __init__(self, points, period, repeats=1):
        self.points = points
        self.period = period
        self.start_point = points[0]
        self.repeats = repeats

    def draw(self, rate, start_time, hold_orientation=True):
        points = np.array(map(pose_to_array, self.points))
        pos_points = np.vstack((points[:, :3], points[0, :3]))
        displacements = np.diff(pos_points, axis=0)     
        distances = np.linalg.norm(displacements, axis=1)
        total_steps = int(self.period*rate)
        proportions = distances / np.sum(distances)
        steps = np.around(proportions * total_steps).astype(int)
        error = total_steps - np.sum(steps)
        steps[0] += error
        pos = np.empty((0, 3))

        for pos_point, displacement, step_count in zip(pos_points, displacements, steps):
            travelled = np.linspace(0, 1.0, num=step_count).reshape(-1, 1)
            section = displacement.reshape((1, 3))*travelled + pos_point.reshape((1, 3)) 
            pos = np.vstack((pos, section))

        pos = np.tile(pos, (self.repeats, 1))
        time = np.linspace(start_time, start_time + self.period*self.repeats, num=pos.shape[0])

        if hold_orientation:
            quat = np.ones((pos.shape[0], 4)) * points[0, 3:]

        return arrays_to_multi_dof_trajectory_msg(np.hstack((pos, quat)), time)



class Jab(Segment):

    def __init__(self, start_point, end_point, period, repeats=1):
        self.start_point = start_point
        self.end_point = end_point
        self.period = period
        self.repeats = repeats


    def draw(self, rate, start_time, hold_orientation=True):
        start_pos = cartesian_to_array(self.start_point.position)
        displacement = cartesian_to_array(self.end_point.position) - start_pos
        half_step_count = int(self.period * rate / 2)
        travelled = np.linspace(0, 1.0, num=half_step_count)
        out_jab = displacement.reshape((1, 3))*travelled.reshape((-1, 1)) + start_pos.reshape((1, 3))
        in_jab = np.flipud(out_jab)
        pos = np.vstack((out_jab, in_jab))
        pos = np.tile(pos, (self.repeats, 1))
        time = np.linspace(start_time, start_time + self.period*self.repeats, num=pos.shape[0])

        if hold_orientation:
            quat = np.ones((pos.shape[0], 4)) * quaternion_to_array(self.start_point.orientation).reshape((1, 4))

        return arrays_to_multi_dof_trajectory_msg(np.hstack((pos, quat)), time)


class Waypoint(Segment):

    def __init__(self, point):
        self.start_point = point

    def draw(self, rate, start_time, hold_orientation=True):
        return arrays_to_multi_dof_trajectory_msg(pose_to_array(self.start_point).reshape((1, 7)), np.array([start_time]))


class EETrajectoryRecorder(object):

    def __init__(self, limb, play_trajectory=None):

        self.segment_map = {'w': self.add_waypoint,
                            'j': self.add_jab,
                            'p': self.add_polygon,
                            'c': self.add_circle,
                            'l': self.play_last}

        self.limb = limb
        self.segments = []
        self.transition_times = []
        self.ee_pose = None
        self.sub = rospy.Subscriber('/robot/limb/%s/endpoint_state' % limb, EndpointState, self.get_pose)
        self.play_trajectory = play_trajectory

    def get_pose(self, msg):
        self.ee_pose = msg.pose

    def ask_period(self):
        return float(raw_input('Enter period for recorded segment:\n'))

    def ask_repeats(self):
        return int(raw_input('Enter repetitions to perform:\n'))

    def ask_transition(self):
        return float(raw_input('Enter transition time from last segment to this one:\n'))

    def add_waypoint(self):
        raw_input('Press ENTER to record waypoint')
        transition = self.ask_transition()
        self.transition_times.append(transition)
        self.segments.append(Waypoint(self.ee_pose))

    def add_jab(self):
        raw_input('Press ENTER to record start point')
        start_point = self.ee_pose
        transition = self.ask_transition()

        raw_input('Press ENTER to record end point')
        end_point = self.ee_pose
        
        period = self.ask_period()
        repeats = self.ask_repeats()
        self.transition_times.append(transition)
        self.segments.append(Jab(start_point, end_point, period, repeats))

    def add_polygon(self):
        points = []
        while raw_input('Press ENTER to record a new point, any key to end') == '':
            points.append(self.ee_pose)

        transition = self.ask_transition()

        period = self.ask_period()
        repeats = self.ask_repeats()

        self.transition_times.append(transition)
        self.segments.append(Polygon(points, period, repeats))

    def add_circle(self):
        raw_input('Press ENTER to record start point')
        start_point = self.ee_pose
        transition = self.ask_transition()
        raw_input('Press ENTER to record far point')
        far_point = self.ee_pose

        resp = raw_input('Press Enter to record an intermediate point on the circle '
            'or +- x, y or z to give the normal:\n')
        if resp == '':
            p1, p2, p3 = map(cartesian_to_array, [start_point.position, self.ee_pose.position, far_point.position])
            normal = get_normal_to_points(p1, p2, p3)
            
        else:
            sign = -1 if '-' in resp else 1
            resp = resp[-1]
            normal = [sign*int(v == resp) for v in 'xyz']
            setattr(far_point.position, resp, getattr(start_point.position, resp))
        
        period = self.ask_period()
        travel = np.deg2rad(float(raw_input('Enter the travel in degrees for the circle:\n')))
        self.transition_times.append(transition)
        self.segments.append(Circle(start_point, far_point, normal, period, travel))

    def draw(self, rate, hold_orientation=True):
        points = []
        last_time = 0
        for segment, transition in zip(self.segments, self.transition_times):
            points.extend(segment.draw(rate, last_time + transition, hold_orientation).points)
            last_time = points[-1].time_from_start.to_sec()

        return MultiDOFJointTrajectory(points=points)

    def play_last(self, rate=20):
        self.play_trajectory(self.segments[-1].draw(rate, max(self.transition_times[-1], 3)), self.limb)

    def new_segment(self):
        resp = raw_input('Press ENTER to quit or choose one of the following options:\n'
            'w: new waypoint\n'
            'j: new jab\n'
            'p: new polygon\n'
            'c: new circle\n'
            'l: play last segment\n\n')

        if resp in self.segment_map:
            self.segment_map[resp]()
            return True

        else:
            return False

    def record(self):
        while self.new_segment():
            pass
        