#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose
from std_msgs.msg import Header
from kinmodel import Transform



FRAMERATE = 50
GROUP_NAME = 'tree'


def main():
    ros_init = False
    parser = argparse.ArgumentParser()
    parser.add_argument('system_history', help='The system input npz file')
    args = parser.parse_args()

    # Load the calibration sequence
    data = np.load(args.system_history)
    group = raw_input('Choose a state group to plot:\n%s\nGroup (<ENTER> to quit):' % data.keys())

    while group != '':
        traj_dicts = data[group]
        trial_num = group.split('_')[-1]
        time = np.array(data['time_'+str(trial_num)])

        if isinstance(traj_dicts[-1].values()[0], Transform):
            transform_traj = [traj_dict.values()[0] for traj_dict in traj_dicts]
            transform_name = traj_dicts[-1].keys()[0]
            rate = 1/np.mean(np.diff(time)) * 10
            if not ros_init:
                rospy.init_node('learn_human_human')

            stream_pose(transform_traj, transform_name, rate)

        else:
            traj_keys = traj_dicts[-1].keys()
            trajectories = zip(*[[float(d[key]) for key in traj_keys] for d in traj_dicts if set(d)==set(traj_keys)])
            if len(trajectories[0]) == 1:
                print(trajectories)

            else:
                plot_each(trajectories, traj_keys)
                plt.show()

        group = raw_input('Choose a state group to plot:\n%s\nGroup (<ENTER> to quit):' % data.keys())


def plot_each(traj, keys):
    for element, key in zip(traj, keys):
        plt.plot(element, label=key)
    plt.legend()


def stream_pose(transform_trajectory, name, rate):
    pub = rospy.Publisher(name, PoseStamped, queue_size=100)
    ros_rate = rospy.Rate(rate)
    pub.publish(PoseStamped(header=Header(stamp=rospy.Time.now(), frame_id='map')))
    raw_input('Press <ENTER> to begin streaming.')

    for transform in transform_trajectory:
        pose = transform.pose(convention='quaternion')
        header = Header(stamp=rospy.Time.now(), frame_id='map')
        msg = PoseStamped(header=header, pose=Pose(position=Point(*pose[:3]), orientation=Quaternion(*pose[3:])))
        pub.publish(msg)
        ros_rate.sleep()


if __name__ == '__main__':
    # cProfile.run('main()', 'fit_kinmodel.profile')
    main()