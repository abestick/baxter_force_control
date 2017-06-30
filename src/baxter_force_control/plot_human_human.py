#!/usr/bin/env python
import argparse
import matplotlib.pyplot as plt
import pickle
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose
from std_msgs.msg import Header
from pandas_tools import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('system_history', help='The system history npz file')
    args = parser.parse_args()
    sys_hist_split = args.system_history.split('.')

    if sys_hist_split[-1] == 'pkl':
        pkl_file = open(args.system_history, 'rb')
        trials = pickle.load(pkl_file)

    else:
        raise ValueError('system_history must be a pkl. You passed %s' % args.system_history)

    choice = ''
    while choice != 'q':

        trial_name = raw_input('Choose trial\n%s\n' % trials.keys())
        trial, grouped = trials[trial_name]

        subset_str = raw_input('Choose elements, sperated by a /\n%s\n' % list(trial.columns))
        subsets = subset_str.split('/')
        selected_data = trial[subsets]
        plotting_data = one_d_columns(selected_data)
        plotting_data.plot()
        plt.show()

        choice = raw_input('Quit = q, New plot = n, Subset plot = s, Orthoganal test = o')

        while choice == 's' or choice == 'o':

            if choice == 'o':
                pairs_str = raw_input('Select pairs in format a1,a2/b1,b2/c1,c2 \n%s\n' % list(selected_data.columns))
                pairs = [pair.split(',') for pair in pairs_str.split('/')]
                dots = dot_products(selected_data, pairs)
                dots.plot()
                plt.show()

            else:
                subset_str = raw_input('Choose elements, sperated by a /\n%s\n' % list(selected_data.columns))
                subsets = subset_str.split('/')
                subsetted_data = one_d_columns(trial[subsets])
                subsetted_data.plot()
                plt.show()

            choice = raw_input('Quit = q, New plot = n, Subset plot = s, Orthoganal test = o')


def plot_each(trajectories, keys):

    for trajectory, key in zip(trajectories, keys):
        plt.plot(trajectory, label=key)
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


def int_string(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    main()