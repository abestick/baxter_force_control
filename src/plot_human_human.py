#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt



FRAMERATE = 50
GROUP_NAME = 'tree'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('system_history', help='The system input npz file')
    args = parser.parse_args()

    # Load the calibration sequence
    data = np.load(args.system_history)
    group = raw_input('Choose a state group to plot:\n%s\nGroup (<ENTER> to quit):' % data.keys())

    while group != '':
        traj_dicts = data[group]
        traj_keys = traj_dicts[-1].keys()
        trajectories = zip(*[[float(d[key]) for key in traj_keys] for d in traj_dicts if set(d)==set(traj_keys)])

        plot_each(trajectories, traj_keys)
        plt.show()
        group = raw_input('Choose a state group to plot:\n%s\nGroup (<ENTER> to quit):' % data.keys())


def plot_each(traj, keys):
    for element, key in zip(traj, keys):
        plt.plot(element, label=key)
    plt.legend()


if __name__ == '__main__':
    # cProfile.run('main()', 'fit_kinmodel.profile')
    main()