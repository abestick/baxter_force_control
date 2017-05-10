import argparse
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class PlottableMocapData(object):

    def __init__(self, marker_data, time_data, start, goal):
        self.marker_data = marker_data
        self.time_data = time_data
        self.start = start
        self.goal = goal

    def __getitem__(self, tup):
        trials, markers, dimensions, time = unpad(tup, *[slice(None)]*3+[slice(-np.inf, np.inf)])
        plot_list = []

        if not isinstance(time, slice):
            raise TypeError("Time indexing must be via a slice!")

        if isinstance(trials, int):
            trials = slice(trials, trials+1)

        if isinstance(markers, int):
            markers = slice(markers, markers+1)

        if isinstance(dimensions, int):
            dimensions = slice(dimensions, dimensions+1)

        if isinstance(trials, (list, tuple, np.ndarray)):
            marker_trials = []
            time_trials = []

            for t in trials:
                marker_trials.append(self.marker_data[t])
                time_trials.append(self.time_data[t])

            trial_subset = zip([self.marker_data[t] for t in trials])

        else:
            trial_subset = zip(self.marker_data[trials], self.time_data[trials])

        for trial, times in trial_subset:
            t_indices = (times >= time.start) & (times <= time.stop)

            for trial_marker in trial[markers, :, :]:
                for trial_marker_dim in trial_marker[dimensions, :]:
                    plot_list += [times[t_indices], trial_marker_dim[t_indices]]

        return plot_list


class MocapProcessor(object):

    def __init__(self, input_data, marker_dict):
        self.start = input_data['start']
        self.goal = input_data['goal']

        # number of trials will be half of whats left after subtracting start, goal and marker nums
        self.trials = (len(input_data.keys()) - 5) / 2

        self.marker_pos = []
        self.marker_vel = []
        self.time_data = []

        for i in range(self.trials):
            self.marker_pos.append(input_data['full_sequence_%d' % i])
            self.time_data.append(input_data['time_%d' % i])
            self.marker_vel.append(np.diff(self.marker_pos[-1], axis=2) / np.diff(self.time_data[-1]))

        self.marker_dict = marker_dict
        self.groups = {}
        self.plottable_pos = PlottableMocapData(self.marker_pos, self.time_data, self.start, self.goal)
        self.plottable_vel = PlottableMocapData(self.marker_vel, [time[1:] for time in self.time_data], self.start,
                                                self.goal)

    def clone(self):
        return deepcopy(self)

    def _parse_markers(self, markers):
        marker_list = []

        try:
            for marker in markers:
                try:
                    marker_list.append(self.marker_dict[marker])

                except KeyError:
                    if all([isinstance(i, int) for i in markers]):
                        return markers

                    else:
                        raise ValueError("markers must be a slice, int, key or an iterable of keys or ints")

                else:
                    return marker_list

        except TypeError:
            try:
                return self.marker_dict[markers]

            except KeyError:
                if isinstance(markers, (int, slice)):
                    return markers

                else:
                    raise ValueError("markers must be a slice, int, key or an iterable of keys or ints")

    def __iter__(self):
        for marker_pos, marker_vel, time_data in zip(self.marker_pos, self.marker_vel, self.time_data):
            yield marker_pos, marker_vel, time_data


    def assign_group(self, group_name, markers):
        self.groups[group_name] = self.groups.get(group_name, set()) | set(markers)

    def get_group(self, group_name):
        return self.groups[group_name]

    def plot_pos(self, ax, trials=slice(None), markers=None, start_time=None, stop_time=None):
        markers = self._parse_markers(markers)
        time = slice(start_time, stop_time)
        ax.plot(*self.plottable_pos[trials, markers, :, time])

    def plot_vel(self, ax, trials=slice(None), markers=None, start_time=None, stop_time=None):
        markers = self._parse_markers(markers)
        time = slice(start_time, stop_time)
        ax.plot(*self.plottable_pos[trials, markers, :, time])

    def plot_pos_vel(self, ax_pos, ax_vel, markers):
        self.plot_pos(ax_pos, markers)
        self.plot_vel(ax_vel, markers)

    def velocity_crop(self, max_vel, trials=slice(None), axis=None):
        cropped = self.clone()
        for i, (marker_pos, marker_vel, time) in enumerate(cropped):
            start_index = np.argmax(np.all(marker_vel > max_vel, axis=[0, 1]))
            stop_index = np.argmax(np.all(marker_vel > max_vel, axis=[0, 1])[::-1])
            cropped.marker_pos[i] = marker_pos[:,:,start_index:stop_index]
            cropped.marker_vel[i] = marker_vel[:,:,start_index:stop_index]
            cropped.time_data[i] = time[start_index:stop_index]

        cropped.plottable_pos = PlottableMocapData(cropped.marker_pos, cropped.time_data, cropped.start, cropped.goal)
        cropped.plottable_vel = PlottableMocapData(cropped.marker_vel, [time[1:] for time in cropped.time_data], cropped.start,
                                                cropped.goal)

        return cropped

def unpad(tup, *args):
    return tup + args[len(tup):]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_data_npz', help='The input npz data')
    parser.add_argument('output_data_npz', help='The output npz data')
    args = parser.parse_args()

    raw_data = np.load(args.input_data_npz)



    with open(args.output_npz, 'w') as output_file:
        np.savez_compressed(output_file)
        print('Task sequences saved to ' + args.output_data_npz)