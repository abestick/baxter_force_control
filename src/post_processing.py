#!/usr/bin/env python
import rospy
import argparse
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import OrderedDict
from scipy.signal import medfilt


class PlottableMocapData(object):

    def __init__(self, marker_data, time_data, start, goal):
        self.marker_data = marker_data
        self.time_data = time_data
        self.start = start
        self.goal = goal

    def __getitem__(self, tup):
        trials, markers, dimensions, time = pad_tuple(tup, *[slice(None)] * 3 + [slice(-np.inf, np.inf)])
        plot_list = []

        if not isinstance(time, slice):
            raise TypeError("Time indexing must be a slice!")

        if time.start is None:
            time = slice(-np.inf, time.stop)

        if time.stop is None:
            time = slice(time.stop, np.inf)

        if isinstance(markers, int):
            markers = slice(markers, markers+1)

        if isinstance(dimensions, int):
            dimensions = slice(dimensions, dimensions+1)

        if isinstance(trials, int):
            trials = [trials]

        elif isinstance(trials, slice):
            trials = range(*trials.indices(len(self.time_data)))

        elif not isinstance(trials, (list, tuple, np.ndarray)):
            raise TypeError("trials must be a  slice, int or iterable of ints")
        for trial in trials:
            trial_data = self.marker_data[trial]
            time_data = self.time_data[trial]
            t_indices = (time_data >= time.start) & (time_data <= time.stop)
            for trial_marker_data in trial_data[markers, :, :]:
                for trial_marker_dim_data in trial_marker_data[dimensions, :]:
                    plot_list += [time_data[t_indices], trial_marker_dim_data[t_indices]]

        return plot_list


class MocapProcessor(object):

    def __init__(self, input_data, marker_dict):
        self.start = input_data['start']
        self.goal = input_data['goal']

        # number of trials will be half of whats left after subtracting start, goal and marker nums
        self.trials = (len(input_data.keys()) - 3) / 2

        self.marker_pos = OrderedDict()
        self.marker_vel = OrderedDict()
        self.time_data = OrderedDict()

        for trial in range(self.trials):
            self.marker_pos[trial] = input_data['full_sequence_%d' % trial]
            self.time_data[trial] = input_data['time_%d' % trial]
            self.marker_vel[trial] = np.diff(self.marker_pos[trial], axis=2) / np.diff(self.time_data[trial])

        self.marker_dict = marker_dict
        self.groups = {}
        self.plottable_pos = None
        self.plottable_vel = None

        self._setup_plottables()

    def _setup_plottables(self):
        self.plottable_pos = PlottableMocapData(self.marker_pos, self.time_data, self.start, self.goal)
        self.plottable_vel = PlottableMocapData(self.marker_vel, [time[1:] for time in self.time_data.values()], self.start,
                                                self.goal)

    def _recalculate_velocity(self):
        for trial, pos in self.marker_pos.items():
            self.marker_vel[trial] = np.diff(pos, axis=2) / np.diff(self.time_data[trial])

    def clone(self):
        return deepcopy(self)

    @staticmethod
    def _parse_component(component):

        if component == '' or component == 'both':
            pos, vel = True, True

        else:
            pos = 'pos' in component
            vel = 'vel' in component
            assert pos != vel, "Invalid component string! Must be pos(ition), vel(ocity) both (or empty)"

        return pos, vel

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

            except (KeyError, TypeError):
                if isinstance(markers, (int, slice)):
                    return markers

                else:
                    raise ValueError("markers must be a slice, int, key or an iterable of keys or ints")

    def _parse_trials(self, trials):

        if isinstance(trials, int):
            trials = [trials]

        elif isinstance(trials, slice):
            trials = range(*trials.indices(self.trials))

        elif not isinstance(trials, (list, tuple, np.ndarray)):
            raise TypeError("trials must be a  slice, int or iterable of ints")

        return trials

    def _parse_time(self, trial, start_time, stop_time):
        return (self.time_data[trial] >= start_time) & (self.time_data[trial] <= stop_time)

    def __iter__(self):
        for trial in self.marker_pos.keys():
            yield trial, self.marker_pos[trial], self.marker_vel[trial], self.time_data[trial]

    def __getitem__(self, tup):
        trials, markers, dimensions, time = pad_tuple(tup, *[slice(None)] * 3 + [slice(-np.inf, np.inf)])
        new_one =  self.clone()
        if not isinstance(time, slice):
            raise TypeError("Time indexing must be via a slice!")

        if isinstance(markers, int):
            markers = slice(markers, markers+1)

        if isinstance(dimensions, int):
            dimensions = slice(dimensions, dimensions+1)

        trials = self._parse_trials(trials)

        new_one.marker_pos = OrderedDict()
        new_one.marker_vel = OrderedDict()
        new_one.time_data = OrderedDict()

        for trial in trials:
            t_indices = self._parse_time(trial, time.start, time.stop)
            new_one.marker_pos[trial] = self.marker_pos[trial][markers, dimensions, t_indices]
            new_one.marker_vel[trial] = self.marker_vel[trial][markers, dimensions, t_indices]
            new_one.time_data[trial] = self.time_data[trial][t_indices]

        new_one.trials = len(trials)
        new_one._setup_plottables()
        return new_one

    def assign_group(self, group_name, markers):
        self.groups[group_name] = self.groups.get(group_name, set()) | set(markers)

    def get_group(self, group_name):
        return self.groups[group_name]

    def plot_pos(self, ax, trials=slice(None), markers=slice(None), start_time=None, stop_time=None):
        markers = self._parse_markers(markers)
        time = slice(start_time, stop_time)
        ax.plot(*self.plottable_pos[trials, markers, :, time])

    def plot_vel(self, ax, trials=slice(None), markers=slice(None), start_time=None, stop_time=None):
        markers = self._parse_markers(markers)
        time = slice(start_time, stop_time)
        ax.plot(*self.plottable_vel[trials, markers, :, time])

    def plot_pos_vel(self, ax_pos, ax_vel, markers):
        self.plot_pos(ax_pos, markers)
        self.plot_vel(ax_vel, markers)

    def time_slice(self, trial, start_time, stop_time):
        sliced = self.clone()
        time = self._parse_time(trial, start_time, stop_time)
        sliced.marker_pos[trial] = sliced.marker_pos[trial][:, :, time]
        sliced.marker_vel[trial] = sliced.marker_vel[trial][:, :, time[1:]]
        sliced.time_data[trial] = sliced.time_data[trial][time]
        sliced._setup_plottables()
        return sliced

    def velocity_crop(self, max_vel, trials=slice(None), axis=None):
        trials = self._parse_trials(trials)

        cropped = self.clone()
        for trial in trials:
            start_index = np.argmax(np.any(abs(np.nan_to_num(cropped.marker_vel[trial])) > max_vel, axis=(0, 1)))
            stop_index = -np.argmax(np.any(abs(np.nan_to_num(cropped.marker_vel[trial])) > max_vel, axis=(0, 1))[::-1])
            if stop_index == 0:
                stop_index = None
            print(start_index, stop_index)
            cropped.marker_pos[trial] = cropped.marker_pos[trial][:, :, start_index:stop_index]
            cropped.marker_vel[trial] = cropped.marker_vel[trial][:, :, start_index:stop_index]
            cropped.time_data[trial] = cropped.time_data[trial][start_index:stop_index]

        cropped._setup_plottables()
        if axis is not None:
            axis.plot(*cropped.plottable_vel[trials])

        return cropped

    def mutate(self, trials, func, args=None, kwargs=None, indexing=slice(None), component=''):
        pos, vel = self._parse_component(component)

        if kwargs is None:
            kwargs = {}

        trials = self._parse_trials(trials)

        for trial in trials:
            if pos:
                self.marker_pos[trial][indexing] = func(self.marker_pos[trial][indexing], *args, **kwargs)

            if vel:
                self.marker_vel[trial][indexing] = func(self.marker_vel[trial][indexing], *args, **kwargs)

    def medfilt(self, trials=slice(None), kernel_size=None, axis=None, component=''):
        trials = self._parse_trials(trials)
        pos, vel = self._parse_component(component)
        filtered = self.clone()

        for trial in trials:
            markers, dims, _ = filtered.marker_pos[trial].shape
            for marker in range(markers):
                for dim in range(dims):
                    indexing = (marker, dim)
                    if pos:
                        filtered.mutate(trial, medfilt, args=(kernel_size,), indexing=indexing, component='pos')
                        filtered._recalculate_velocity()

                    if vel:
                        filtered.mutate(trial, medfilt, args=(kernel_size,), indexing=indexing, component='vel')

        if axis is not None:
            if pos:
                filtered.plot_pos(axis, trials)

            filtered.plot_vel(axis, trials)
        return filtered

    def get_npz_data(self, select=None, exclude=None):

        if select is not None and exclude is not None:
            raise ValueError("You cannot pass select and exclude at the same time!")

        if select is None:
            select = self.marker_pos.keys()

        if exclude is None:
            exclude = []

        save_data = {'goal': self.goal, 'start': self.start}
        base = 'full_sequence_'

        for trial, pos, vel, time in self:
            if trial in select and trial not in exclude:
                save_data[base + str(trial)] = pos.copy()
                save_data['vel_' + str(trial)] = vel.copy()
                save_data['time_' + str(trial)] = time.copy()

        return save_data


def pad_tuple(tup, *args):
    try:
        tup = tuple(tup)
    except TypeError:
        tup = (tup, )
    return tup + args[len(tup):]


def show_full_screen():
    pass


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input_data_npz', help='The input npz data')
    parser.add_argument('output_data_npz', help='The output npz data')
    args = parser.parse_args()

    raw_data = np.load(args.input_data_npz)
    processor = MocapProcessor(raw_data, {})
    cropped = processor.medfilt(kernel_size=5)
    cropped_tmp = cropped.clone()
    exclude = []
    max_vel = 0

    for trial, _, _, _ in processor:
        print("Trial %d/%d" % (trial, processor.trials))
        f, (ax_p, ax_v) = plt.subplots(2, sharex=True)
        cropped.plot_vel(ax_v, trial)
        cropped.plot_pos(ax_p, trial)
        plt.show()
        resp = raw_input("Velocity cutoff? [a to accept, m to manually enter times, "
                         "<ENTER> to use %f again, d to discard this trial]:\n" % max_vel)

        while resp != 'a':

            if resp == 'd':
                exclude.append(trial)
                break

            f, (ax_p, ax_v) = plt.subplots(2, sharex=True)

            if resp == 'm':
                start = float(raw_input("Start time?\n"))
                stop = float(raw_input("Stop time?\n"))
                cropped_tmp = cropped.time_slice(trial, start, stop)

            elif resp != '':
                max_vel = float(resp)
                print("velcrop", max_vel)
                cropped_tmp = cropped.velocity_crop(max_vel, trial, ax_v)

            cropped_tmp.plot_pos(ax_p, trial)
            cropped_tmp.plot_vel(ax_v, trial)
            plt.show()
            resp = raw_input("Velocity cutoff? [a to accept, m to manually enter times, "
                                "<ENTER> to use %f again]:\n" % max_vel)

        cropped = cropped_tmp

    with open(args.output_data_npz, 'w') as output_file:
        np.savez_compressed(output_file, **cropped.get_npz_data(exclude=exclude))
        print('Task sequences saved to ' + args.output_data_npz)



def combine():
    parser = argparse.ArgumentParser()
    parser.add_argument('first', help='The first input npz data')
    parser.add_argument('second', help='The second input npz data')
    parser.add_argument('output_data_npz', help='The output npz data')
    args = parser.parse_args()

    raw_data = np.load(args.input_data_npz)

    with open(args.output_data_npz, 'w') as output_file:
        np.savez_compressed(output_file, **cropped.get_npz_data(exclude=exclude))
        print('Task sequences saved to ' + args.output_data_npz)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass