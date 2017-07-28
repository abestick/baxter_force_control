import numpy as np


class AttributeDict:

    @classmethod
    def __getitem__(cls, item):
        return getattr(cls, item)


class ArmDefs(AttributeDict):
    joint_names = ('s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2')

    s0 = joint_names.index('s0')
    s1 = joint_names.index('s1')
    e0 = joint_names.index('e0')
    e1 = joint_names.index('e1')
    w0 = joint_names.index('w0')
    w1 = joint_names.index('w1')
    w2 = joint_names.index('w2')
    
    pos_limits = np.zeros(len(joint_names), dtype={'names': ['min', 'max'], 'formats': ['f4', 'f4']})
    vel_limits = np.zeros(len(joint_names))

    pos_limits['min'][s0] = -1.7016
    pos_limits['min'][s1] = -2.147
    pos_limits['min'][e0] = -3.0541
    pos_limits['min'][e1] = -0.05
    pos_limits['min'][w0] = -3.059
    pos_limits['min'][w1] = -np.pi / 2
    pos_limits['min'][w2] = -3.059

    pos_limits['max'][s0] = 1.7016
    pos_limits['max'][s1] = 1.047
    pos_limits['max'][e0] = 3.0541
    pos_limits['max'][e1] = 2.618
    pos_limits['max'][w0] = 3.059
    pos_limits['max'][w1] = 2.094
    pos_limits['max'][w2] = 3.059

    vel_limits[s0] = 2.0
    vel_limits[s1] = 2.0
    vel_limits[e0] = 2.0
    vel_limits[e1] = 2.0
    vel_limits[w0] = 4.0
    vel_limits[w1] = 4.0
    vel_limits[w2] = 4.0

    ranges = pos_limits['min'] - pos_limits['max']
    neutrals = (pos_limits['max'] + pos_limits['min']) / 2


class BaxterDefs(AttributeDict):

    arm = ArmDefs




