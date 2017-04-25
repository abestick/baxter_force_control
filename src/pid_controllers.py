from tf.transformations import quaternion_multiply, quaternion_inverse, euler_from_quaternion, quaternion_from_euler
import numpy as np

class PidController:
    def __init__(self, k_p=0.0, k_i=0.0, k_d=0.0):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

    def get_control_cmd(self, cur_value, cur_derivative, reference):
        control_cmd = self.k_p * (reference - cur_value)
        control_cmd += -self.k_d * cur_derivative
        return control_cmd


class PositionPidController:
    def __init__(self, k_p=(0.0, 0.0, 0.0), k_i=(0.0, 0.0, 0.0), k_d=(0.0, 0.0, 0.0)):
        if not isinstance(k_p, (tuple, list)):
            k_p = (k_p,)*3

        if not isinstance(k_i, (tuple, list)):
            k_i = (k_i,)*3

        if not isinstance(k_d, (tuple, list)):
            k_d = (k_d,)*3


        pid = zip(k_p, k_i, k_d)

        self.x = PidController(*pid[0])
        self.y = PidController(*pid[1])
        self.z = PidController(*pid[2])

    def get_control_cmd(self, cur_value, cur_derivative, reference):
        x, y, z = zip(cur_value, cur_derivative, reference)

        return [self.x.get_control_cmd(*x), self.y.get_control_cmd(*y), self.z.get_control_cmd(*z)]


class OrientationPidController:
    def __init__(self, k_p=0.0, k_i=0.0, k_d=0.0):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

    def get_control_cmd(self, cur_value, reference, eular=False):

        if eular:
            reference = quaternion_from_euler(*reference)

        quaternion_diff = quaternion_multiply(reference, quaternion_inverse(cur_value))
        euler_diff = np.array(euler_from_quaternion(quaternion_diff))
        control_cmd = self.k_p * euler_diff
        return list(control_cmd)
