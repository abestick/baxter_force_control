class PidController:
    def __init__(self, k_p=0.0, k_i=0.0, k_d=0.0):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        self.desired_value = None

    def set_desired_value(self, value):
        self.desired_value = value

    def get_control_cmd(self, cur_value, cur_derivative):
        if self.desired_value is not None:
            control_cmd = self.k_p * (self.desired_value - cur_value)
            control_cmd += -self.k_d * cur_derivative
            return control_cmd
        else:
            return 0.0