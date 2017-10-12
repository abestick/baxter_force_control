import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data = pd.read_pickle(
    '/home/pedge/experiment/results/charlott/processed/mon/charlott_init_0/nnls_rel_controller_estimator_sink.p')
merger = pd.read_pickle(
    '/home/pedge/experiment/results/charlott/processed/mon/charlott_init_0/merger_sink.p')

erg_weights = np.array(merger['ergonomic_cost_1_relative_nnls'])
conf_weights = np.array(merger['configuration_cost_1_relative_nnls'])
d_u = np.array(list(data['u_orig']))
s_d_u = np.empty_like(d_u)

elbow = np.array(merger['joint_1'])
s_elbow = np.empty_like(elbow)
d_elbow = np.empty_like(elbow)

last = np.zeros(4)
alpha = 0.3
for i, row in enumerate(d_u):
    s_d_u[i, :] = (1 - alpha)*last + alpha*row
    last = s_d_u[i, :]

al = 0.15
be = 0

last_s = None
last_b = None
dt = 0.01875
thresh = 0.25
last_d = 0
a = np.empty_like(elbow)

for i, x in enumerate(elbow):
    if last_s is None:
        last_s = x
        last_b = 0
        s_elbow[i] = x
        d_elbow[i] = 0
        a[i] = 0
        continue

    # if last_b is None:
    #     s_elbow[i] = x
    #     d = (x - last_s) / dt
    #     d_elbow[i] = d if abs(d) > thresh else 0
    #     a[i] = 0
    #     last_b = x - last_s
    #     last_s = x
    #     last_d = d

    s = al*x + (1 - al)*(last_s + last_b)
    b = be*(s - last_s) + (1 - be)*last_b
    s_elbow[i] = s
    d = (s - last_s) / dt
    d_elbow[i] = d if abs(d) > thresh else 0
    a[i] = (d - last_d)/dt
    last_s = s
    last_b = b
    last_d = d

ax = plt.axes()
merger.reset_index().plot('index', 'joint_1', ax=ax)
# ax.plot(d_u[:, 3])
# ax.plot(s_d_u[:, 3])
ax.plot(s_elbow, label='s_elbow')
ax.plot(d_elbow, label='d_elbow')
# ax.plot(a, label='a_elbow')
ax.legend()
plt.show()