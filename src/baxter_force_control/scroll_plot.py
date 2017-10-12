import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
from itertools import compress

nn = 0
nnls = 'nnls_' if nn else ''
nnls_ = '_nnls' if nn else ''
mode = 'relative'
# mode = 'static'
mode_s = 'rel' if mode == 'relative' else 'stat'
run = 'jolts_2'

data = pd.read_pickle(
    '/home/pedge/experiment/results/peter1/processed/last/peter1_%s/%s%s_velocity_estimator_sink.p' %(run, nnls, mode_s))
udata = pd.read_pickle(
    '/home/pedge/experiment/results/peter1/processed/last/peter1_%s/%s%s_controller_estimator_sink.p' %(run, nnls, mode_s))
weight_data = pd.read_pickle(
    '/home/pedge/experiment/results/peter1/processed/last/peter1_%s/merger_sink.p' % run)

print(weight_data.columns)

erg_weights = np.array(weight_data['ergonomic_cost_1_%s%s' % (mode, nnls_)])
conf_weights = np.array(weight_data['configuration_cost_1_%s%s' % (mode, nnls_)])
robot_mag = np.array(weight_data['robot_twist'])
if mode == 'relative':
    ff_weights = np.array(weight_data['ff_gain_1_%s%s' % (mode, nnls_)])

erg_weights /= 10.0 #erg_weights.max()
conf_weights /= 10.0 #conf_weights.max()
if mode == 'relative':
    ff_weights /= 10.0 #conf_weights.max()

xyz_order = data.pop('V_order')
# data.pop('V_ff_gain')
# check all same
print(data.columns)

quivers = []
u_norms = []
e_norms = []
e_ratio = []
conf_norms = []
erg_norms = []
v_norms = []
ff_norms = []
cols = ['k', 'c', 'm', 'r', 'y', 'b', 'g']
c = map(to_rgba, cols)
labels = ['cost based control', 'configuration cost descent', 'ergonomic cost descent', 'estimation error', 'feed forward control', 'estimated control', 'observed control']

# C = np.vstack((c, np.repeat(c, 2, axis=0)))
Cs = []

col_names = data.columns
v_err_idx = col_names.get_loc('V_err')
v_ff_idx = col_names.get_loc('V_ff')
v_erg_idx = col_names.get_loc('V_ergonomic_cost')
v_conf_idx = col_names.get_loc('V_configuration_cost')

for r, row in data.iterrows():
    quiver = np.hstack((np.zeros((len(row), 3)), list(row)))
    norms = np.linalg.norm(list(row), axis=1)

    cc = [col for col, norm in zip(c, norms) if norm > 0 ]
    Cs.append(np.vstack((cc, np.repeat(cc, 2, axis=0))))

    quiver[v_err_idx, :3] = row['V_hat'] # start error vector at estimation vector
    quiver[v_ff_idx, :3] = row['V_c'] # start feed forward vector at cost control vector
    u_norms.append(np.linalg.norm(udata.iloc[r]['u_orig']))
    e_norms.append(np.linalg.norm(udata.iloc[r]['u_err']))
    v_norm = np.linalg.norm(udata.iloc[r]['u_hat'])
    # e_ratio.append(e_norms[-1]/u_norms[-1] if u_norms[-1] > 0 else 0)
    e_ratio.append((udata.iloc[r]['u_hat']/v_norm).dot(udata.iloc[r]['u_orig']/u_norms[-1]) if u_norms[-1] > 0 else 0)
    conf_norm = np.linalg.norm(quiver[v_conf_idx, 3:])
    erg_norm = np.linalg.norm(quiver[v_erg_idx, 3:])
    v_norms.append(v_norm)
    if conf_norm > 0:
        quiver[v_conf_idx, 3:] *= u_norms[-1] / conf_norm

    if erg_norm > 0:
        quiver[v_erg_idx, 3:] *= u_norms[-1] / erg_norm

    conf_norms.append(conf_norm)
    erg_norms.append(erg_norm)
    quivers.append(quiver)

moving = np.array(u_norms) > 25

# cutoff = float(raw_input('cuttoff?'))
# quivers = list(compress(quivers, moving))

min_rat = min(e_ratio)
print(min_rat)
print(max(e_ratio))
X, Y, Z, U, V, W = quivers[0].T
t = np.arange(0, len(quivers)*0.01875-0.01, 0.01875)

max_u = max(u_norms)

fig0 = plt.figure()
fig0.suptitle('Non Linear Least Squares Analysis')
ax1 = fig0.add_subplot(212)
ax0 = fig0.add_subplot(211, sharex=ax1)
ax0.plot(t, u_norms, color='g', label='input magnitude')
ax0.plot(t, e_norms, color='r', label='error magnitude')
ax1.plot(t, e_ratio, label='normalized projection of estimate onto observed')
ax0.legend()
ax1.legend()
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Alignment')
ax0.set_ylabel('Magnitude')
ax1.set_ylim([-1.1, 1.1])
ax1.set_xlim([0, t[-1]])

fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')

dummies = [ax.plot([], [], ls='-', c=c)[0] for c in cols]
ax.legend(dummies, labels, bbox_to_anchor=(1, 1),
          bbox_transform=plt.gcf().transFigure)

ax2 = fig.add_subplot(212)
fig.suptitle('Velocities Induced by Control')
# ax2.plot(t, e_ratio)
u = np.array(u_norms)
u /= u.max()
ax2.plot(t, u, label='input magnitude', color='g')
ax2.plot(t, erg_norms, label='ergonomic cost', color='b')
ax2.plot(t, conf_norms, label='configuration cost', color='k')
ax2.plot(t, erg_weights, label='ergonomic weight', color='m')
ax2.plot(t, conf_weights, label='configuration weight', color='c')
ax2.plot(t, robot_mag, label='disturbance magnitude', color='r')
vline = ax2.axvline(0.0)
if mode == 'relative':
    ax2.plot(t, ff_weights, label='feed forward gain', color='y')
ax2.legend()
ax.quiver(X, Y, Z, U, V, W, color=Cs[0])
r = max_u
rl = [-r, r]
ax.set_xlim(rl)
ax.set_ylim(rl)
ax.set_zlim(rl)
ax2.set_xlim([0, t[-1]])
ax2.set_ylim([-1, 1])
# ax2.set_ylim([0, 100])


from matplotlib import animation
from matplotlib.widgets import Slider

def update_quiver(num):
    """updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """
    vline.set_xdata(num)
    num = int(num/0.01875)
    X, Y, Z, U, V, W = quivers[num].T
    ax.collections.pop(0)
    return ax.quiver(X, Y, Z, U, V, W, color=Cs[num])


# you need to set blit=False, or the first set of arrows never gets
# cleared on subsequent frames
# anim = animation.FuncAnimation(fig, update_quiver, interval=500, blit=False)

axcolor = 'lightgoldenrodyellow'
axpos = plt.axes([0.125, 0.05, 0.775, 0.03], facecolor=axcolor)
spos = Slider(axpos, 'Pos', 0, len(quivers)*0.01875)
spos.on_changed(update_quiver)

plt.show()