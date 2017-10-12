import argparse
import rosbag
import rospy
import pandas
import numpy as np
from baxter_force_control.steppables import BagReader
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

dt = 0.1875


def get_joint_states(topic, bag_reader):
    time = np.arange(0, dt*len(bag_reader), dt)
    state_dicts = map(joint_state_to_dict, bag_reader[topic])
    return pandas.DataFrame(state_dicts, index=time)


def joint_state_to_dict(joint_state_msg):
    return {name: position for name, position in zip(joint_state_msg.name, joint_state_msg.position)}


def vector_to_array(vector_msg):
    return np.array([vector_msg.vector.x, vector_msg.vector.y, vector_msg.vector.z])


def component_names(name):
    return [name + '_x', name + '_y', name + '_z']


def get_vectors(topic, bag_reader):
    time = np.arange(0, dt*len(bag_reader), dt)
    vectors = map(vector_to_array, bag_reader[topic])
    return pandas.DataFrame(vectors, index=time, columns=component_names(topic))


def normalize(df, columns, lnorm=1):
    lnorm = None if lnorm == 2 else lnorm
    sub_df = df[list(columns)]
    n = np.linalg.norm(sub_df, axis=1, ord=lnorm).reshape((-1, 1)) * np.ones(sub_df.shape)
    result = sub_df/n
    suffix = '_'.join(columns[0].split('_')[2:])
    result['k_'+suffix] = pandas.Series(n[:, 0], index=result.index)
    return result


def final_result(df, suffix='relative_nnls', non_zero=False, thresh=1):
    bases = ['configuration_cost', 'ergonomic_cost']
    suffix = suffix if suffix == '' else '_' + suffix
    names = [b + suffix for b in bases]
    normed = normalize(df, names)
    u_mag = df['arm_joint_mag']
    condition = u_mag > thresh
    if non_zero:
        condition = np.all((np.all(normed!=0, axis=1), condition), axis=0)

    return normed[condition]


def compare(df1, df2, suffix='relative_nnls', non_zero=False, thresh=1):
    w1 = final_result(df1, suffix, non_zero, thresh)
    w2 = final_result(df2, suffix, non_zero, thresh)


def regress():
    cnt3 = final_result(get_results('experiment/results/charlott/processed/nnls_new/PROCESSED_charlott_no_trans_3.bag'))[['ergonomic_cost_relative_nnls']]
    cnt6 = final_result(get_results('experiment/results/charlott/processed/nnls_new/PROCESSED_charlott_no_trans_6.bag'))[['ergonomic_cost_relative_nnls']]
    pnt3 = final_result(get_results('experiment/results/peter1/processed/pf1/PROCESSED_peter1_no_trans_3.bag'))[['ergonomic_cost_relative_nnls']]
    pnt6 = final_result(get_results('experiment/results/peter1/processed/pf1/PROCESSED_peter1_no_trans_6.bag'))[['ergonomic_cost_relative_nnls']]
    locs = [
        (100, 120),
        (150, 160),
        (170, 180),
        (190, 215),
        (410, 430),
        (520, 540)
    ]
    data = pandas.concat([cnt3.add_suffix('_3c'), cnt6.add_suffix('_6c'), pnt3.add_suffix('_3p'), pnt6.add_suffix('_6p')])

    results = []
    for s, e in locs:
        result = []
        row_sub = data.query('%d < index < %d' %(s, e))
        for col in row_sub:
            d = row_sub.reset_index()
            result.append(sm.ols(formula='%s ~ index'%col, data=d).fit())

        results.append(result)

    for res in results:
        for r in res:
            raw_input(r.summary())
    return results


def regress():
    cnt3 = get_results('experiment/results/charlott/processed/nnls_new/PROCESSED_charlott_no_trans_3.bag')
    cnt6 = get_results('experiment/results/charlott/processed/nnls_new/PROCESSED_charlott_no_trans_6.bag')
    pnt3 = get_results('experiment/results/peter1/processed/pf1/PROCESSED_peter1_no_trans_3.bag')
    pnt6 = get_results('experiment/results/peter1/processed/pf1/PROCESSED_peter1_no_trans_6.bag')
    ncnt3 = final_result(cnt3)
    ncnt6 = final_result(cnt6)
    npnt3 = final_result(pnt3)
    npnt6 = final_result(pnt6)

    locs = [
        (100, 120),
        (150, 160),
        (170, 180),
        (190, 215),
        (410, 430),
        (520, 540)
    ]
    data = pandas.concat([cnt3.add_suffix('_3c'), cnt6.add_suffix('_6c'), pnt3.add_suffix('_3p'), pnt6.add_suffix('_6p')])

    results = []
    for s, e in locs:
        row_sub = data.query('%d < index < %d' %(s, e))
        row_sub.plot.reset_index().plot(x='index', y=['ergonomic_cost_relative_nnls', ])



def plotem(w=10):
    #
    # cnt3 = get_results('experiment/results/charlott/processed/nnls_new/PROCESSED_charlott_no_trans_3.bag')
    # cnt6 = get_results('experiment/results/charlott/processed/nnls_new/PROCESSED_charlott_no_trans_6.bag')
    pnt3 = get_results('experiment/results/peter1/processed/qwer/PROCESSED_peter1_jolts_2.bag')
    pnt6 = get_results('experiment/results/peter1/processed/qwer/PROCESSED_peter1_jolts_4.bag')
    # pnt3 = get_results('experiment/results/peter1/processed/qwer/PROCESSED_peter1_no_trans_3.bag')
    # pnt6 = get_results('experiment/results/peter1/processed/qwer/PROCESSED_peter1_no_trans_6.bag')
    # ncnt3 = final_result(cnt3)
    # ncnt6 = final_result(cnt6)
    rnpnt3 = normalize(pnt3, ['ergonomic_cost_%d_relative_nnls'%w, 'configuration_cost_%d_relative_nnls'%w])
    snpnt3 = normalize(pnt3, ['ergonomic_cost_%d_static_nnls'%w, 'configuration_cost_%d_static_nnls'%w])
    rnpnt6 = normalize(pnt6, ['ergonomic_cost_%d_relative_nnls'%w, 'configuration_cost_%d_relative_nnls'%w])
    snpnt6 = normalize(pnt6, ['ergonomic_cost_%d_static_nnls'%w, 'configuration_cost_%d_static_nnls'%w])
    rnpnt6 = normalize(pnt6, ['ergonomic_cost_%d_relative'%w, 'configuration_cost_%d_relative'%w])
    snpnt6 = normalize(pnt6, ['ergonomic_cost_%d_static'%w, 'configuration_cost_%d_static'%w])
    # elif t=='jolts':
    #     cnt3 = \
    #     final_result(get_results('experiment/results/charlott/processed/nnls_new/PROCESSED_charlott_jolts_2.bag'))[
    #         ['ergonomic_cost_relative_nnls']]
    #     cnt6 = \
    #     final_result(get_results('experiment/results/charlott/processed/nnls_new/PROCESSED_charlott_jolts_4.bag'))[
    #         ['ergonomic_cost_relative_nnls']]
    #     pnt3 = final_result(get_results('experiment/results/peter1/processed/pf1/PROCESSED_peter1_jolts_2.bag'))[
    #         ['ergonomic_cost_relative_nnls']]
    #     pnt6 = final_result(get_results('experiment/results/peter1/processed/pf1/PROCESSED_peter1_jolts_4.bag'))[
    #         ['ergonomic_cost_relative_nnls']]

    interests = ['arm_joint_mag', 'robot_twist', 'grip_twist', 'ergonomic_cost', 'configuration_cost', 'shoulder_0',
                 'shoulder_1', 'shoulder_2', 'elbow'] + \
                ['ergonomic_cost_%d_relative_nnls' % w, 'configuration_cost_%d_relative_nnls' % w] + \
                ['ergonomic_cost_%d_static_nnls' % w, 'configuration_cost_%d_static_nnls' % w] + \
                ['ergonomic_cost_%d_relative' % w, 'configuration_cost_%d_relative' % w] + \
                ['ergonomic_cost_%d_static' % w, 'configuration_cost_%d_static' % w]

    nnls = ''
    nnls = ' (NNLS)'
    labels = ['Arm Joint Magnitude', 'Disturbance Magnitude', '', 'Ergonomic Cost (x10)', 'Configuration Cost (x10)', '', '', '', '',
              'Feed Forward Ergonomic Weight'+nnls, 'Feed Forward Configuration Weight'+nnls,
              'Ergonomic Weight'+nnls, 'Configuration Weight' + nnls,

              'Feed Forward Ergonomic Weight', 'Feed Forward Configuration Weight',
              'Ergonomic Weight', 'Configuration Weight']
    data3 = pandas.concat([pnt3[interests], rnpnt3.add_suffix('_normalized'), snpnt3.add_suffix('_normalized')])
    data3 = pandas.concat([pnt6[interests], rnpnt6.add_suffix('_normalized'), snpnt6.add_suffix('_normalized')])

    data3 = pnt6[interests]
    data3.index /= 10.0
    data3['ergonomic_cost'] *= 10
    data3['configuration_cost'] *= 10

    q = ''

    while q != 'q':
        show = ['%d: %s'%(i, s) for i, s in enumerate(data3.keys())]
        l = raw_input('\n'.join(show)+'\n')
        l = l.split(' ')
        l = [int(i) for i in l]
        ll = [labels[interests.index(k)] for k in data3.keys()]
        print(ll)
        col = list('bgrmcyk')*10
        ax = get_ax()
        for i, c in zip(l, col):
            data3.reset_index().plot(x='index', y=data3.keys()[i], label=ll[i], ax=ax, color=c)
        ax.legend()
        ax.set_title(raw_input('title: '))
        ax.set_xlim(41.0, 52.0)
        plt.show()
        q=raw_input('q?\n')


def nearest(w):
    cnt3 = final_result(get_results('experiment/results/charlott/processed/nnls_new/PROCESSED_charlott_no_trans_3.bag'))[['ergonomic_cost_relative_nnls']]
    cnt6 = final_result(get_results('experiment/results/charlott/processed/nnls_new/PROCESSED_charlott_no_trans_6.bag'))[['ergonomic_cost_relative_nnls']]
    pnt3 = final_result(get_results('experiment/results/peter1/processed/pf1/PROCESSED_peter1_no_trans_3.bag'))[['ergonomic_cost_relative_nnls']]
    pnt6 = final_result(get_results('experiment/results/peter1/processed/pf1/PROCESSED_peter1_no_trans_6.bag'))[['ergonomic_cost_relative_nnls']]

    data = pandas.concat([cnt3.add_suffix('_3c'), cnt6.add_suffix('_6c'), pnt3.add_suffix('_3p'), pnt6.add_suffix('_6p')])
    means = []
    for t0 in data.index[data.index<(data.index[-1]-w)]:
        tw = data.query('%d < index < %d' %(t0, t0+w))
        twm = tw.mean(axis=0)
        if any(twm.isnull()):
            print(twm)
            continue
        means.append(twm)

    means = np.array(means)
    em = []
    for col in means.T:
        row = []
        for col2 in means.T:
            row.append(np.linalg.norm(col-col2))
        em.append(row)

    return em, means


    return data

def get_ax():
    fig = plt.figure()
    return fig.add_subplot(111)


def plot_results(df, suffix='relative_nnls', non_zero=False, thresh=1, line=False, ax=None, **kwargs):
    to_plot = final_result(df, suffix, non_zero, thresh)
    to_plot['norm'] = to_plot['norm']/np.max(to_plot['norm'])
    suffix = suffix if suffix == '' else '_' + suffix
    y = 'ergonomic_cost' + suffix
    plot = to_plot.reset_index().plot if line else to_plot.reset_index().plot.scatter
    plot(x='index', y=y, ax=ax, **kwargs)
    kwargs['label'] = 'k'
    to_plot.reset_index().plot(x='index', y='norm', ax=ax, **kwargs)


def get_results(bag_name):
    rospy.init_node('results_viewer')
    bag = rosbag.Bag(bag_name)
    bag_reader = BagReader(bag)

    outputs = get_joint_states('all_output', bag_reader)
    weight_pairs = [
        ['configuration_cost_relative', 'ergonomic_cost_relative'],
        ['configuration_cost_relative_nnls', 'ergonomic_cost_relative_nnls'],
        ['configuration_cost_static', 'ergonomic_cost_static'],
        ['configuration_cost_static_nnls', 'ergonomic_cost_static_nnls']
    ]

    return outputs
