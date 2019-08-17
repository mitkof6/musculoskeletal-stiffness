# Calculation and visualization of the feasible joint stiffness.
#
# @author Dimitar Stanev (stanev@ece.upatras.gr)
import os
import pickle
import opensim
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
from util import readMotionFile, to_np_mat, to_np_array
plt.rcParams['font.size'] = 13

###############################################################################
# functionality


def calculate_feasible_joint_stiffness(model_file, ik_file, results_dir):
    """The calculation of the feasible joint stiffness is described in
    more detailed [2].

    [2] D. Stanev and K. Moustakas, Stiffness Modulation of Redundant
        Musculoskeletal Systems, Journal of Biomechanics}, accepted
        Jan. 2019

    """
    print('Initialization ...')
    with open(results_dir + 'f_set.dat', 'rb') as f_s, \
         open(results_dir + 'R.dat', 'rb') as f_r:
        f_set = pickle.load(f_s)
        R = pickle.load(f_r)

    RT = R.transpose()

    # load OpenSim data
    model = opensim.Model(model_file)
    ik_header, ik_labels, ik_data = readMotionFile(ik_file)
    ik_data = np.array(ik_data)
    time = ik_data[:, 0]
    assert(ik_data.shape[0] == len(f_set))
    coordinates = ik_labels[1:]
    with open(results_dir + 'ik_labels.dat', 'wb') as fo:
        pickle.dump(ik_labels, fo)

    # calculate symbolic derivatives
    q = [sp.Symbol(c) for c in coordinates]
    RTDq = sp.derive_by_array(RT, q)

    # calculate lm0 (optimal fiber length)
    lm0 = []
    for m in model.getMuscles():
        lm0.append(m.getOptimalFiberLength())

    Kj_min = []
    Kj_max = []
    print('Calculating feasible joint stiffness ...')
    for i in tqdm(range(0, len(f_set))):
        pose = np.deg2rad(ik_data[i, 1:])
        configuration = dict(zip(q, pose))
        R_temp = to_np_mat(R.subs(configuration))
        RT_temp = to_np_mat(RT.subs(configuration))
        RTDq_temp = to_np_array(RTDq.subs(configuration))  # 3D array
        Kj = []
        for fm in f_set[i]:
            assert(np.all(fm > -1e-5) == True)
            # calculate muscle stiffness from sort range stiffness (ignores
            # tendon stiffness)
            gamma = 23.5
            Km = np.diag([gamma * fm[m] / lm0[m] for m in range(0, len(lm0))])
            # Km = np.diag([fm[m] for m in range(0, len(lm0))]) calculate joint
            # stiffness, transpose is required because n(dq) x n(q) x d(t) and
            # we need n(q) x n(dq)
            RTDqfm = np.matmul(RTDq_temp, fm)
            Kj_temp = RTDqfm.T + RT_temp * Km * R_temp
            Kj.append(np.diagonal(Kj_temp))

        Kj_min.append(np.min(Kj, axis=0))
        Kj_max.append(np.max(Kj, axis=0))

    # serialization
    Kj_min = np.array(Kj_min)
    Kj_max = np.array(Kj_max)
    with open(results_dir + 'Kj_min.dat', 'wb') as f_kj_min, \
         open(results_dir + 'Kj_max.dat', 'wb') as f_kj_max, \
         open(results_dir + 'time.dat', 'wb') as f_time:
        pickle.dump(Kj_min, f_kj_min)
        pickle.dump(Kj_max, f_kj_max)
        pickle.dump(time, f_time)


def visualize_feasible_joint_stiffness(results_dir, figures_dir):
    """Visualize feasible joint stiffness.
    """
    # load data
    Kj_min = []
    Kj_max = []
    time = []
    ik_labels = []
    with open(results_dir + 'Kj_min.dat', 'rb') as f_kj_min, \
         open(results_dir + 'Kj_max.dat', 'rb') as f_kj_max, \
         open(results_dir + 'time.dat', 'rb') as f_time, \
         open(results_dir + 'ik_labels.dat', 'rb') as f_ik_leb:
        Kj_min = pickle.load(f_kj_min)
        Kj_max = pickle.load(f_kj_max)
        time = pickle.load(f_time)
        ik_labels = pickle.load(f_ik_leb)

    # remove flexion and angle from labels (reviewer comments)
    ik_labels = [l.replace('flexion_', '') for l in ik_labels]
    ik_labels = [l.replace('angle_', '') for l in ik_labels]

    heel_strike_right = [0.65, 1.85]
    toe_off_right = [0.15, 1.4]
    heel_strike_left = [0.0, 1.25]
    toe_off_left = [0.8, 2]

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 5))
    ax = ax.flatten()
    heel_strike = heel_strike_right
    toe_off = toe_off_right
    for joint in range(3, 9):
        i = joint - 3
        if i > 2:
            heel_strike = heel_strike_left
            toe_off = toe_off_left

        ax[i].fill_between(
            time[:], Kj_min[:, joint], Kj_max[:, joint], color='b', alpha=0.2)
        # ax[i].set_yscale('log')
        ax[i].set_title(ik_labels[joint + 1])
        ax[i].set_xlabel('time (s)')
        ax[i].set_ylabel('joint stiffness (Nm / rad)')
        ax[i].vlines(x=heel_strike, ymin=0, ymax=np.max(Kj_max[:, joint]),
                     color='r', linestyle='--', label='HS')
        ax[i].vlines(x=toe_off, ymin=0, ymax=np.max(Kj_max[:, joint]),
                     color='b', linestyle=':', label='TO')
    # annotate
    ax[2].legend()
    ax[-1].legend()

    fig.tight_layout()
    fig.savefig(figures_dir + 'feasible_joint_stiffness.png',
                format='png', dpi=300)
    fig.savefig(figures_dir + 'feasible_joint_stiffness.pdf',
                format='pdf', dpi=300)


#############################################################################
# main

def main():
    # initialization and computation takes time
    compute = True
    subject_dir = os.getcwd() + '/../data/gait1018/'
    model_file = subject_dir + 'subject01_scaled.osim'
    ik_file = os.getcwd() + '/results/subject01_walk1_ik.mot'
    results_dir = os.getcwd() + '/results/'

    # read opensim files
    if not (os.path.isfile(model_file) and
            os.path.isfile(ik_file)):
        raise RuntimeError('required files do not exist')

    if not os.path.isdir(results_dir):
        raise RuntimeError('required folders do not exist')

    if compute:
        calculate_feasible_joint_stiffness(model_file, ik_file, results_dir)

    visualize_feasible_joint_stiffness(results_dir, results_dir)
