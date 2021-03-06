# This script calculates the feasible muscle forces that satisfy the action
# (motion) and the physiological constraints of the muscles. The requirements
# for the calculation are the model, the motion from inverse kinematics, the
# muscle forces from static optimization. Results must be stored in the
# appropriate directory so that perform_joint_reaction_batch.py can locate the
# files.
#
# @author Dimitar Stanev (jimstanev@gmail.com)
#
import os
import pickle
import numpy as np
from tqdm import tqdm
from opensim_utils import calculate_muscle_data
from util import null_space, construct_muscle_space_inequality, \
    convex_bounded_vertex_enumeration, readMotionFile

###############################################################################
# utilities


def calculate_feasible_muscle_forces(model_file, ik_file, so_file,
                                     results_dir):
    """The calculation of the feasible muscle forces that satisfy the movement and
    physiological muscle constraints is based on the method developed in [1].

    [1] D. Stanev and K. Moustakas, Modeling musculoskeletal kinematic and
        dynamic redundancy using null space projection, PLoS ONE, 14(1):
        e0209171, Jan. 2019, DOI: https://doi.org/10.1371/journal.pone.0209171

    """
    moment_arm, max_force = calculate_muscle_data(model_file, ik_file)

    so_header, so_labels, so_data = readMotionFile(so_file)
    so_data = np.array(so_data)

    muscles = moment_arm[0].shape[1]
    time = so_data[:, 0]
    entries = time.shape[0]

    # collect quantities for computing the feasible muscle forces
    NR = []
    Z = []
    b = []
    fm_par = []
    print('Collecting data ...')
    for t in tqdm(range(0, entries)):
        # get tau, R, Fmax
        fm = so_data[t, 1:(muscles + 1)]  # time the first column
        RT_temp = moment_arm[t, :, :]
        fmax_temp = max_force[t, :]

        # calculate the reduced rank (independent columns) null space to avoid
        # singularities
        NR_temp = null_space(RT_temp)
        # fm_par = fm is used instead of fm_par = -RBarT * tau because the
        # muscle may not be able to satisfy the action. In OpenSim residual
        # actuators are used to ensure that Static Optimization can satisfy the
        # action. In this case, we ignore the reserve forces and assume that fm
        # is the minimum effort solution. If the model is able to satisfy the
        # action without needing reserve forces then we can use fm_par = -RBarT
        # * tau as obtained form Inverse Dynamics.
        # A better implementation that usese fm_par = -RBarT * tau is provided:
        # https://github.com/mitkof6/feasible_muscle_force_analysis
        # this implementation also supports nonlinear muscles and can excluded 
        # muscles and coordinates from the analysis.
        fm_par_temp = fm

        Z_temp, b_temp = construct_muscle_space_inequality(NR_temp,
                                                           fm_par_temp,
                                                           fmax_temp)

        # append results
        NR.append(NR_temp)
        Z.append(Z_temp)
        b.append(b_temp)
        fm_par.append(fm_par_temp)

    # calculate the feasible muscle force set
    print('Calculating null space ...')
    f_set = []
    for t in tqdm(range(0, entries)):
        try:
            fs = convex_bounded_vertex_enumeration(Z[t], b[t][:, 0], 0,
                                                   method='lrs')
        except:
            print('inequlity is infeasible thus append previous iteration')
            f_set.append(f_set[-1])
            continue

        temp = []
        for i in range(0, fs.shape[0]):
            temp.append(fm_par[t] + NR[t].dot(fs[i, :]))

        f_set.append(temp)

    # serialization f_set -> [time x feasible force set set x muscles]
    pickle.dump(f_set, file(results_dir + 'f_set.dat', 'w'))


###############################################################################
# main

def main():
    # when computed once results are stored into files so that they can be loaded
    # with (pickle)
    subject_dir = os.getcwd() + '/../dataset/Gait10dof18musc/'
    model_file = subject_dir + 'subject01.osim'
    ik_file = os.getcwd() + '/notebook_results/subject01_walk_ik.mot'
    so_file = os.getcwd() + '/notebook_results/subject01_walk_StaticOptimization_force.sto'
    results_dir = os.getcwd() + '/notebook_results/'

    # read opensim files
    if not (os.path.isfile(model_file) and
            os.path.isfile(ik_file) and
            os.path.isfile(so_file)):
        raise RuntimeError('required files do not exist')

    if not os.path.isdir(results_dir):
        raise RuntimeError('required folders do not exist')

    calculate_feasible_muscle_forces(model_file, ik_file, so_file, results_dir)
