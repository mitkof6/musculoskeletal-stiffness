# This script calculates the feasible muscle forces that satisfy the action
# (motion) and the physiological constraints of the muscles. The requirements
# for the calculation are the model, the motion from inverse kinematics, the
# generalized forces tau from inverse dynamics.
#
# @author Dimitar Stanev (stanev@ece.upatras.gr)
import os
import pickle
import numpy as np
from tqdm import tqdm
from opensim_utils import perform_ik, perform_id, calculate_muscle_data, \
    getMuscleIndices, getCoordinateIndices
from util import null_space, construct_muscle_space_inequality, \
    convex_bounded_vertex_enumeration, readMotionFile

###############################################################################
# utilities


def calculate_feasible_muscle_forces(model_file, ik_file, id_file, results_dir,
                                     excluded_coordinates, excluded_muscles):
    """The calculation of the feasible muscle forces that satisfy the
    movement and physiological muscle constraints is based on the
    method developed in [1].

    [1] D. Stanev and K. Moustakas, Modeling musculoskeletal kinematic and
        dynamic redundancy using null space projection, PLoS ONE, 14(1):
        e0209171, Jan. 2019, DOI: https://doi.org/10.1371/journal.pone.0209171

    """
    moment_arm, max_force = calculate_muscle_data(model_file, ik_file)
    coordinate_indices = getCoordinateIndices(model_file, excluded_coordinates)
    muscle_indices = getMuscleIndices(model_file, excluded_muscles)
    print('Active coordinates: ', coordinate_indices)
    print('Actuve muscles: ', muscle_indices)

    ik_header, ik_labels, ik_data = readMotionFile(ik_file)
    ik_data = np.array(ik_data)
    id_header, id_labels, id_data = readMotionFile(id_file)
    id_data = np.array(id_data)

    time = ik_data[:, 0]
    id_data = id_data[:, 1:]  # remove first column
    entries = time.shape[0]

    # collect quantities for computing the feasible muscle forces
    NR = []
    Z = []
    b = []
    fm_par = []
    print('Collecting data ...')
    for t in tqdm(range(entries)):
        # get tau, R, Fmax
        tau = id_data[t, coordinate_indices]
        RT = moment_arm[t, coordinate_indices, :]
        RT = RT[:, muscle_indices]
        RBarT = np.linalg.pinv(RT)
        fmax = max_force[t, muscle_indices]

        NR_temp = null_space(RT)
        fm_par_temp = - RBarT.dot(tau)
        Z_temp, b_temp = construct_muscle_space_inequality(NR_temp,
                                                           fm_par_temp,
                                                           fmax)

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
    with open(results_dir + 'f_set.dat', 'wb') as fo:
        pickle.dump(f_set, fo)


###############################################################################
# main

def main():
    # required files and directories
    subject_dir = os.getcwd() + '/../data/gait1018/'
    model_file = subject_dir + 'subject01_scaled.osim'
    trc_file = subject_dir + 'subject01_walk1.trc'
    grf_file = subject_dir + 'subject01_walk1_grf.mot'
    grf_xml_file = subject_dir + 'subject01_walk1_grf.xml'
    results_dir = os.getcwd() + '/results/'

    if not (os.path.isfile(model_file) and
            os.path.isfile(trc_file) and
            os.path.isfile(grf_file) and
            os.path.isfile(grf_xml_file)):
        raise RuntimeError('required files do not exist')

    if not os.path.isdir(results_dir):
        raise RuntimeError('required folders do not exist')

    ik_file = perform_ik(model_file, trc_file, results_dir)
    id_file = perform_id(model_file, ik_file, grf_file, grf_xml_file,
                         results_dir)
    calculate_feasible_muscle_forces(model_file, ik_file, id_file, results_dir,
                                     '^pelvis_.*|^lumbar_.*',
                                     'do_not_exclude_any_muscle')
