# \brief Calculates the symbolic expression of the muscle moment arm for an
#  OpenSim .osim model. The moment arm is sampled and approximated by a
#  multivariate polynomial, so that higher order derivatives can be
#  computed. This implementation works with OpenSim v4.0 API and Python
#  bindings.
#
# Dependencies: opensim, matplotlib, numpy, sympy, multipolyfit, tqdm
#
# @author Dimitar Stanev (stanev@ece.upatras.gr)
import os
import csv
import pickle
import opensim
# import collections
import numpy as np
import sympy as sp
import operator  # used in sorted
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for projection='3d'
from matplotlib.backends.backend_pdf import PdfPages
from multipolyfit import multipolyfit, mk_sympy_function
from util import construct_coordinate_grid, find_intermediate_joints
plt.rcParams['font.size'] = 13


###############################################################################
# utilities


def visualize_moment_arm(moment_arm_coordinate, muscle, coordinates,
                         sampling_dict, model_coordinates, model_muscles, R,
                         pdf):
    """Visualize moment arm as 2D or 3D plot.

    Parameters
    ----------

    moment_arm_coordinate: string
        which moment arm (coordinate)

    muscle: string
        which muscle

    coordinates: list of strings
        which coordinates affect the moment arm variable (one or two only)

    sampling_dict: dictionary
        calculated from calculate_moment_arm_symbolically

    model_coordinates: dictionary
        coordinate names and their corresponding indices in the model

    model_muscles: dictionary
        muscle names and their corresponding indices in the model

    R: symbolic moment arm matrix

    pdf: PdfPages
    """
    if isinstance(coordinates, str):
        # coordinates = sampling_dict[muscle]['coordinates']
        sampling_grid = sampling_dict[muscle]['sampling_grid']
        moment_arm = sampling_dict[muscle]['moment_arm']
        idx = coordinates.index(moment_arm_coordinate)
        poly = R[model_muscles[muscle],
                 model_coordinates[moment_arm_coordinate]]
        moment_arm_poly = np.array([
            poly.subs(dict(zip(poly.free_symbols, x))) for x in sampling_grid
        ], np.float)

        fig = plt.figure()
        ax = fig.gca()
        ax.plot(
            sampling_grid[:, idx], moment_arm[:, idx] * 100.0, 'rx',
            label='sampled')
        ax.plot(sampling_grid[:, idx], moment_arm_poly * 100.0, 'b-',
                label='analytical')
        ax.set_xlabel(coordinates + ' (rad)')
        ax.set_ylabel(moment_arm_coordinate + ' (cm)')
        ax.set_title(muscle)
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        # plt.close()
    elif isinstance(coordinates, list) and len(coordinates) == 2:
        # coordinates = sampling_dict[muscle]['coordinates']
        sampling_grid = sampling_dict[muscle]['sampling_grid']
        moment_arm = sampling_dict[muscle]['moment_arm']
        idx = coordinates.index(moment_arm_coordinate)
        poly = R[model_muscles[muscle], model_coordinates[
            moment_arm_coordinate]]

        # poly.free_symbols is not used because it may not preserve order
        poly_symbols = [sp.Symbol(x) for x in coordinates]
        moment_arm_poly = np.array([
            poly.subs(dict(zip(poly_symbols, x))) for x in sampling_grid
        ], np.float)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(
            sampling_grid[:, 0],
            sampling_grid[:, 1],
            moment_arm[:, idx] * 100.0,
            label='sampled',
            color='r')
        surf = ax.plot_trisurf(
            sampling_grid[:, 0],
            sampling_grid[:, 1],
            moment_arm_poly * 100.0,
            label='analytical',
            facecolor='b',
            edgecolor='k',
            linewidth=0.1,
            alpha=0.5,
            antialiased=True)
        surf._facecolors2d = surf._facecolors3d
        surf._edgecolors2d = surf._edgecolors3d
        ax.set_xlabel(coordinates[0] + ' (rad)')
        ax.set_ylabel(coordinates[1] + ' (rad)')
        ax.set_zlabel(moment_arm_coordinate + ' (cm)')
        ax.set_title(muscle)
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        # plt.close()
    else:
        return


def calculate_moment_arm_symbolically(model_file, results_dir):
    """Calculate the muscle moment arm matrix symbolically for a
     particular OpenSim model.

    """
    print('Calculating...')

    # parse csv
    muscle_coordinates = {}
    with open(results_dir + 'muscle_coordinates.csv') as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        for row in reader:
            muscle_coordinates[row[0]] = row[1:]

    # load opensim model
    model = opensim.Model(model_file)
    state = model.initSystem()

    model_coordinates = {}
    for i, coordinate in enumerate(model.getCoordinateSet()):
        model_coordinates[coordinate.getName()] = i

    model_muscles = {}
    for i, muscle in enumerate(model.getMuscles()):
        model_muscles[muscle.getName()] = i

    # calculate moment arm matrix (R) symbolically
    R = []
    sampling_dict = {}
    resolution = {1: 15, 2: 10, 3: 8, 4: 5, 5: 5}
    for muscle, k in tqdm(sorted(model_muscles.items(),
                                 key=operator.itemgetter(1))):
        # get initial state each time
        coordinates = muscle_coordinates[muscle]
        N = resolution[len(coordinates)]

        # calculate moment arms for this muscle and spanning coordinates
        sampling_grid = construct_coordinate_grid(model, coordinates, N)
        moment_arm = []
        for q in sampling_grid:
            for i, coordinate in enumerate(coordinates):
                model.updCoordinateSet().get(coordinate).setValue(state, q[i])

            model.realizePosition(state)
            tmp = []
            for coordinate in coordinates:
                coord = model.getCoordinateSet().get(coordinate)
                tmp.append(model.getMuscles()
                           .get(muscle).computeMomentArm(state, coord))

            moment_arm.append(tmp)

        moment_arm = np.array(moment_arm)
        sampling_dict[muscle] = {
            'coordinates': coordinates,
            'sampling_grid': sampling_grid,
            'moment_arm': moment_arm
        }

        # polynomial regression
        degree = 5
        muscle_moment_row = [0] * len(model_coordinates)
        for i, coordinate in enumerate(coordinates):
            coeffs, powers = multipolyfit(sampling_grid,
                                          moment_arm[:, i],
                                          degree, powers_out=True)
            polynomial = mk_sympy_function(coeffs, powers)

            # the order of the free symbols may be incorrect
            free_symbols = list(polynomial.free_symbols)
            if len(free_symbols) > 1 and \
               str(free_symbols[0]) > str(free_symbols[1]):
                free_symbols = free_symbols[::-1]

            polynomial = polynomial.subs(
                dict(zip(free_symbols, [sp.Symbol(x) for x in coordinates])))
            muscle_moment_row[model_coordinates[coordinate]] = polynomial

        R.append(muscle_moment_row)

    # export data to file because the process is time consuming
    R = sp.Matrix(R)
    with open(results_dir + 'R.dat', 'wb') as f_R,\
         open(results_dir + 'sampling_dict.dat', 'wb') as f_sd,\
         open(results_dir + 'model_muscles.dat', 'wb') as f_mm,\
         open(results_dir + 'model_coordinates.dat', 'wb') as f_mc:
        pickle.dump(R, f_R)
        pickle.dump(sampling_dict, f_sd)
        pickle.dump(model_muscles, f_mm)
        pickle.dump(model_coordinates, f_mc)


def calculate_spanning_muscle_coordinates(model_file, results_dir):
    """Calculates the coordinates that are spanned by each muscle. Useful for
    reducing the required computation of the muscle moment arm matrix.

    """
    model = opensim.Model(model_file)
    state = model.initSystem()

    # construct model tree (parent body - joint - child body)
    model_tree = []
    for joint in model.getJointSet():
        model_tree.append({
            'parent':
            joint.getParentFrame().getName().replace('_offset', ''),
            'joint':
            joint.getName(),
            'child':
            joint.getChildFrame().getName().replace('_offset', '')
        })

    ordered_body_set = []
    for body in model.getBodySet():
        ordered_body_set.append(body.getName())

    ordered_coordinate_set = []
    for coordinate in model.getCoordinateSet():
        ordered_coordinate_set.append(coordinate.getName())

    # get the coordinates that are spanned by the muscles
    muscle_coordinates = {}
    for muscle in model.getMuscles():
        path = muscle.getGeometryPath().getPathPointSet()
        muscle_bodies = []
        for point in path:
            muscle_bodies.append(point.getBodyName())

        # remove duplicate bodies and sort by multibody tree order
        muscle_bodies = list(set(muscle_bodies))
        muscle_bodies = sorted(muscle_bodies,
                               key=lambda x: ordered_body_set.index(x))

        # find intermediate joints
        assert(len(muscle_bodies) > 1)
        joints = []
        find_intermediate_joints(muscle_bodies[0], muscle_bodies[-1],
                                 model_tree, joints)

        # find spanning coordinates
        muscle_coordinates[muscle.getName()] = []
        for joint in joints:
            joint = model.getJointSet().get(joint)
            for i in range(0, joint.numCoordinates()):
                if joint.get_coordinates(i).isDependent(state):
                    continue
                else:
                    muscle_coordinates[muscle.getName()].append(
                        joint.get_coordinates(i).getName())

            # sort coordinates by model order
            muscle_coordinates[muscle.getName()] = sorted(
                muscle_coordinates[muscle.getName()],
                key=lambda x: ordered_coordinate_set.index(x))

    # write results to file
    with open(results_dir + 'muscle_coordinates.csv', 'w') as csv_file:
        for key, values in muscle_coordinates.items():
            csv_file.write(key)
            for value in values:
                csv_file.write(';' + value)

            csv_file.write('\n')


###############################################################################
# main

def main():
    # parameters
    subject_dir = os.getcwd() + '/../data/gait1018/'
    model_file = subject_dir + 'subject01_scaled.osim'
    results_dir = os.getcwd() + '/results/'

    # read opensim files
    if not os.path.isfile(model_file):
        raise RuntimeError('required files do not exist')

    if not os.path.isdir(results_dir):
        raise RuntimeError('required folders do not exist')

    # when computed once results are stored into files and loaded with
    # (pickle)
    compute = True
    visualize = True

    if compute:
        calculate_spanning_muscle_coordinates(model_file, results_dir)
        calculate_moment_arm_symbolically(model_file, results_dir)

    if visualize:
        with open(results_dir + 'R.dat', 'rb') as f_r,\
                open(results_dir + 'sampling_dict.dat', 'rb') as f_sd,\
                open(results_dir + 'model_coordinates.dat', 'rb') as f_mc,\
                open(results_dir + 'model_muscles.dat', 'rb') as f_mm:
            R = pickle.load(f_r)
            sampling_dict = pickle.load(f_sd)
            model_coordinates = pickle.load(f_mc)
            model_muscles = pickle.load(f_mm)

        # visualize data
        with PdfPages(results_dir + 'compare_ma.pdf') as pdf:
            for muscle in sampling_dict.keys():
                coordinates = sampling_dict[muscle]['coordinates']
                if len(coordinates) == 1:
                    visualize_moment_arm(coordinates[0], muscle,
                                         coordinates[0],
                                         sampling_dict,
                                         model_coordinates,
                                         model_muscles, R, pdf)
                elif len(coordinates) == 2:
                    visualize_moment_arm(coordinates[0], muscle, coordinates,
                                         sampling_dict, model_coordinates,
                                         model_muscles, R, pdf)
                    visualize_moment_arm(coordinates[1], muscle, coordinates,
                                         sampling_dict, model_coordinates,
                                         model_muscles, R, pdf)
                else:
                    print('only 2D and 3D visualization, skip: ' + muscle)
