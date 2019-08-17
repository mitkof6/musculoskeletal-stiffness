# A variety of useful utilities.
#
# @author Dimitar Stanev (stanev@ece.upatras.gr)
import os
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from fractions import Fraction


def plot_sto(sto_file, plots_per_row, pattern=None, save=False,
             fig_format='.pdf'):
    """Plots the .sto file (OpenSim) by constructing a grid of subplots.

    Parameters
    ----------
    sto_file: str
        path to file
    plots_per_row: int
        subplot columns
    pattern: str, optional, default=None
        plot based on pattern (e.g. only pelvis coordinates)
    save: bool default=False
        save figures
    fig_format: str, optional, default='.pdf'
        format to store the generated plot
    """
    header, labels, data = readMotionFile(sto_file)
    data = np.array(data)
    indices = []
    if pattern is not None:
        indices = index_containing_substring(labels, pattern)
    else:
        indices = range(1, len(labels))

    n = len(indices)
    nrows = int(np.ceil(float(n) / plots_per_row))
    ncols = int(plots_per_row)
    if ncols > n:
        ncols = n

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=(20, 20), sharey=False)
    ax = ax.flatten()
    for p, i in enumerate(indices):
        ax[p].plot(data[:, 0], data[:, i])
        ax[p].set_title(labels[i])
        ax[p].set_xlabel('time (s)')
        # ax[i - 1].set_ylabel('coordinate (deg)')

    fig.tight_layout()
    fig.show()

    if save:
        fig.savefig(sto_file[:-4] + fig_format, dpi=300)


def tensor3_vector_product(T, v):
    """Implements a product of a rank-3 tensor (3D array) with a vector using
    tensor product and tensor contraction.

    Parameters
    ----------

    T: sp.Array of dimensions n x m x k

    v: sp.Array of dimensions k x 1

    Returns
    -------

    A: sp.Array of dimensions n x m

    Example
    -------

    >>>T = sp.Array([[[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]],
                     [[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]]])
    ⎡⎡1  4  7  10⎤  ⎡13  16  19  22⎤⎤
    ⎢⎢           ⎥  ⎢              ⎥⎥
    ⎢⎢2  5  8  11⎥  ⎢14  17  20  23⎥⎥
    ⎢⎢           ⎥  ⎢              ⎥⎥
    ⎣⎣3  6  9  12⎦  ⎣15  18  21  24⎦⎦
    >>>v = sp.Array([1, 2, 3, 4]).reshape(4, 1)
    ⎡1⎤
    ⎢ ⎥
    ⎢2⎥
    ⎢ ⎥
    ⎢3⎥
    ⎢ ⎥
    ⎣4⎦
    >>>tensor3_vector_product(T, v)
    ⎡⎡70⎤  ⎡190⎤⎤
    ⎢⎢  ⎥  ⎢   ⎥⎥
    ⎢⎢80⎥  ⎢200⎥⎥
    ⎢⎢  ⎥  ⎢   ⎥⎥
    ⎣⎣90⎦  ⎣210⎦⎦

    """
    assert(T.rank() == 3)
    # reshape v to ensure 1D vector so that contraction do not contain x 1
    # dimension
    v.reshape(v.shape[0], )
    p = sp.tensorproduct(T, v)
    return sp.tensorcontraction(p, (2, 3))


def mat_show(mat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(to_np_mat(mat), interpolation='nearest')
    fig.colorbar(cax)


def mat(array):
    """For a given 2D array return a numpy matrix.
    """
    return np.matrix(array)


def vec(vector):
    """Construct a column vector of type numpy matrix.
    """
    return np.matrix(vector).reshape(-1, 1)


def to_np_array(sympy_mat):
    """Cast sympy Matrix to numpy matrix of float type. Works for N-D
    matrices as compared to to_np_mat().

    Parameters
    ----------
    m: sympy 2D matrix

    Returns
    -------
    a numpy asmatrix

    """
    return np.asarray(sympy_mat.tolist(), dtype=np.float)


def to_np_mat(sympy_mat):
    """Cast sympy Matrix to numpy matrix of float type.

    Parameters
    ----------
    m: sympy 2D matrix

    Returns
    -------
    a numpy asmatrix

    """
    return np.asmatrix(sympy_mat.tolist(), dtype=np.float)


def to_np_vec(sympy_vec):
    """Transforms a 1D sympy vector (e.g. 5 x 1) to numpy array (e.g. (5,)).

    Parameters
    ----------
    v: 1D sympy vector

    Returns
    -------
    a 1D numpy array

    """
    return np.asarray(sp.flatten(sympy_vec), dtype=np.float)


def lrs_inequality_vertex_enumeration(A, b):
    """Find the vertices given an inequality system A * x <= b. This function
    depends on lrs library.

    Parameters
    ----------

    A: numpy array [m x n]

    b: numpy array [m]

    Returns
    -------

    v: numpy array [k x n]
        the vertices of the polytope

    """
    # export H-representation
    with open('temp.ine', 'w') as file_handle:
        file_handle.write('Feasible_Set\n')
        file_handle.write('H-representation\n')
        file_handle.write('begin\n')
        file_handle.write(str(A.shape[0]) + ' ' +
                          str(A.shape[1] + 1) + ' rational\n')
        for i in range(0, A.shape[0]):
            file_handle.write(str(Fraction(b[i])))
            for j in range(0, A.shape[1]):
                file_handle.write(' ' + str(Fraction(-A[i, j])))

            file_handle.write('\n')

        file_handle.write('end\n')

    # call lrs
    try:
        os.system('lrs temp.ine > temp.ext')
    except OSError as e:
        raise RuntimeError(e)

    # read the V-representation
    vertices = []
    with open('temp.ext', 'r') as file_handle:
        begin = False
        for line in file_handle:
            if begin:
                if 'end' in line:
                    break

                comp = line.split()
                v_type = comp.pop(0)
                if v_type is '1':
                    v = [float(Fraction(i)) for i in comp]
                    vertices.append(v)

            else:
                if 'begin' in line:
                    begin = True

    # delete temporary files
    try:
        os.system('rm temp.ine temp.ext')
    except OSError as e:
        pass

    return vertices


def ccd_inequality_vertex_enumeration(A, b):
    """Find the vertices given an inequality system A * x <= b. This
    function depends on pycddlib (cdd).

    Parameters
    ----------

    A: numpy array [m x n]

    b: numpy array [m]

    Returns
    -------

    v: numpy array [k x n]
        the vertices of the polytope

    """
    import cdd
    # try floating point, if problem fails try exact arithmetics (slow)
    try:
        M = cdd.Matrix(np.hstack((b.reshape(-1, 1), -A)),
                       number_type='float')
        M.rep_type = cdd.RepType.INEQUALITY
        p = cdd.Polyhedron(M)
    except:
        print('Warning: switch to exact arithmetics')
        M = cdd.Matrix(np.hstack((b.reshape(-1, 1), -A)),
                       number_type='fraction')
        M.rep_type = cdd.RepType.INEQUALITY
        p = cdd.Polyhedron(M)

    G = np.array(p.get_generators())

    if not G.shape[0] == 0:
        return G[np.where(G[:, 0] == 1.0)[0], 1:].tolist()
    else:
        raise ValueError('Infeasible Inequality')


def convex_bounded_vertex_enumeration(A, b, convex_combination_passes=1,
                                      method='lrs'):
    """Sample a convex, bounded inequality system A * x <= b. The vertices
    of the convex polytope are first determined. Then the convexity
    property is used to generate additional solutions within the
    polytope.

    Parameters
    ----------

    A: numpy array [m x n]

    b: numpy array [m]

    convex_combination_passes: int (default 1)
        recombine vertices to generate additional solutions using the
        convex property

    method: str (lrs or cdd)

    Returns
    -------

    v: numpy array [k x n]
        solutions within the convex polytope

    """
    # find polytope vertices
    if method == 'lrs':
        solutions = lrs_inequality_vertex_enumeration(A, b)
    elif method == 'cdd':
        solutions = ccd_inequality_vertex_enumeration(A, b)
    else:
        raise RuntimeError('Unsupported method: choose "lrs" or "cdd"')

    # since the feasible space is a convex set we can find additional solution
    # in the form z = a * x_i + (1-a) x_j
    for g in range(0, convex_combination_passes):
        n = len(solutions)
        for i in range(0, n):
            for j in range(0, n):
                if i == j:
                    continue

                a = 0.5
                x1 = np.array(solutions[i])
                x2 = np.array(solutions[j])
                z = a * x1 + (1 - a) * x2
                solutions.append(z.tolist())

    # remove duplicates from 2D list
    solutions = [list(t) for t in set(tuple(element) for element in solutions)]

    return np.array(solutions, np.float)


def construct_muscle_space_inequality(NR, fm_par, fmax):
    """Construct the feasible muscle space Z f_m0 <= B.

    Parameters
    ----------

    NR: moment arm null space matrix

    fm_par: particular muscle forces

    fmax: maximum muscle force

    """
    Z0 = -NR
    Z1 = NR
    b0 = fm_par.reshape(-1, 1)
    b1 = fmax.reshape(-1, 1) - fm_par.reshape(-1, 1)
    Z = np.concatenate((Z0, Z1), axis=0)
    b = np.concatenate((b0, b1), axis=0)
    return Z, b


def null_space(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def readMotionFile(filename):
    """Reads OpenSim .sto files.

    Parameters
    ----------
    filename: str
        absolute path to the .sto file

    Returns
    -------
    header: list of str
        the header of the .sto
    labels: list of str
        the labels of the columns
    data: list of lists
        an array of the data

    """

    if not os.path.exists(filename):
        print('file do not exists')

    file_id = open(filename, 'r')

    # read header
    next_line = file_id.readline()
    header = [next_line]
    nc = 0
    nr = 0
    while not 'endheader' in next_line:
        if 'datacolumns' in next_line:
            nc = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'datarows' in next_line:
            nr = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'nColumns' in next_line:
            nc = int(next_line[next_line.index('=') + 1:len(next_line)])
        elif 'nRows' in next_line:
            nr = int(next_line[next_line.index('=') + 1:len(next_line)])

        next_line = file_id.readline()
        header.append(next_line)

    # process column labels
    next_line = file_id.readline()
    if next_line.isspace() == True:
        next_line = file_id.readline()

    labels = next_line.split()

    # get data
    data = []
    for i in range(1, nr + 1):
        d = [float(x) for x in file_id.readline().split()]
        data.append(d)

    file_id.close()

    return header, labels, data


def index_containing_substring(list_str, pattern):
    """For a given list of strings finds the index of the element that
    contains the substring.

    Parameters
    ----------
    list_str: list of str

    pattern: str
         pattern


    Returns
    -------
    indices: list of int
         the indices where the pattern matches

    """
    indices = []
    for i, s in enumerate(list_str):
        if pattern in s:
            indices.append(i)

    return indices


def simbody_matrix_to_list(M):
    """ Convert simbody Matrix to python list.

    Parameters
    ----------

    M: opensim.Matrix

    """
    return [[M.get(i, j) for j in range(0, M.ncol())]
            for i in range(0, M.nrow())]


def cartesian(arrays, out=None):
    """Generate a cartesian product of input arrays.

    Parameters
    ----------

    arrays: list of array-like
        1-D arrays to form the cartesian product of.

    out: ndarray
        Array to place the cartesian product in.

    Returns
    -------

    out: ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------

    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]

    return out


def construct_coordinate_grid(model, coordinates, N=5):
    """Given n coordinates get the coordinate range and generate a
    coordinate grid of combinations using cartesian product.

    Parameters
    ----------

    model: opensim.Model

    coordinates: list of string

    N: int (default=5)
        the number of points per coordinate

    Returns
    -------

    sampling_grid: np.array
        all combination of coordinates

    """
    sampling_grid = []
    for coordinate in coordinates:
        min_range = model.getCoordinateSet().get(coordinate).getRangeMin()
        max_range = model.getCoordinateSet().get(coordinate).getRangeMax()
        sampling_grid.append(np.linspace(min_range, max_range, N,
                                         endpoint=True))

    return cartesian(sampling_grid)


def find_intermediate_joints(origin_body, insertion_body, model_tree, joints):
    """Finds the intermediate joints between two bodies.

    Parameters
    ----------

    origin_body: string
        first body in the model tree

    insertion_body: string
        last body in the branch

    model_tree: list of dictionary relations {parent, joint, child}

    joints: list of strings
        intermediate joints
    """
    if origin_body == insertion_body:
        return True

    children = filter(lambda x: x['parent'] == origin_body, model_tree)
    for child in children:
        found = find_intermediate_joints(child['child'], insertion_body,
                                         model_tree, joints)
        if found:
            joints.append(child['joint'])
            return True

    return False
