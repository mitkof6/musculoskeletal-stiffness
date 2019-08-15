import itertools
import numpy as np
import pylab as plt
import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from util import to_np_mat, to_np_array, plot_corr_ellipses, draw_ellipse, \
    convex_bounded_vertex_enumeration, nullspace
from logger import Logger


# ------------------------------------------------------------------------
# FeasibleMuscleSetAnalysis
# ------------------------------------------------------------------------

def construct_muscle_space_inequality(NR, fm_par, Fmax):
    """Construct the feasible muscle space Z f_m0 <= B .

    Parameters
    ----------

    NR: moment arm null space matrix

    fm_par: particular muscle forces

    Fmax: maximum muscle force

    """
    Z0 = -NR
    Z1 = NR
    B0 = fm_par
    B1 = np.asmatrix(np.diag(Fmax)).reshape(Fmax.shape[0], 1) - fm_par
    Z = np.concatenate((Z0, Z1), axis=0)
    B = np.concatenate((B0, B1), axis=0)
    return Z, B


class FeasibleMuscleSetAnalysis:
    """Feasible muscle set analysis.

    The required command along with the state of the system are recorded. Then
    this information is used to compute the feasible muscle null space and
    visualize it.

    """

    def __init__(self, model, simulation_reporter):
        """
        """
        self.logger = Logger('FeasibleMuscleSetAnalsysis')
        self.model = model
        self.simulation_reporter = simulation_reporter

    def visualize_simple_muscle(self, t, ax=None):
        """Visualize the feasible force set at a particular time instance for a linear
        muscle.

        Parameters
        ----------

        t: time

        ax: 1 x 3 axis

        """
        m = self.model.md
        q, Z, B, NR, fm_par = self.calculate_simple_muscles(t)
        x_max = np.max(to_np_mat(self.model.Fmax))
        fm_set = self.generate_solutions(Z, B, NR, fm_par)
        dataframe = pd.DataFrame(fm_set, columns=['$m_' + str(i) + '$' for i in
                                                  range(1, m + 1)])

        # box plot
        if ax is None or ax.shape[0] < 3:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # box plot
        dataframe.plot.box(ax=ax[0])
        ax[0].set_xlabel('muscle id')
        ax[0].set_ylabel('force $(N)$')
        ax[0].set_title('Muscle-Force Box Plot')
        ax[0].set_ylim([0, 1.1 * x_max])

        # correlation matrix
        cmap = mpl.cm.jet
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        corr = dataframe.corr()
        m = plot_corr_ellipses(corr, ax=ax[1], norm=norm, cmap=cmap)
        cb = plt.colorbar(m, ax=ax[1], orientation='vertical', norm=norm,
                          cmap=cmap)
        cb.set_label('Correlation Coefficient')
        ax[1].margins(0.1)
        ax[1].set_xlabel('muscle id')
        ax[1].set_ylabel('muscle id')
        ax[1].set_title('Correlation Matrix')
        ax[1].axis('equal')

        # draw model
        self.model.draw_model(q, False, ax[2], scale=0.7, text=False)

    def calculate_simple_muscles(self, t):
        """Construct Z f_m0 <= B for the case of a linear muscle model for a particular
        time instance.

        Parameters
        ----------

        t: time

        """
        # find nearesrt index corresponding to t
        idx = np.abs(np.array(self.simulation_reporter.t) - t).argmin()
        t = self.simulation_reporter.t[idx]
        q = self.simulation_reporter.q[idx]
        u = self.simulation_reporter.u[idx]
        tau = self.simulation_reporter.tau[idx]
        pose = self.model.model_parameters(q=q, u=u)
        n = self.model.nd

        # calculate required variables
        R = to_np_mat(self.model.R.subs(pose))
        RBarT = np.asmatrix(np.linalg.pinv(R.T))
        # reduce to independent columns to avoid singularities (proposition 3)
        NR = nullspace(R.transpose())
        fm_par = np.asmatrix(-RBarT * tau.reshape((n, 1)))
        Fmax = to_np_mat(self.model.Fmax)

        Z, B = construct_muscle_space_inequality(NR, fm_par, Fmax)

        return q, Z, B, NR, fm_par

    def generate_solutions(self, A, b, NR, fm_par):
        """Sample the solution space that satisfy A x <= b.

        Parameters
        ----------

        A: matrix A

        b: column vector

        NR: moment arm nullspace

        fm_par: particular solution

        Returns
        -------

        muscle forces: a set of solutions that satisfy the problem

        """
        feasible_set = []
        fm0_set = convex_bounded_vertex_enumeration(np.array(A),
                                                    np.array(b).flatten(), 0)
        n = fm0_set.shape[0]
        for i in range(0, n):
            fm = fm_par + NR * np.matrix(fm0_set[i, :]).reshape(-1, 1)
            feasible_set.append(fm)

        return np.array(feasible_set).reshape(n, -1)


def test_feasible_set(model):
    feasible_set = FeasibleMuscleSetAnalysis(model)
    n = model.nd
    m = model.md
    feasible_set.record(1,
                        np.random.random((m, 1)),
                        np.random.random((m, m)),
                        np.random.random((n, 1)))
    fig, ax = plt.subplots(2, 3, figsize=(10, 10))
    feasible_set.visualize_simple_muscle(1, ax[0])
    feasible_set.visualize_simple_muscle(1, ax[1])
    plt.show()

# ------------------------------------------------------------------------
# StiffnessAnalysis
# ------------------------------------------------------------------------


class StiffnessAnalysis:
    """Stiffness analysis.

    """

    def __init__(self, model, task, simulation_reporter,
                 feasible_muscle_set_analysis):
        """Constructor.

        Parameters
        ----------

        model: ArmModel

        task: TaskSpace

        simulation_reporter: SimulationReporter

        feasible_muscle_set_analysis: FeasibleMuscleSetAnalysis

        """
        self.logger = Logger('StiffnessAnalysis')
        self.model = model
        self.task = task
        self.simulation_reporter = simulation_reporter
        self.feasible_muscle_set_analysis = feasible_muscle_set_analysis
        self.dataframe = pd.DataFrame()
        self.color = itertools.cycle(('r', 'g', 'b'))
        self.marker = itertools.cycle(('o', '+', '*'))
        self.linestyle = itertools.cycle(('-', '--', ':'))
        self.hatch = itertools.cycle(('//', '\\', 'x'))

    def visualize_stiffness_properties(self, t, calc_feasible_stiffness,
                                       scale_factor, alpha, ax,
                                       axis_limits=None):
        """Visualize task stiffness ellipse.

        Parameters
        ----------

        t: float
            time

        calc_feasible_stiffness: bool
            whether to calculate the feasible stiffness

        scale_factor: float
            ellipse scale factor

        alpha: float
            alpha value for drawing the model

        ax: matplotlib

        axis_limits: list of pairs
            axis limits for the 3D plot

        """
        at, q, xc, Km, Kj, Kt = self.calculate_stiffness_properties(t,
                                                                    calc_feasible_stiffness)

        # ellipse
        self.model.draw_model(q, False, ax[0], 1, True, alpha, False)
        phi = []
        eigen_values = []
        area = []
        eccentricity = []
        for K in Kt:
            phi_temp, eigen_values_temp, v = draw_ellipse(ax[0], xc, K,
                                                          scale_factor,
                                                          True)
            axes_length = np.abs(eigen_values_temp).flatten()
            idx = axes_length.argsort()[::-1]
            eccentricity_temp = np.sqrt(1 - axes_length[idx[1]]**2
                                        / axes_length[idx[0]]**2)
            area_temp = scale_factor ** 2 * np.pi * axes_length[0] * axes_length[1]

            if phi_temp < 0:
                phi_temp = phi_temp + 180

            phi.append(phi_temp)
            eigen_values.append(eigen_values_temp)
            area.append(area_temp)
            eccentricity.append(eccentricity_temp)

        ax[0].set_xlabel('x $(m)$')
        ax[0].set_ylabel('y $(m)$')
        ax[0].set_title('Task Stiffness Ellipses')

        # ellipse properties
        if axis_limits is not None:
            ax[1].set_xlim(axis_limits[0][0], axis_limits[0][1])
            # ax[1].set_ylim(axis_limits[1][0], axis_limits[1][1])
            ax[1].set_ylim(axis_limits[2][0], axis_limits[2][1])
            for i in range(0, len(area)):  # remove outliers
                if area[i] < axis_limits[0][0] or area[i] > axis_limits[0][1] or \
                   eccentricity[i] < axis_limits[1][0] or eccentricity[i] > axis_limits[1][1] or \
                   phi[i] < axis_limits[2][0] or phi[i] > axis_limits[2][1]:
                    area[i] = np.NaN
                    eccentricity[i] = np.NaN
                    phi[i] = np.NaN


        # color_cycle = ax[1]._get_lines.prop_cycler
        # color = next(color_cycle)['color']
        color = self.color.__next__()
        marker = self.marker.__next__()
        linestyle = self.linestyle.__next__()
        data = pd.DataFrame()
        data['area'] = area
        data['phi'] = phi
        sns.distplot(data['area'].dropna(), kde=True, rug=False, hist=False, vertical=False,
                     color=color, kde_kws={'linestyle': linestyle}, ax=ax[1])
        sns.distplot(data['phi'].dropna(), kde=True, rug=False, hist=False, vertical=True,
                     color=color, kde_kws={'linestyle': linestyle}, ax=ax[1])
        ax[1].scatter(area, phi, label=str(t) + 's', color=color, marker=marker)
        ax[1].set_xlabel('area $(m^2)$')
        # ax[1].set_ylabel('$\epsilon$')
        ax[1].set_ylabel('$\phi (deg)$')
        ax[1].set_title('Task Stiffness Properties')
        ax[1].legend()

        # joint stiffness
        Kj_temp = [np.abs(kj.diagonal()).reshape(-1, 1) for kj in Kj]
        Kj_temp = np.array(Kj_temp).reshape(len(Kj_temp), -1)
        n = self.model.nd
        current_df = pd.DataFrame(Kj_temp, columns=['$J_' + str(i) + '$'
                                                    for i in range(1, n + 1)])
        current_df['Time'] = t
        if not self.dataframe.empty:
            self.dataframe = self.dataframe.append(current_df)
        else:
            self.dataframe = current_df

        ax[2].clear()
        boxplot = sns.boxplot(x='Time', y='Stiffness', hue='Joint',
                    data=self.dataframe.set_index('Time', append=True)
                    .stack()
                    .to_frame()
                    .reset_index()
                    .rename(columns={'level_2': 'Joint', 0: 'Stiffness'})
                    .drop('level_0', axis='columns'), ax=ax[2])
        for b in boxplot.artists:
            b.set_hatch(self.hatch.__next__())

        boxplot.legend()
        ax[2].set_xlabel('time $(s)$')
        ax[2].set_ylabel('joint stiffness $(Nm / rad)$')
        ax[2].set_title('Joint Stiffness')
        ax[2].set_ylim([0, 50])


    def calculate_stiffness_properties(self, t, calc_feasible_stiffness=False):
        """Calculates the stiffness properties of the model at a particular time
        instance.

        Parameters
        ----------

        t: float
            time of interest

        calc_feasible_stiffness: bool
            whether to calculate the feasible stiffness

        Returns
        -------

        t: float
            actual time (closest to recorded values, not interpolated)

        q: mat n x 1 (mat = numpy.matrix)
            generalized coordinates

        xc: mat d x 1
            position of the task

        Km: mat m x m
            muscle space stiffness

        Kj: mat n x n
            joint space stiffness

        Kt: mat d x d
            task space stiffness

        """
        # find nearesrt index corresponding to t
        idx = np.abs(np.array(self.simulation_reporter.t) - t).argmin()
        t = self.simulation_reporter.t[idx]
        q = self.simulation_reporter.q[idx]
        u = self.simulation_reporter.u[idx]
        fm = self.simulation_reporter.fm[idx]
        ft = self.simulation_reporter.ft[idx]
        pose = self.model.model_parameters(q=q, u=u)

        # calculate required variables
        R = to_np_mat(self.model.R.subs(pose))
        RT = R.transpose()
        RTDq = to_np_array(self.model.RTDq.subs(pose))
        Jt = to_np_mat(self.task.Jt.subs(pose))
        JtPInv = np.linalg.pinv(Jt)
        JtTPInv = JtPInv.transpose()
        JtTDq = to_np_array(self.task.JtTDq.subs(pose))
        xc = to_np_mat(self.task.x(pose))

        # calculate feasible muscle forces
        if calc_feasible_stiffness:
            q, Z, B, NR, fm_par = self.feasible_muscle_set_analysis\
                                      .calculate_simple_muscles(t)
            fm_set = self.feasible_muscle_set_analysis.generate_solutions(Z, B,
                                                                          NR,
                                                                          fm_par)

        # calculate stiffness properties
        Km = []
        Kj = []
        Kt = []
        for i in range(0, fm_set.shape[0]):  # , fm_set.shape[0] / 500):
            if calc_feasible_stiffness:
                fm = fm_set[i, :]

            # calculate muscle stiffness from sort range stiffness (ignores
            # tendon stiffness)
            gamma = 23.5
            Km.append(np.asmatrix(np.diag([gamma * fm[i] / self.model.lm0[i]
                                           for i in
                                           range(0, self.model.lm0.shape[0])]),
                                  np.float))

            # switches for taking into account (=1) the tensor products
            dynamic = 1.0
            static = 1.0

            # transpose is required in the tensor product because n(dq) x n(q) x
            # d(t) and we need n(q) x n(dq)

            # calculate joint stiffness
            RTDqfm = np.matmul(RTDq, fm)
            Kj.append(-dynamic * RTDqfm.T - static * RT * Km[-1] * R)

            # calculate task stiffness
            JtTDqft = np.matmul(JtTDq, ft)
            Kt.append(JtTPInv * (Kj[-1] - dynamic * JtTDqft.T) * JtPInv)

        return t, q, xc, Km, Kj, Kt
