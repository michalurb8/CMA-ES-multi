import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import ndsorter
from random import shuffle

_EPS = 1e-50
_POINT_MAX = 1e100
_SIGMA_MAX = 1e100

_DELAY = 0.1

INF = float('inf')

# COLORS = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'magenta']
COLORS = ['red', 'blue', 'green', 'black', 'cyan', 'orange', 'purple', 'magenta', 'yellow']
COLOR_NUM = len(COLORS)

class CMAES:
    """
    Parameters
    ----------
    objective_function: str
        Chosen from: quadratic, elliptic, bent, rastrigin, rosenbrock, ackley
    dimensions: int
        Objective function dimensionality.
    lambda_arg: int
        Population count. Must be > 3, if set to None, default value will be computed.
    stop_after: int
        How many iterations are to be run
    visuals: bool
        If True, every algorithm generation will be visualised (only 2 first dimensions)
    """
    def __init__(self, objective_functions, dimensions: int, stop_after:int, lambda_arg: int = None, visuals: bool = False):
        assert dimensions > 0, "Number of dimensions must be greater than 0"
        self._dimension = dimensions
        self._criteria = objective_functions
        self._stop_after = stop_after
        self._visuals = visuals

        # Initial point
        self._xmean = 5*np.random.normal(size = self._dimension)
        # Step size
        self._sigma = 1

        # Population size
        if lambda_arg == None:
            self._lambda = 4 * self._dimension # default population size
        else:
            assert lambda_arg > 3, "Population size must be greater than 3"
            self._lambda = lambda_arg

        # Number of parents/points to be selected
        self._mu = self._lambda // 2

        # Learning rates for rank-one and rank-mu update
        self._lr_c1 = 2 / ((self._dimension + 1.3) ** 2 + self._mu)
        self._lr_c_mu = (2 * (self._mu - 2 + 1 / self._mu) /
                         ((self._dimension + 2) ** 2 + self._mu))
        self._lr_c_mu = min(1-self._lr_c1, self._lr_c_mu)

        # Time constants for cumulation for step-size control
        self._time_sigma = (self._mu + 2) / (self._dimension + self._mu + 5)
        self._damping = 1 + 2 * max(0, np.sqrt((self._mu- 1) / (self._dimension + 1)) - 1) + self._time_sigma

        # Time constants for cumulation for rank-one update
        self._time_c = ((4 + self._mu/ self._dimension) /
                        (4 + self._dimension + 2 * self._mu/ self._dimension))

        # E||N(0, I)||
        self._chi = np.sqrt(self._dimension) * (1 - 1 / (4 * self._dimension) + 1 / (21 * self._dimension ** 2))

        # Evolution paths
        self._path_c = np.zeros(self._dimension)
        self._path_sigma = np.zeros(self._dimension)
        # B defines coordinate system
        self._B = None
        # D defines scaling (diagonal matrix)
        self._D = None
        # Covariance matrix
        self._C = np.eye(self._dimension)

        # Store current generation number
        self._generation = 0

        self.contours_calculated = False
        # Run the algorithm immediately
        self._run()

    def _run(self):
        for _ in range(self._stop_after):
            self._B, self._D = self._eigen_decomposition()

            solutions = [] # this is a list of tuples: [(x, value), (x, value), ...]
            for _ in range(self._lambda):
                x = self._sample_solution()

                values = np.array([criterium(x) for criterium in self._criteria])
                solutions.append([x, values, INF])
            # Update algorithm parameters.
            assert len(solutions) == self._lambda, "There must be exactly lambda points generated"
            self._update(solutions)

    def _sample_solution(self) -> np.ndarray:
        std = np.random.standard_normal(self._dimension)
        return self._xmean + self._sigma * np.matmul(np.matmul(self._B, np.diag(self._D)), std)

    def _eigen_decomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        D2, B = np.linalg.eigh(self._C)
        D = np.sqrt(np.where(D2 < 0, _EPS, D2))
        return B, D

    def _update(self, solutions: List[Tuple[np.ndarray, np.ndarray, float]]) -> None:
        assert len(solutions) == self._lambda, "Must evaluate solutions with length equal to population size."
        for s in solutions:
            assert np.all(
                np.abs(s[0]) < _POINT_MAX
            ), f"Absolute value of all generated points must be less than {_POINT_MAX} to avoid overflow errors."

        self._generation += 1

        ndsorter.calcRank(solutions)
        shuffle(solutions)
        solutions.sort(key=lambda solution: solution[-1]) #sort population by rank

        # ~ N(m, sigma^2 C)
        population = np.array([s[0] for s in solutions])
        # ~ N(0, C)
        y_k = (population - self._xmean) / (self._sigma + _EPS)

        # Selection
        selected = y_k[: self._mu]
        y_w = np.mean(selected, axis=0) # cumulated delta vector

        self._xmean += self._sigma * y_w

        if self._visuals == True and self._dimension > 1:
            title = "Iteracja " + str(self._generation) + ", "
            title += "Liczebność populacji: " + str(self._lambda) + ", "
            title += "Wymiarowość: " + str(self._dimension)
            # title += "Funkcja celu: " + str(self._fitness.__name__)
            plt.rcParams["figure.figsize"] = (7,7)
            plt.rcParams['font.size'] = '12'
            plt.tight_layout()
            plt.subplots_adjust(top = 0.95, bottom = 0.1, left = 0.1, right = 0.99)
            plt.title(title)

            x_ax = np.linspace(-5, 5, 100)
            y_ax = np.linspace(-5, 5, 100)
            xGrid, yGrid = np.meshgrid(x_ax, y_ax)

            if not self.contours_calculated:
                self.zs = []
                for index, criterium in enumerate(self._criteria):
                    newZ = criterium([xGrid, yGrid])
                    newZ = newZ - np.amin(newZ)
                    newZ = newZ / np.amax(newZ)
                    newZ = 10 * newZ
                    self.zs.append(newZ)
                self.contours_calculated = True

            for index, z in enumerate(self.zs):
                plt.contour(xGrid, yGrid, z, levels=list(np.array(list(range(30)))/3), colors=COLORS[index%COLOR_NUM], alpha=0.3)

            # plt.axis('equal')

            plt.axvline(0, linewidth=4, c='black')
            plt.axhline(0, linewidth=4, c='black')
            x1 = [point[-1] for point in population]
            x2 = [point[-2] for point in population]
            plt.scatter(x1, x2, s=50)
            x1 = [point[-1] for point in population[:self._mu]]
            x2 = [point[-2] for point in population[:self._mu]]
            plt.scatter(x1, x2, s=15)
            plt.scatter(self._xmean[-1], self._xmean[-2], s=100, c='black')
            plt.grid()
            # zoom_out = 1.3
            # max1 = zoom_out*max([abs(point[-1]) for point in population])
            # max2 = zoom_out*max([abs(point[-2]) for point in population])
            plt.xlim(-5, 5)
            plt.ylim(-5, 5)
            plt.pause(_DELAY)
            plt.clf()
            plt.cla()

        # Step-size control
        # C^(-1/2) = B D^(-1) B^T
        C_2 = np.matmul(np.matmul(self._B, np.diag(1 / (self._D + _EPS))), self._B.T)

        _path_sigma_delta = (np.sqrt(self._time_sigma * (2 - self._time_sigma) * self._mu) *
                        np.matmul(C_2, y_w))
        self._path_sigma = (1 - self._time_sigma) * self._path_sigma + _path_sigma_delta

        self._sigma *= np.exp( (self._time_sigma / (self._damping + _EPS)) *
                        (np.linalg.norm(self._path_sigma) / self._chi - 1))

        self._sigma = min(self._sigma, _SIGMA_MAX)

        # Covariance matrix adaption
        self._path_c = ((1 - self._time_c) * self._path_c +
                        np.sqrt(self._time_c * (2 - self._time_c) * self._mu) * y_w)

        # np.outer(v, v) <==> np.mat_mul(v, v.T)
        rank_one = np.outer(self._path_c, self._path_c)
        rank_mu = np.mean([np.outer(d, d) for d in selected], axis=0)
        self._C = (
                (1 - self._lr_c1 - self._lr_c_mu) * self._C
                + self._lr_c1 * rank_one
                + self._lr_c_mu * rank_mu
        )