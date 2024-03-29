#NOT USED RIGHT NOW

#NOT USED RIGHT NOW

#NOT USED RIGHT NOW

#NOT USED RIGHT NOW

#NOT USED RIGHT NOW

#NOT USED RIGHT NOW

#NOT USED RIGHT NOW

#NOT USED RIGHT NOW

#NOT USED RIGHT NOW

#NOT USED RIGHT NOW

#NOT USED RIGHT NOW

#NOT USED RIGHT NOW

#NOT USED RIGHT NOW

#NOT USED RIGHT NOW

#NOT USED RIGHT NOW

from cmaes import CMAES
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sys import stdout

_TARGETS = np.array([10 ** i for i in range(-10, 10)])

def evaluate(dimensions: int, iterations: int, objectives: List, lambda_arg: int, stop_after: int, visual: bool):
    """
    evaluate() runs the algorithm multiple times (exactly 'iteration' times).
    Data about ecdf, sigma, condition number etc. is collected.
    Then, data is averaged across different iterations and returned as a tuple of lists, each list ready to be plotted.
    Parameters
    ----------
    dimensions : int
        Objective function dimensionality.
    iterations : int
        Number of algorithm runs to take an average of.
    objectives: List[str]
        List of objective functions. For each function, the algorithm will run 'iteration' times,
        then an average of all runs for all objective functions will be computed.
        Chosen from: quadratic, elliptic, bent, rastrigin, rosenbrock, ackley
    lambda_arg : int
        Population count. Must be > 3, if set to None, default value will be computed.
    stop_after: int
        How many iterations are to be run
    visual: bool
        If True, every algorithm generation will be visualised (only 2 first dimensions)
    ----------
    # run the algorithm multiple times, return averaged results: ECDF values, sigma values, sigma difference values
    """
    ecdfs_list = []
    sigmas_list = []
    diffs_list = []
    eigens_list = []
    cond_list = []
    mean_list = []
    evals_per_gen = None
    print("Starting evaluation...")
    lambda_prompt = str(lambda_arg) if lambda_arg is not None else "default"
    print(f"dimensions: {dimensions}; iterations: {iterations}; population: {lambda_prompt}")
    for objective in objectives:
        print("    Currently running:", objective.__name__)
        for iteration in range(iterations):
            stdout.write(f"\rIteration: {1+iteration} / {iterations}")
            stdout.flush()
            algo = CMAES(objective, dimensions, stop_after, lambda_arg, visual) # algorithm runs here
            if evals_per_gen == None:
                evals_per_gen = algo.evals_per_iteration()
            else:
                assert evals_per_gen == algo.evals_per_iteration(), "Lambda different for same settings"
            ecdfs_list.append(algo.ecdf_history(_TARGETS))
            sigmas_list.append(algo.sigma_history())
            diffs_list.append(algo.diff_history())
            eigens_list.append(algo.eigen_history())
            cond_list.append(algo.cond_history())
            mean_list.append(algo.mean_history())
        print()

    formatted_ecdfs = _format_list(ecdfs_list, evals_per_gen)
    formatted_sigmas = _format_list(sigmas_list, evals_per_gen)
    formatted_diffs = _format_list(diffs_list, evals_per_gen)
    formatted_conds = _format_list(cond_list, evals_per_gen)
    formatted_means = _format_list(mean_list, evals_per_gen)
    formatted_eigens = _format_eigenvalues(eigens_list, evals_per_gen)
    return (formatted_ecdfs, formatted_sigmas, formatted_diffs, formatted_eigens, formatted_conds, formatted_means, evals_per_gen)


def _format_list(input_list: List, evals_per_gen: int) -> Tuple:
    """
    _format_list() takes as input data collected from multiple algorithm runs.
    Then, an average is computed and horizontal axis scaling is applied.
    The return value is a tuple of two list, each corresponds to a plot axis, ready to be plotted.
    """
    run_count = len(input_list)
    run_length = len(input_list[0])
    for i in input_list:
        assert len(i) == run_length, "Runs are of different length, cannot take average"
    y_axis = []
    for i in range(run_length):
        y_axis.append(sum([sigmas[i] for sigmas in input_list]) / run_count)
    x_axis = [x*evals_per_gen for x in range(run_length)]
    return x_axis, y_axis


def _format_eigenvalues(eigens_list: List, evals_per_gen: int) -> Tuple:
    """
    _format_eigenvalues() takes iter matrices of size gen x dim.
    Each matrix is returned by a single algorithm run. Each row corresponds to an algorithm generation, each row to an eigenvalue.
    The average of matrices is computed, then each column is exported as a list to be plotted. One x-axis is created for all plots.
    The return value is a tuple of two lists. The first one represents the x-axis.
    The second is a list of other axes, each being a separate list, like the x-axis.
    """
    (gen1, dim1) = eigens_list[0].shape
    iterations = len(eigens_list)
    sum = np.zeros((gen1, dim1))
    for eigens_matrix in eigens_list:
        assert  eigens_matrix.shape == (gen1, dim1), "Runs are of different length, cannot take average of sigma"
        sum += eigens_matrix
    sum /= iterations
    x_axis = [x*evals_per_gen for x in range(gen1)]
    other_axes = []
    for i in range(dim1):
        other_axes.append(sum[:,i])
    return x_axis, other_axes

def just_show(dimensions: int, iterations: int, lbd: int, stop_after: int, visual: bool, objective):
    for _ in range(iterations):
        algo = CMAES(objective, dimensions, stop_after, lbd, visuals=visual)

def run_test(dimensions: int, iterations: int, lbd: int, stop_after: int, visual: bool, objectives: List[str]):
    """
    run_test()) is the main function. It runs the evaluate function for all the algorithm varianst.
    All of them are plotted separately to be compared.
    """
    ecdf_plots = []
    sigma_plots = []
    diff_plots = []
    eigen_plots = []
    cond_plots = []
    mean_plots = []

    ecdf, sigma, diff, eigen, cond, mean, lambda_val = evaluate(dimensions, iterations, objectives, lbd, stop_after, visual)
    legend = "This is a plot"

    ecdf_plots.append((ecdf[0], ecdf[1], legend))
    sigma_plots.append((sigma[0], sigma[1], legend))
    diff_plots.append((diff[0], diff[1], legend))
    eigen_plots.append((eigen[0], eigen[1], legend))
    cond_plots.append((cond[0], cond[1], legend))
    mean_plots.append((mean[0], mean[1], legend))

    lambda_prompt = str(lbd) if lbd is not None else "Domyślnie 4n=" + str(lambda_val)
    title_str = f"Wymiarowość: {dimensions}; Liczebność populacji: {lambda_prompt};\nLiczba iteracji: {stop_after}; Liczba przebiegów: {iterations}; Funkcja celu: {objectives[0].__name__}"; 
    plt.rcParams['font.size'] = '18'
    ecdf_ax = plt.subplot(411)
    plt.title(title_str)
    plt.setp(ecdf_ax.get_xticklabels(), visible = False)
    for ecdf_plot in ecdf_plots:
        plt.plot(ecdf_plot[0], ecdf_plot[1], label=ecdf_plot[2])
    plt.legend()
    plt.ylabel("ECDF", rotation=45, horizontalalignment="right", verticalalignment="center")
    plt.ylim(0,1)
    
    sigma_ax = plt.subplot(412, sharex=ecdf_ax)
    plt.setp(sigma_ax.get_xticklabels(), visible = False)
    for sigma in sigma_plots:
        plt.plot(sigma[0], sigma[1], label=sigma[2])
    plt.yscale("log")
    plt.ylabel("Wartości sigma", rotation=45, horizontalalignment="right")

    diff_ax = plt.subplot(413, sharex=ecdf_ax)
    plt.setp(diff_ax.get_xticklabels(), visible = False)
    for diff in diff_plots:
        plt.plot(diff[0], diff[1], label=diff[2])
    plt.ylabel("Ilorazy kolejnych sigma", rotation=45, horizontalalignment="right")

    cond_ax = plt.subplot(414, sharex=ecdf_ax)
    plt.setp(cond_ax.get_xticklabels(), visible = False)
    for cond in cond_plots:
        plt.plot(cond[0], cond[1], label=cond[2])
    plt.ylabel("Wskaźnik uwarunkowania C", rotation=45, horizontalalignment="right")

    plt.show()

if __name__ == "__main__":
    run_test()