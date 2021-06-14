from cmaes import CMAES
import numpy as np
import matplotlib.pyplot as plt

_TARGETS = np.array([10**i for i in range(-10, 10)])

def evaluate(dimensions: int, mode: str, frequency: int, iterations: int):
        ecdf_list = []
        print("Starting evaluation...")
        print("mode:", mode, "dimensions:", dimensions, ", frequency:", frequency, ", iterations:", iterations)
        for objective in ['quadratic', 'felli', 'bent']:
            print("    Currently running:", objective)
            for i in range(iterations):
                algo = CMAES(objective, dimensions, mode, frequency)
                algo.generation_loop()
                ecdf_list.append(algo.ecdf(_TARGETS))
        max_length = max([len(ecdf) for ecdf in ecdf_list])

        for ecdf in ecdf_list: #fill ecdf data with 1s so that all lists are of equal lengths
            missing_entries = max_length - len(ecdf)
            if missing_entries > 0:
                ecdf.extend([1.]*missing_entries)
        
        ecdf_result = []
        for i in range(max_length):
            ecdf_result.append(sum([ecdf[i] for ecdf in ecdf_list])/iterations)
        return ecdf_result

def frequencyTest(dimensions: int, iterations: int):
    ecdf1 = evaluate(dimensions, 'mean', 1, iterations)
    ecdf10 = evaluate(dimensions, 'mean', 5, iterations)
    ecdf100 = evaluate(dimensions, 'mean', 50, iterations)

    plt.plot(ecdf1, label = 'mean 1')
    plt.plot(ecdf10, label = 'mean 5')
    plt.plot(ecdf100, label = 'mean 50')
    plt.legend()
    plt.show()

def modificationTest(dimensions: int, iterations: int):
    ecdf_normal = evaluate(dimensions, 'normal', None, iterations)
    ecdf_mean = evaluate(dimensions, 'mean', 1, iterations)

    plt.plot(ecdf_normal, label = 'normal')
    plt.plot(ecdf_mean, label = 'mean 1')
    plt.legend()
    plt.show()