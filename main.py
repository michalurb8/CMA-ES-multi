import argparse
from functions import criteria
from cmaes import CMAES

parser = argparse.ArgumentParser(prog="CMA-ES",
                                 description='This program allows you to run CMA-ES for multi-objective optimization')

parser.add_argument('-i', '--iterations', type=int, default=10,
                    help='How many algorithm runs to be averaged.')

parser.add_argument('-d', '--dimensions', type=int, default=2,
                    help='Number of dimensions.')

parser.add_argument('-l', '--lbd', type=int, default=None,
                    help='Population size.')

parser.add_argument('-s', '--stop', type=int, default=150,
                    help='How many iterations to take average of.')

parser.add_argument('-v', '--vis', default=True,
                    help='Turn off visualisation.', action='store_false')

if __name__ == '__main__':
    args = parser.parse_args()
    for _ in range(args.iterations):
        algo = CMAES(criteria, args.dimensions, args.stop, args.lbd, args.vis)
