import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

from operator import mul
from multiprocessing import Pool
from scipy.optimize import differential_evolution

# ~~~~~~~~~
# global variables

ALL_PATTERNS = [
        '0000',
        '1000',
        '0100',
        '0010',
        '1110',
        '1100',
        '1010',
        '0110'
]

SPLIT_TO_PATTERN_DICT = {
        '0':  '0000',
        '1':  '1000',
        '2':  '0100',
        '3':  '0010',
        '123':'1110',
        '12': '1100',
        '13': '1010',
        '23': '0110'
}

# ~~~~~~~~~

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '--plot-name',
        type=str,
        help='file name for where to save plot',
        default='temporary-plot.svg',
    )
    parser.add_argument(
        '--delta',
        type=float,
        help='coarseness parameter for plotting',
        default=0.01,
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        help='number of processes to use for empirically computing maxima',
        default=1,
    )
    parser.add_argument(
        '--topology',
        action="store_true",
        help='topology plot',
    )
    parser.add_argument(
        '--restricted-branch-lengths',
        action="store_true",
        help='restricted branch lengths plot',
    )
    parser.add_argument(
        '--general-branch-lengths',
        action="store_true",
        help='general branch lengths plot',
    )

    args = parser.parse_args()

    return args

# ~~~~~~~~~
# global functions for exact likelihood computation

def P_FARRIS(theta0, site_pattern):
    """
    Site pattern frequencies for Farris tree
    """
    # fifth branch will always be positive
    theta = [-t if s=='1' else t \
            for t,s in zip(theta0, site_pattern+'0')]
    return 0.125 * (1 + \
         theta[0]*theta[2] + \
         theta[1]*theta[3] + \
         theta[0]*theta[1]*theta[4] + \
         theta[0]*theta[3]*theta[4] + \
         theta[1]*theta[2]*theta[4] + \
         theta[2]*theta[3]*theta[4] + \
         theta[0]*theta[1]*theta[2]*theta[3])

def L_FARRIS(theta, site_pattern, joint=True):
    """
    Likelihood for Farris zone topology
    """
    s1 = reduce(
        mul,
        [(1+t) if s=='0' else (1-t) for t,s in zip(theta, site_pattern+'0')],
        1
    )
    s2 = reduce(
        mul,
        [(1+t) if s=='0' and i % 2 or s=='1' and not i % 2 else (1-t) \
            for i,(t,s) in enumerate(zip(theta, site_pattern+'0'))],
        1
    )
    s3 = reduce(
        mul,
        [(1+t) if s=='1' and i % 2 or s=='0' and not i % 2 else (1-t) \
            for i,(t,s) in enumerate(zip(theta, site_pattern+'1'))],
        1
    )
    s4 = reduce(
        mul,
        [(1+t) if s=='1' else (1-t) for t,s in zip(theta, site_pattern+'1')],
        1
    )
    if joint:
        return 0.125 * max(s1, s2, s3, s4)
    else:
        return 0.125 * sum([s1, s2, s3, s4])

def FARRIS_LIKELIHOOD(x, y, xp, yp, wp, joint=True):
    """
    Total likelihood as function of marginal likelihood
    """
    likelihood = 0.
    theta0 = [x, y, x, y, y]
    theta1 = [xp, yp, xp, yp, wp]
    for site_pattern in ALL_PATTERNS:
        likelihood += P_FARRIS(theta0, site_pattern) * \
                (np.log(L_FARRIS(theta1, site_pattern, joint=joint)) + \
                np.log(P_FARRIS(theta1, site_pattern)))
    return likelihood

def P_FELSENSTEIN(theta0, site_pattern):
    """
    Site pattern frequencies for Felsenstein tree
    """
    # fifth branch will always be positive
    theta = [-t if s=='1' else t \
            for t,s in zip(theta0, site_pattern+'0')]
    return 0.125 * (1 + \
         theta[0]*theta[1] + \
         theta[2]*theta[3] + \
         theta[0]*theta[2]*theta[4] + \
         theta[1]*theta[2]*theta[4] + \
         theta[0]*theta[3]*theta[4] + \
         theta[1]*theta[3]*theta[4] + \
         theta[0]*theta[1]*theta[2]*theta[3])

# ~~~~~~~~~
# functions derived from manuscript

# Generating probabilities
# Left column of tab:sitepatprob
def gen_prob(site_char, x, y, no_split=False):
    if no_split:
        return P_FARRIS([x, y, x, y, y], site_char)
    else:
        return P_FARRIS([x, y, x, y, y], SPLIT_TO_PATTERN_DICT[site_char])

# Upper bound for Farris tree topology
# Constants from eq:a_const
def sum_of_four(split_list, x, y):
    return sum([gen_prob(split, x, y) for split in split_list])

def a_const_1(x, y):
    return 3. - sum_of_four(['1', '3', '12', '23'], x, y)

def a_const_2(x, y):
    return 2. * sum_of_four(['1', '3', '12', '23'], x, y)

def a_const_3(x, y):
    return 3. - sum_of_four(['2', '12', '23', '123'], x, y)

def a_const_4(x, y):
    return 2. * sum_of_four(['2', '12', '23', '123'], x, y)

# Constants from eq:a_const_prime
def a_const_1_prime(x, y):
    return a_const_1(x, y) - 2.*gen_prob('13', x, y)

def a_const_2_prime(x, y):
    return a_const_2(x, y) + 2.*gen_prob('13', x, y)

def a_const_3_prime(x, y):
    return a_const_3(x, y) - 2.*gen_prob('13', x, y)

def a_const_4_prime(x, y):
    return a_const_4(x, y) + 2.*gen_prob('13', x, y)

def max_of_logs(a1, a2):
    """
    the value of max { a1 * log(1 + x) + a2 * log(1 - x), i.e.,
        a1 * log((2*a1/(a1+a2))) + a2 * log((2*a2/(a1+a2)))
    """
    return a1 * np.log((2*a1/(a1+a2))) + a2 * np.log((2*a2/(a1+a2)))

# L^{1}_{\tau_1,\tau_1}(\mathbf{p}_{xy})
def farris_upper_bound_1(x, y):
    return gen_prob('0', x, y)*np.log(2.) \
            + max_of_logs(a_const_1(x, y), a_const_2(x, y)) \
            + max_of_logs(a_const_3(x, y), a_const_4(x, y)) \
            + gen_prob('13', x, y)*np.log(2*gen_prob('13', x, y)) \
            + (1-gen_prob('13', x, y))*np.log(2*(1-gen_prob('13', x, y))) \
            - np.log(8.)

# L^{2}_{\tau_1,\tau_1}(\mathbf{p}_{xy})
def farris_upper_bound_2(x, y):
    return gen_prob('0', x, y)*np.log(2.) \
            + max_of_logs(a_const_1_prime(x, y), a_const_2_prime(x, y)) \
            + max_of_logs(a_const_3(x, y), a_const_4(x, y)) \
            + np.log(2.) \
            - np.log(8.)

# L^{3}_{\tau_1,\tau_1}(\mathbf{p}_{xy})
def farris_upper_bound_3(x, y):
    return gen_prob('0', x, y)*np.log(2.) \
            + max_of_logs(a_const_1(x, y), a_const_2(x, y)) \
            + max_of_logs(a_const_3_prime(x, y), a_const_4_prime(x, y)) \
            + np.log(2.) \
            - np.log(8.)

def farris_upper_bound(x, y):
    return np.maximum(
        farris_upper_bound_1(x, y),
        farris_upper_bound_2(x, y),
        farris_upper_bound_3(x, y)
    )

# Lower bound for the Felsenstein tree topology
def b_const(x, y):
    return sum_of_four(['1', '2', '3', '123'], x, y)

def felsenstein_lower_bound(x, y):
    x_max = 1.
    w_max = 0.
    def y_max(u, v):
        return 1. - b_const(u, v)

    shannon_felsenstein = sum([gen_prob(pattern, x, y, no_split=True) * np.log(P_FELSENSTEIN([x_max, y_max(x, y), x_max, y_max(x, y), w_max], pattern)) for pattern in ALL_PATTERNS])
    partial_felsenstein = 2*np.log(2.)+(2-b_const(x, y))*np.log(2-b_const(x, y))+b_const(x, y)*np.log(b_const(x, y))
    return shannon_felsenstein + partial_felsenstein

def what_lower_minus_one(x, y):
    # Want cases where what_2 is greater than one for x and y between 0 and 1
    alpha_1 = 1./8 * (1+x**2+y**2+4*x*y**2+x**2*y**2)
    alpha_2 = 1./8 * (1+x**2+y**2-4*x*y**2+x**2*y**2)
    beta =    1+x**2+y**2+x**2*y**2
    gamma = 4*x*y

    a = -gamma**2*alpha_1 - gamma**2*alpha_2 - gamma**2
    b = gamma*alpha_1*beta - gamma**2*alpha_1 - gamma*alpha_2*beta - gamma**2*alpha_2
    c = gamma*alpha_1*beta - gamma*alpha_2*beta + beta**2

    what_lower = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)

    return (what_lower - 1)

def max_likelihood(xy):
    """
    differential evolution to find maximum likelihood estimate of interior branch length
    """
    bnds = [(0,1), (0,1), (0,1)]
    what = differential_evolution(
        lambda z: -FARRIS_LIKELIHOOD(xy[0], xy[1], z[0], z[1], z[2]),
        bounds=bnds
    )
    return what.x[2]

def main(args=sys.argv[1:]):
    args = parse_args()

    x_range = np.arange(args.delta, 1.0, args.delta)
    y_range = np.arange(args.delta, 1.0, args.delta)
    X, Y = np.meshgrid(x_range,y_range)

    if args.topology:
        region = farris_upper_bound(X, Y) - felsenstein_lower_bound(X, Y)
        plt.contour(X, Y, region, [0])
        plt.xlabel(r'x', fontsize=16)
        plt.ylabel(r'y', fontsize=16)
        plt.title(r'Region of inconsistency for Farris-generating topology')
        plt.savefig(args.plot_name)
    elif args.restricted_branch_lengths:
        region = what_lower_minus_one(X, Y)
        plt.contour(X, Y, region, [0])
        plt.xlabel(r'x', fontsize=16)
        plt.ylabel(r'y', fontsize=16)
        plt.title(r'Region of inconsistent branch length estimation (restricted case)')
        plt.savefig(args.plot_name)
    elif args.general_branch_lengths:
        if args.n_jobs > 1:
            p = Pool(processes=8)
            ZZ = p.map( max_likelihood , [(x, y) for x, y in zip(X.ravel(), Y.ravel())])
            Z = np.reshape(ZZ, (int(1. / args.delta) - 1, int(1. / args.delta) - 1))
        else:
            Z = np.vectorize(lambda x, y: max_likelihood((x, y)))(X, Y)

        im = plt.imshow(Z, cmap=plt.cm.gray, origin='lower')
        plt.colorbar()
        plt.xlabel(r'x', fontsize=16)
        plt.ylabel(r'y', fontsize=16)
        plt.title(r'Value of $\hat{w}$')
        ax = plt.gca()
        ax.set_xticks(np.arange(0, 1/args.delta, .2/args.delta))
        ax.set_yticks(np.arange(0, 1/args.delta, .2/args.delta))
        ax.set_xticklabels(np.arange(0, 1, .2))
        ax.set_yticklabels(np.arange(0, 1, .2))
        plt.savefig(args.plot_name)
    else:
        print "No plotting argument given!"
        print "Options: --topology, --restricted-branch-lengths, --general-branch-lengths"

if __name__ == "__main__":
    main(sys.argv[1:])
