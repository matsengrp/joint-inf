from __future__ import unicode_literals
import matplotlib
matplotlib.use('SVG')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['figure.autolayout'] = True
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import argparse
import sys
import pickle
import time

from operator import mul
from multiprocessing import Pool
from scipy.optimize import differential_evolution, minimize, basinhopping

sns.set_style('white')
sns.set_style('ticks')

# ~~~~~~~~~
# global variables

FONT_SIZE = 20

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
        '--in-pkl-name',
        type=str,
        help='file name for where to save pkl',
        default=None,
    )
    parser.add_argument(
        '--out-pkl-name',
        type=str,
        help='file name for where to save pkl',
        default='output.pkl',
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
        '--param-index',
        type=int,
        default=2,
        choices=(0, 1, 2),
        help='which parameter to optimize; 0: x, 1: y, 2: w',
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
    parser.add_argument(
        '--output-likelihood',
        action="store_true",
        help='general likelihood pickle',
    )
    parser.add_argument(
        '--analytic',
        action="store_true",
        help='plot analytic',
    )
    parser.add_argument(
        '--empirical',
        action="store_true",
        help='compute empirical',
    )
    parser.add_argument(
        '--marginal',
        action="store_true",
        help='do marginal likelihood instead of joint inference',
    )

    args = parser.parse_args()

    return args

# ~~~~~~~~~
# global functions for exact likelihood computation

def P_INVFELS(theta0, site_pattern):
    """
    Site pattern frequencies for inverse Felsenstein tree
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

def L_INVFELS(theta, site_pattern, joint=True):
    """
    Likelihood for inverse Felsenstein topology
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
        return 0.03125 * max(s1, s2, s3, s4)
    else:
        return 0.03125 * sum([s1, s2, s3, s4])

def INVFELS_LIKELIHOOD(x, y, xp, yp, wp, joint=True):
    """
    Total likelihood as function of marginal likelihood
    """
    likelihood = 0.
    theta0 = [x, y, x, y, y]
    theta1 = [xp, yp, xp, yp, wp]
    for site_pattern in ALL_PATTERNS:
        if not np.isclose(P_INVFELS(theta0, site_pattern), 0):
            if np.isclose(P_INVFELS(theta1, site_pattern), 0) or np.isclose(L_INVFELS(theta1, site_pattern, joint=joint), 0):
                likelihood = -np.inf
                break

            likelihood += P_INVFELS(theta0, site_pattern) * \
                (np.log(L_INVFELS(theta1, site_pattern, joint=joint)) + \
                np.log(P_INVFELS(theta1, site_pattern)))

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
        return P_INVFELS([x, y, x, y, y], site_char)
    else:
        return P_INVFELS([x, y, x, y, y], SPLIT_TO_PATTERN_DICT[site_char])

# Upper bound for inverse Felsenstein tree topology
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
def invfels_upper_bound_1(x, y):
    return gen_prob('0', x, y)*np.log(2.) \
            + max_of_logs(a_const_1(x, y), a_const_2(x, y)) \
            + max_of_logs(a_const_3(x, y), a_const_4(x, y)) \
            + gen_prob('13', x, y)*np.log(2*gen_prob('13', x, y)) \
            + (1-gen_prob('13', x, y))*np.log(2*(1-gen_prob('13', x, y))) \
            - np.log(8.)

# L^{2}_{\tau_1,\tau_1}(\mathbf{p}_{xy})
def invfels_upper_bound_2(x, y):
    return gen_prob('0', x, y)*np.log(2.) \
            + max_of_logs(a_const_1_prime(x, y), a_const_2_prime(x, y)) \
            + max_of_logs(a_const_3(x, y), a_const_4(x, y)) \
            + np.log(2.) \
            - np.log(8.)

# L^{3}_{\tau_1,\tau_1}(\mathbf{p}_{xy})
def invfels_upper_bound_3(x, y):
    return gen_prob('0', x, y)*np.log(2.) \
            + max_of_logs(a_const_1(x, y), a_const_2(x, y)) \
            + max_of_logs(a_const_3_prime(x, y), a_const_4_prime(x, y)) \
            + np.log(2.) \
            - np.log(8.)

def invfels_upper_bound(x, y):
    return np.maximum(
        invfels_upper_bound_1(x, y),
        invfels_upper_bound_2(x, y),
        invfels_upper_bound_3(x, y)
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

    # a + b + c = -gamma**2 * (1+.5*beta) + 2*gamma*beta*x*y**2 + beta**2
    # for general case, gamma are fns of (x', y') and 2nd and 3rd betas are too
    #return -gamma**2 * (1+.5*beta) + 2*gamma*beta*x*y**2 + beta**2

    return (a + b + c)

def what_lower_general(x, y):
    beta = 1+x**2+y**2+x**2*y**2
    gamma = 4*x*y
    x_l = x-.1
    x_u = x+.1
    y_l = y-.1
    y_u = y+.1
    gamma_l = 4*x_l*y_l
    gamma_u = 4*x_u*y_u
    beta_l = 1+x_l**2+y_l**2+x_l**2*y_l**2

    return -gamma_u**2 * (1+.5*beta) + 2*gamma_l*beta_l*x*y**2 + beta_l**2

def max_likelihood(xy, delta, joint=True):
    """
    max likelihood as function of marginal likelihood
    """

    minimizer_kwargs = dict(method="L-BFGS-B", bounds=[(delta,1-delta)]*3)
    what_gen = basinhopping(lambda z: -INVFELS_LIKELIHOOD(xy[0], xy[1], z[0], z[1], z[2], joint=joint),
        x0=[xy[0], xy[1], xy[1]],
        minimizer_kwargs=minimizer_kwargs
    )
    minimizer_kwargs = dict(method="L-BFGS-B", bounds=[(delta,1-delta)]*2)
    what_res = basinhopping(lambda z: -INVFELS_LIKELIHOOD(xy[0], xy[1], z[0], z[1], 1.-delta, joint=joint),
        x0=[xy[0], xy[1]],
        minimizer_kwargs=minimizer_kwargs
    )
    if -what_res.fun > -what_gen.fun:
        what = 1.-delta
    else:
        what = what_gen.x[2]

    return what

class MaxLikelihood(object):
    def __init__(self, delta, joint=True):
        self.delta = delta
        self.joint = joint
    def __call__(self, xy):
        return max_likelihood(xy, self.delta, joint=self.joint)

class Legend(object):
    pass

class LegendHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle([x0, y0], width, height, facecolor='w',
                                   edgecolor='black', lw=1,
                                   transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch

def main(args=sys.argv[1:]):
    args = parse_args()

    st_time = time.time()
    x_range = np.arange(args.delta, 1.0, args.delta)
    y_range = np.arange(args.delta, 1.0, args.delta)
    X, Y = np.meshgrid(x_range,y_range)

    if args.analytic:
        if args.topology:
            region = invfels_upper_bound(X, Y) - felsenstein_lower_bound(X, Y)
            plottitle = r'Region of inconsistency for InvFels--generating topology'
            legendtext = r'joint inference inconsistent'
        elif args.restricted_branch_lengths:
            region = what_lower_minus_one(X, Y)
            plottitle = r'Region of inconsistent branch parameter estimation''\n'r'(restricted case)'
            legendtext = r'$\hat{w}=1$'
        elif args.general_branch_lengths:
            region = what_lower_general(X, Y)
            plottitle = r'Region of inconsistent branch parameter estimation''\n'r'(general case)'
            legendtext = r'$\hat{w}=1$ or $\hat{x}$ or $\hat{y}$''\n'r'poorly estimated'

        ct = plt.contour(X, Y, region, [0], colors='k', linewidths=1)
        plt.axis('scaled')
        plt.xlabel(r'$x^*$', fontsize=FONT_SIZE)
        plt.ylabel(r'$y^*$', fontsize=FONT_SIZE)
        ttl = plt.title(plottitle, fontsize=FONT_SIZE+2)
        ttl.set_position([.5, 1.05])
        ct.ax.tick_params(labelsize=FONT_SIZE-2)
        if not args.topology:
            vec = ct.collections[0].get_paths()[0].vertices
            x = np.concatenate(([1-args.delta], vec[:,0], [args.delta]))
            y = np.concatenate(([args.delta], vec[:,1], [1-args.delta]))
            plt.plot(x, y, '-', alpha=.15, lw=.5)

        if args.restricted_branch_lengths:
            plt.legend([Legend()], [legendtext],
                handler_map={Legend: LegendHandler()},
                bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=FONT_SIZE-4)
        else:
            plt.legend([Legend()], [legendtext],
                handler_map={Legend: LegendHandler()}, fontsize=FONT_SIZE-4)

        ax = plt.gca()
        ax.set_xticks(np.arange(0, 1.1, .2))
        ax.set_yticks(np.arange(0, 1.1, .2))
        ax.set_xticklabels([r'$0.0$', r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$', r'$1.0$'], fontsize=FONT_SIZE-2)
        ax.set_yticklabels([r'$0.0$', r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$', r'$1.0$'], fontsize=FONT_SIZE-2)
        sns.despine()
        plt.savefig(args.plot_name)
    elif args.empirical:
        if args.general_branch_lengths:
            if args.in_pkl_name is None:
                if args.n_jobs > 1:
                    p = Pool(processes=args.n_jobs)
                    Z_pool = p.map( MaxLikelihood(args.delta, joint=not args.marginal) , [(x, y) for x, y in zip(X.ravel(), Y.ravel())])
                    Z = np.reshape(Z_pool, (int(1. / args.delta) - 1, int(1. / args.delta) - 1))
                else:
                    Z = np.vectorize(lambda x, y: max_likelihood((x, y), delta=args.delta, joint=not args.marginal))(X, Y)

                with open(args.out_pkl_name, 'w') as f:
                    pickle.dump((X, Y, Z, args.delta), f)
            else:
                with open(args.in_pkl_name, 'r') as f:
                    X, Y, Z, args.delta = pickle.load(f)

            im = plt.imshow(Z, cmap=plt.cm.gray, origin='lower')
            plt.xlabel(r'$x^*$', fontsize=FONT_SIZE)
            plt.ylabel(r'$y^*$', fontsize=FONT_SIZE)
            ttl = plt.title(r'Value of $\hat{w}$', fontsize=FONT_SIZE+2)
            ttl.set_position([.5, 1.05])
            ax = plt.gca()
            ax.set_xticks(np.arange(0, 1/args.delta, .2/args.delta))
            ax.set_yticks(np.arange(0, 1/args.delta, .2/args.delta))
            ax.set_xticklabels([r'$0.0$', r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$'], fontsize=FONT_SIZE-2)
            ax.set_yticklabels([r'$0.0$', r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$'], fontsize=FONT_SIZE-2)
            sns.despine()
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=FONT_SIZE-2)
            plt.savefig(args.plot_name)
            plt.close()
            bias = Z - Y
            ones_mask = (Z != 1 - args.delta)
            print "Error range [%.0E, %.0E], Mean: %.2E" % (min(bias[ones_mask]), max(bias[ones_mask]), np.mean(bias[ones_mask]))

            # plot bias
            bias_to_plot = np.where(ones_mask, bias, .09)
            im = plt.imshow(bias_to_plot[int(.6/args.delta):, int(.6/args.delta):], cmap=plt.cm.gray, origin='lower')
            plt.xlabel(r'$x^*$', fontsize=FONT_SIZE)
            plt.ylabel(r'$y^*$', fontsize=FONT_SIZE)
            ttl = plt.title(r'Bias: $\hat{w}-y^*$', fontsize=FONT_SIZE+2)
            ttl.set_position([.5, 1.05])
            ax = plt.gca()
            ax.set_xticks(np.arange(0, .4/args.delta, .1/args.delta))
            ax.set_yticks(np.arange(0, .4/args.delta, .1/args.delta))
            ax.set_xticklabels([r'$0.6$', r'$0.7$', r'$0.8$', r'$0.9$'], fontsize=FONT_SIZE-2)
            ax.set_yticklabels([r'$0.6$', r'$0.7$', r'$0.8$', r'$0.9$'], fontsize=FONT_SIZE-2)
            sns.despine()
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=FONT_SIZE-2)
            plt.savefig(args.plot_name.replace('.svg', '_bias.svg'))
    else:
        print "No plotting argument given!"

    print "Completed! Time: %s" % str(time.time() - st_time)


if __name__ == "__main__":
    main(sys.argv[1:])
