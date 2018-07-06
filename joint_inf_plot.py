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
import time
import pickle

from multiprocessing import Pool
from scipy.optimize import minimize
from itertools import product

sns.set_style('white')
sns.set_style('ticks')

# ~~~~~~~~~
# global variables

FONT_SIZE = 20

# constants to slightly speed up computation
LOG2 = np.log(2)
LOG32 = np.log(32)

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
        '--ancestral-state-conditions',
        action="store_true",
        help='plot analytic',
    )
    parser.add_argument(
        '--empirical-parameter-estimate',
        action="store_true",
        help='compute empirical',
    )
    parser.add_argument(
        '--marginal',
        action="store_true",
        help='compute marginal',
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
        '--analytic-inconsistency',
        action="store_true",
    )

    args = parser.parse_args()

    return args

# ~~~~~~~~~
# Functions for exact likelihood bounds


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

# ~~~~~~~~~
# Functions for empirical plot

def safe_log(x):
    return np.log(x) if x > 0. else -np.inf

def safe_p_logq(p, logq):
    return p * logq if p > 0. else 0.

def marginal_likelihood(theta, theta_hat):
    """
    Table S4 in manuscript: we have 81 possible likelihood functions we can compute
    """
    probs = {pattern: P_INVFELS(theta, pattern) for pattern in ALL_PATTERNS}
    fit_probs = {pattern: P_INVFELS(theta_hat, pattern) for pattern in ALL_PATTERNS}

    likelihood = np.sum([safe_p_logq(p_gen, safe_log(fit_probs[pattern])) for pattern, p_gen in probs.iteritems()])
    likelihood -= LOG32
    likelihood += get_partial_likelihood(probs, theta_hat)
    return likelihood

def get_partial_likelihood(probs, theta):
    all_terms = []
    likelihood = 0.
    for t in theta:
        all_terms.append([1+t, 1-t])

    anc_state_dict = {'0': (0,0), '1': (1,0), '2': (1,1), '3': (0,1)}
    for pattern in ALL_PATTERNS:
        curr_sum = 0.
        for curr_anc_state in '0123':
            curr_prod = 1.
            anc_state = anc_state_dict[curr_anc_state]
            curr_prod *= all_terms[4][int(curr_anc_state) % 2]
            for idx, sub_pattern in enumerate(pattern):
                curr_prod *= all_terms[idx][(anc_state[int(idx % 2)]+int(sub_pattern)) % 2]
            curr_sum += curr_prod
        likelihood += safe_p_logq(probs[pattern], safe_log(curr_sum))

    return likelihood

def get_log_coefficients(probs, anc_states):
    all_terms = []
    for _ in range(5):
        all_terms.append([0., 0.])

    # unambiguous ancestral states
    unamb_patterns = ['0000', '1000', '0100', '1100']
    for pattern in unamb_patterns:
        all_terms[4][0] += probs[pattern]
        for idx, sub_pattern in enumerate(pattern):
            all_terms[idx][int(sub_pattern)] += probs[pattern]

    # ambiguous ancestral states
    anc_state_dict = {'0': (0,0), '1': (1,0), '2': (1,1)}
    amb_patterns = ['0010', '1110', '1010', '0110']
    for curr_anc_state, pattern in zip(anc_states, amb_patterns):
        amb_anc_state = anc_state_dict[curr_anc_state]
        all_terms[4][int(curr_anc_state) % 2] += probs[pattern]
        for idx, sub_pattern in enumerate(pattern):
            all_terms[idx][(amb_anc_state[int(idx % 2)]+int(sub_pattern)) % 2] += probs[pattern]

    return all_terms

def likelihood_lower_bound(theta):
    """
    Table S4 but with w=1 and y_2=1 (and then x_1, y_1, x_2 their optima)
    """
    probs = {pattern: P_INVFELS(theta, pattern) for pattern in ALL_PATTERNS}
    all_terms = get_log_coefficients(probs, '0000')
    theta_hat = [theta[0]*theta[1]*theta[1], theta[1]*theta[1], theta[0]*theta[1]*theta[1], 1., 1.]
    fit_probs = {pattern: P_INVFELS(theta_hat, pattern) for pattern in ALL_PATTERNS}

    likelihood = np.sum([safe_p_logq(p_gen, safe_log(fit_probs[pattern])) for pattern, p_gen in probs.iteritems()])
    likelihood -= LOG32
    for term, est in zip(all_terms, theta_hat):
        likelihood += safe_p_logq(term[0], safe_log(1+est))
        likelihood += safe_p_logq(term[1], safe_log(1-est))
    return likelihood

def likelihood_upper_bound(theta, anc_states='0000'):
    """
    maximized at t=(a1-a2)/(a1+a2), but note a1+a2=1, so a1log(1+a1-a2) + a2log(1+a2-a1) is max
    """
    probs = {pattern: P_INVFELS(theta, pattern) for pattern in ALL_PATTERNS}
    all_terms = get_log_coefficients(probs, anc_states)
    likelihood = np.sum([safe_p_logq(p_gen, safe_log(p_gen)) for pattern, p_gen in probs.iteritems()])
    likelihood -= LOG32
    for term in all_terms:
        likelihood += safe_p_logq(term[0], safe_log(term[0]))
        likelihood += safe_p_logq(term[1], safe_log(term[1]))
        likelihood += LOG2
    return likelihood

def likelihood(theta, theta_hat, anc_states='0000'):
    """
    Table S4 in manuscript: we have 81 possible likelihood functions we can compute
    """
    probs = {pattern: P_INVFELS(theta, pattern) for pattern in ALL_PATTERNS}
    fit_probs = {pattern: P_INVFELS(theta_hat, pattern) for pattern in ALL_PATTERNS}
    all_terms = get_log_coefficients(probs, anc_states)

    likelihood = np.sum([safe_p_logq(p_gen, safe_log(fit_probs[pattern])) for pattern, p_gen in probs.iteritems()])
    likelihood -= LOG32
    for term, est in zip(all_terms, theta_hat):
        likelihood += safe_p_logq(term[0], safe_log(1+est))
        likelihood += safe_p_logq(term[1], safe_log(1-est))
    return likelihood

def get_ancestral_state_conds(xy):
    """
    """
    if xy[0] == 1. or xy[0] == 0. or xy[1] == 1. or xy[1] == 0.:
        return -np.inf
    theta = [xy[0], xy[1], xy[0], xy[1], xy[1]]
    bnd = likelihood_lower_bound(theta)
    output = []
    for anc_state in product('012', repeat=4):
        if ''.join(anc_state) == '0000':
            continue
        output.append(likelihood_upper_bound(theta, anc_state))
    return bnd - max(output)

def get_analytic_inconsistency(xy):
    """
    """
    x = xy[0]
    y = xy[1]
    if x == 1. or x == 0. or y == 1. or y == 0.:
        return np.nan
    theta = [x, y, x, y, y]
    cx = (1+x*x) / (2.*x)
    cy = (1-y)
    cxcy = cx*cy
    p13 = P_INVFELS(theta, '1010')
    p2 = P_INVFELS(theta, '0100')
    cond1 = cxcy >= 1
    cond2 = 2 - cxcy - cx + p13*(3-cxcy) + p2*(3-cx) <= 0
    cond3 = 1 + cx*cxcy - cxcy - cx + p13*(2-2*cxcy) + p2*(2-2*cx) >= 0
    bnd = likelihood_lower_bound(theta)
    output = []
    for anc_state in product('012', repeat=4):
        if ''.join(anc_state) == '0000':
            continue
        output.append(likelihood_upper_bound(theta, anc_state))
    if bnd >= max(output) and cond1 and cond2 and cond3:
        return -1.
    else:
        return 1.

def get_marginal_what(xy):
    """
    """
    # if we're in an edge case, return .25 to gray out plot
    if xy[0] == 1. or xy[0] == 0. or xy[1] == 1. or xy[1] == 0.:
        return np.nan
    theta = [xy[0], xy[1], xy[0], xy[1], xy[1]]
    max_obj = -np.inf
    for init_theta in [theta, [0.]*5, [1.]*5, [.5]*5]:
        output = minimize(lambda z: -marginal_likelihood(theta, z),
            x0=init_theta,
            method="L-BFGS-B",
            bounds=[(0.,1.)]*5,
        )
        if -output.fun > max_obj:
            max_obj = -output.fun
            out_obj = output
    what = out_obj.x[-1]
    return (1 - what) / 2

def get_empirical_what(xy):
    """
    """
    # if we're in an edge case, return .25 to gray out plot
    if xy[0] == 1. or xy[0] == 0. or xy[1] == 1. or xy[1] == 0.:
        return np.nan
    theta = [xy[0], xy[1], xy[0], xy[1], xy[1]]
    objs = []
    params = []
    for anc_state in product('012', repeat=4):
        max_obj = -np.inf
        for init_theta in [theta, [0.]*5, [1.]*5, [.5]*5]:
            output = minimize(lambda z: -likelihood(theta, z, anc_state),
                x0=init_theta,
                method="L-BFGS-B",
                bounds=[(0.,1.)]*5,
            )
            if -output.fun > max_obj:
                max_obj = -output.fun
                out_obj = output
        objs.append(-out_obj.fun)
        params.append(out_obj.x[-1])
    max_out = max(objs)
    what = params[objs.index(max_out)]
    return (1 - what) / 2

def obj_fn(xy, obj_type='get_ancestral_state_conds'):
    """
    """
    if obj_type == 'get_empirical_what':
        return get_empirical_what(xy)
    elif obj_type == 'get_ancestral_state_conds':
        return get_ancestral_state_conds(xy)
    elif obj_type == 'get_marginal_what':
        return get_marginal_what(xy)
    elif obj_type == 'get_analytic_inconsistency':
        return get_analytic_inconsistency(xy)

def make_plot(plottitle, plotname, legendtext=None, ct=None, scale=1, plotbar=False, move_legend=False):
    """
    """
    plt.axis('scaled')
    plt.xlabel(r'$p_{x^*}$', fontsize=FONT_SIZE-4)
    plt.ylabel(r'$p_{y^*}$', fontsize=FONT_SIZE-4)
    ttl = plt.title(plottitle, fontsize=FONT_SIZE+2)
    ttl.set_position([.5, 1.05])
    if ct is not None:
        ct.ax.tick_params(labelsize=FONT_SIZE-2)
        x = np.concatenate(([0.], [.5], [.5]))
        y = np.concatenate(([.5], [.5], [0.]))
        plt.plot(x, y, '-', alpha=.15, lw=.5)
    if legendtext is not None:
        if move_legend:
            plt.legend([Legend()], [legendtext],
                handler_map={Legend: LegendHandler()},
                bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=FONT_SIZE-4)
        else:
            plt.legend([Legend()], [legendtext], loc=2,
                handler_map={Legend: LegendHandler()}, fontsize=FONT_SIZE-4)

    ax = plt.gca()
    ax.set_xticks(np.arange(0., 0.5/scale+.1, .1/scale))
    ax.set_yticks(np.arange(0., 0.5/scale+.1, .1/scale))
    ax.set_xticklabels([r'$0.0$', r'$0.1$', r'$0.2$', r'$0.3$', r'$0.4$', r'$0.5$'], fontsize=FONT_SIZE-2)
    ax.set_yticklabels([r'$0.0$', r'$0.1$', r'$0.2$', r'$0.3$', r'$0.4$', r'$0.5$'], fontsize=FONT_SIZE-2)
    sns.despine()
    if plotbar:
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=FONT_SIZE-2)
    plt.savefig(plotname)
    plt.close()

class Obj(object):
    def __init__(self, obj_type):
        self.obj_type = obj_type
    def __call__(self, xy):
        return obj_fn(xy, self.obj_type)

def main(args=sys.argv[1:]):
    args = parse_args()

    st_time = time.time()
    px_range = np.arange(0., 0.5+args.delta, args.delta)
    py_range = np.arange(0., 0.5+args.delta, args.delta)
    PX, PY = np.meshgrid(px_range,py_range)
    X = 1-2*PX
    Y = 1-2*PY

    plotbar = False
    move_legend = False
    scale = 1
    legendtext = None
    if args.ancestral_state_conditions:
        plottitle = r'Region where $\emptyset$ is maximal ancestral state split'
        legendtext = r'$\emptyset$ is maximal'
        obj_type = 'get_ancestral_state_conds'
        move_legend = True
    elif args.empirical_parameter_estimate:
        plottitle = r'Estimated $\hat{p}_w$'
        obj_type = 'get_empirical_what'
        scale = args.delta
        plotbar = True
    elif args.marginal:
        plottitle = r'Estimated $\hat{p}_w$ (marginal inference)'
        obj_type = 'get_marginal_what'
        scale = args.delta
        plotbar = True
    elif args.analytic_inconsistency:
        plottitle = r'Region of inconsistency'
        legendtext = r'joint inference inconsistent'
        obj_type = 'get_analytic_inconsistency'

    if args.in_pkl_name is None:
        if args.n_jobs > 1:
            p = Pool(processes=args.n_jobs)
            Z_pool = p.map( Obj(obj_type) , [(x, y) for x, y in zip(X.ravel(), Y.ravel())])
            Z = np.reshape(Z_pool, (int(.5 / args.delta)+1, int(.5 / args.delta)+1))
        else:
            Z = np.vectorize(lambda x, y: obj_fn((x, y), obj_type=obj_type))(X, Y)

        with open(args.out_pkl_name, 'w') as f:
            pickle.dump((X, Y, Z, args.delta), f)
    else:
        with open(args.in_pkl_name, 'r') as f:
            X, Y, Z, args.delta = pickle.load(f)

    if not args.empirical_parameter_estimate and not args.marginal:
        ct = plt.contour(PX, PY, Z, [0], colors='k', linewidths=1)
    else:
        plt.imshow(Z, cmap=plt.cm.gray_r, origin='lower')
        ct = None

    make_plot(plottitle, args.plot_name, legendtext=legendtext, ct=ct, scale=scale, plotbar=plotbar)

    if args.empirical_parameter_estimate or args.marginal:
        # plot bias
        bias = Z - PY
        nan_mask = ~np.isnan(Z)
        print "Error range [%.0E, %.0E], Mean: %.2E" % (min(bias[nan_mask]), max(bias[nan_mask]), np.mean(bias[nan_mask]))
        small_mask = np.logical_and(~np.isnan(Z), np.logical_and(PY <= .1, PX <= .1))
        print "Error range (p < .1) [%.0E, %.0E], Mean: %.2E" % (min(bias[small_mask]), max(bias[small_mask]), np.mean(bias[small_mask]))

        plt.imshow(bias[0:int(.1/args.delta)+1, 0:int(.1/args.delta)+1], cmap=plt.cm.gray_r, origin='lower')
        plt.xlabel(r'$p_{x^*}$', fontsize=FONT_SIZE)
        plt.ylabel(r'$p_{y^*}$', fontsize=FONT_SIZE)
        ttl = plt.title(r'Bias: $\hat{p}_w-p_{y^*}$', fontsize=FONT_SIZE+2)
        ttl.set_position([.5, 1.05])
        ax = plt.gca()
        ax.set_xticks(np.arange(0, .1/args.delta+args.delta, .02/args.delta))
        ax.set_yticks(np.arange(0, .1/args.delta+args.delta, .02/args.delta))
        ax.set_xticklabels([r'$0.0$', r'$0.02$', r'$0.04$', r'$0.06$', r'$0.08$', r'$0.1$'], fontsize=FONT_SIZE-2)
        ax.set_yticklabels([r'$0.0$', r'$0.02$', r'$0.04$', r'$0.06$', r'$0.08$', r'$0.1$'], fontsize=FONT_SIZE-2)
        sns.despine()
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=FONT_SIZE-2)
        plt.savefig(args.plot_name.replace('.svg', '-bias.svg'))

    print "Completed! Time: %s" % str(time.time() - st_time)

if __name__ == "__main__":
    main(sys.argv[1:])
