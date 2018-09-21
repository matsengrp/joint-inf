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

from joint_inf_helpers import ALL_PATTERNS

sns.set_style('white')
sns.set_style('ticks')

# ~~~~~~~~~
# global variables

FONT_SIZE = 20

# constants to slightly speed up computation
LOG2 = np.log(2)
LOG32 = np.log(32)
ANC_STATE_DICT = {'0': (0,0), '1': (1,0), '2': (1,1), '3': (0,1)}


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
        '--cutoff',
        type=float,
        help='upper bound for p_x* and p_y* (default .5)',
        default=.5,
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        help='number of processes to use for empirically computing maxima',
        default=1,
    )
    parser.add_argument(
        '--plot-type',
        type=str,
        help="""
             choices:
             ancestral_state_conditions-- ,
             joint_empirical-- plot empirical estimate of \hat{w} for joint inference (Fig. 3),
             marginal_empirical-- plot empirical estimate of \hat{w} for marginal inference (Fig. S4),
             """,
        choices=(
             'ancestral_state_conditions',
             'joint_empirical',
             'marginal_empirical',
            ),
    )
    parser.add_argument(
        '--bias',
        action="store_true",
        help='plot and print bias instead of parameter estimate (Fig. S3)',
    )
    parser.add_argument(
        '--generating-topology',
        type=str,
        choices=('invfels', 'fels'),
        default='invfels',
        help='which topology generated the data',
    )
    parser.add_argument(
        '--in-pkl-name',
        type=str,
        help='file name for where saved output is, if computed, for time saving',
        default=None,
    )
    parser.add_argument(
        '--out-pkl-name',
        type=str,
        help='file name for where to save output for time saving',
        default='output.pkl',
    )

    args = parser.parse_args()

    return args

# ~~~~~~~~~
# Functions for exact likelihood bounds


def get_generating_probability(theta0, site_pattern, generating_topology='invfels'):
    """
    Site pattern frequencies for inverse Felsenstein tree
    """
    # fifth branch will always be positive
    theta = [-t if s=='1' else t \
            for t,s in zip(theta0, site_pattern+'0')]
    if generating_topology == 'invfels':
        prob = 0.125 * (1 + \
            theta[0]*theta[2] + \
            theta[1]*theta[3] + \
            theta[0]*theta[1]*theta[4] + \
            theta[0]*theta[3]*theta[4] + \
            theta[1]*theta[2]*theta[4] + \
            theta[2]*theta[3]*theta[4] + \
            theta[0]*theta[1]*theta[2]*theta[3])
    else:
        prob = 0.125 * (1 + \
            theta[0]*theta[1] + \
            theta[2]*theta[3] + \
            theta[0]*theta[2]*theta[4] + \
            theta[1]*theta[2]*theta[4] + \
            theta[0]*theta[3]*theta[4] + \
            theta[1]*theta[3]*theta[4] + \
            theta[0]*theta[1]*theta[2]*theta[3])
    return prob

# ~~~~~~~~~
# Functions for empirical plot

# Safe log and safe p*log(q) so we do not have as many edge cases in optimization
def safe_log(x):
    return np.log(x) if x > 0. else -np.inf

def safe_p_logq(p, logq):
    return p * logq if p > 0. else 0.

def marginal_likelihood(theta, theta_hat, generating_topology='invfels'):
    """
    Marginal likelihood given generating theta and estimated theta_hat
    """
    probs = {pattern: get_generating_probability(theta, pattern, generating_topology=generating_topology) for pattern in ALL_PATTERNS}
    all_terms = []
    likelihood = 0.
    for t in theta_hat:
        all_terms.append([1+t, 1-t])

    for pattern in ALL_PATTERNS:
        curr_sum = 0.
        for anc_state_idx, anc_state in ANC_STATE_DICT.iteritems():
            curr_prod = 1.
            anc_state = ANC_STATE_DICT[anc_state_idx]

            # w term is just whether or not ancestral states are equal
            curr_prod *= all_terms[4][int(anc_state_idx) % 2]

            # remaining terms depend on topology and tip states
            for idx, sub_pattern in enumerate(pattern):
                # the ancestral state index depends on topology
                offset_idx = int(idx % 2) if generating_topology == 'invfels' else int(idx / 2)
                curr_prod *= all_terms[idx][(anc_state[offset_idx]+int(sub_pattern)) % 2]
            curr_sum += curr_prod
        likelihood += safe_p_logq(probs[pattern], safe_log(curr_sum))

    likelihood -= LOG32
    return likelihood

def get_log_coefficients(theta, anc_states, estimating_topology='invfels', generating_topology='invfels'):
    probs = {pattern: get_generating_probability(theta, pattern, generating_topology) for pattern in ALL_PATTERNS}
    all_terms = []
    for _ in range(len(theta)):
        all_terms.append([0., 0.])

    # unambiguous ancestral states
    if estimating_topology == 'invfels' and generating_topology == 'invfels':
        unamb_patterns = ['0000', '1000', '0100', '1100']
        amb_patterns = ['0010', '1110', '0110', '1010']
    else:
        unamb_patterns = ['0000']
        amb_patterns = ['1000', '0010', '1010', '0100', '1110', '1100', '0110']

    for pattern in unamb_patterns:
        all_terms[4][0] += probs[pattern]
        for idx, sub_pattern in enumerate(pattern):
            all_terms[idx][int(sub_pattern)] += probs[pattern]

    # ambiguous ancestral states
    for anc_state_idx, pattern in zip(anc_states, amb_patterns):
        amb_anc_state = ANC_STATE_DICT[anc_state_idx]
        all_terms[4][int(anc_state_idx) % 2] += probs[pattern]
        for idx, sub_pattern in enumerate(pattern):
            offset_idx = int(idx % 2) if estimating_topology == 'invfels' else int(idx / 2)
            all_terms[idx][(amb_anc_state[offset_idx]+int(sub_pattern)) % 2] += probs[pattern]
    return all_terms

def likelihood(theta, anc_states='0000', estimating_topology='invfels', generating_topology='invfels'):
    """
    Compute upper bound of the likelihood in Lemma 2

    maximized at t=(a1-a2)/(a1+a2), but note a1+a2=1, so a1log(1+a1-a2) + a2log(1+a2-a1) is max
    """
    all_terms = get_log_coefficients(theta, anc_states, estimating_topology, generating_topology)
    likelihood = 0.
    params = []
    for term in all_terms:
        likelihood += safe_p_logq(term[0], safe_log(term[0]))
        likelihood += safe_p_logq(term[1], safe_log(term[1]))
        likelihood += LOG2
        params.append(term[0]-term[1])
    likelihood -= LOG32
    return likelihood, params

def get_ancestral_state_conds(xy, generating_topology='invfels'):
    """
    takes in xy---the generating parameter values (a 2-tuple)

    returns the difference between the lower bound of the likelihood with $\emptyset$ as the
    maximal ancestral state (i.e., Table S4) and the upper bound of the remaining likelihoods
    """
    if xy[0] == 1. or xy[0] == 0. or xy[1] == 1. or xy[1] == 0.:
        return -np.inf
    theta = [xy[0], xy[1], xy[0], xy[1], xy[1]]

    if generating_topology == 'invfels':
        anc_states = '012'
        sites = 4
    elif generating_topology == 'fels':
        anc_states = '0123'
        sites = 7

    likes = []
    for anc_state in product(anc_states, repeat=sites):
        like_val, _ = likelihood(theta, anc_state, estimating_topology=generating_topology, generating_topology=generating_topology)
        likes.append(like_val)
    max_out = max(likes)
    est = likes.index(max_out)
    return est

def get_marginal_what(xy, bias=False, generating_topology='invfels'):
    """
    Get marginal value of $\hat{w}$ empirically to compare with $\hat{w}$ estimated through joint inf
    """
    # if we're in an edge case, return nan
    if xy[0] == 1. or xy[0] == 0. or xy[1] == 1. or xy[1] == 0.:
        return np.nan
    theta = [xy[0], xy[1], xy[0], xy[1], xy[1]]
    max_obj = -np.inf
    for init_theta in [theta, [0.]*5, [1.]*5, [.5]*5]:
        output = minimize(lambda z: -marginal_likelihood(theta, z, generating_topology=generating_topology),
            x0=init_theta,
            method="L-BFGS-B",
            bounds=[(0.,1.)]*5,
        )
        if -output.fun > max_obj:
            max_obj = -output.fun
            out_obj = output
    what = out_obj.x[-1]
    if bias:
        return .5 * (xy[1] - what)
    else:
        return .5 * (1 - what)

def get_empirical_what(xy, bias=False, generating_topology='invfels'):
    """
    """
    # if we're in an edge case, return nan to gray out plot
    if xy[0] == 1. or xy[0] == 0. or xy[1] == 1. or xy[1] == 0.:
        return np.nan
    theta = [xy[0], xy[1], xy[0], xy[1], xy[1]]

    if generating_topology == 'invfels':
        anc_states = '012'
        sites = 4
        idx = -1
    elif generating_topology == 'fels':
        anc_states = '0123'
        sites = 7
        idx = -2

    likes = []
    params = []
    for anc_state in product(anc_states, repeat=sites):
        like_val, param_val = likelihood(theta, anc_state, estimating_topology=generating_topology, generating_topology=generating_topology)
        likes.append(like_val)
        params.append(param_val[idx])
    max_out = max(likes)
    what = params[likes.index(max_out)]
    if bias:
        return .5 * (xy[1] - what)
    else:
        return .5 * (1 - what)

def obj_fn(xy, plot_type='joint_empirical', bias=False, generating_topology='invfels'):
    """
    return different objective functions based on what we are interested in computing

    needed for Pool to work properly
    """
    if plot_type == 'joint_empirical':
        return get_empirical_what(xy, bias=bias, generating_topology=generating_topology)
    elif plot_type == 'marginal_empirical':
        return get_marginal_what(xy, bias=bias, generating_topology=generating_topology)
    elif plot_type == 'ancestral_state_conditions':
        return get_ancestral_state_conds(xy, generating_topology=generating_topology)
    else:
        raise ValueError('Must pass actual plot_type')

def make_plot(plottitle, plotname, legendtext=None, ct=None, scale=1, plotbar=False, move_legend=False, cutoff=.5):
    """
    tweaked plotting settings for each plot
    """
    plt.axis('scaled')
    plt.xlabel(r'$p_{x^*}$', fontsize=FONT_SIZE-4)
    plt.ylabel(r'$p_{y^*}$', fontsize=FONT_SIZE-4)
    ttl = plt.title(plottitle, fontsize=FONT_SIZE+2)
    ttl.set_position([.5, 1.05])
    if ct is not None:
        ct.ax.tick_params(labelsize=FONT_SIZE-2)
        x = np.concatenate(([0.], [cutoff], [cutoff]))
        y = np.concatenate(([cutoff], [cutoff], [0.]))
        plt.plot(x, y, '-', alpha=.15, lw=.5)
    if legendtext is not None:
        if move_legend:
            artists, labels = ct.legend_elements()
            plt.legend(artists, legendtext,
                bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=FONT_SIZE-4, mode='expand')
        else:
            plt.legend([Legend()], [legendtext], loc=2,
                handler_map={Legend: LegendHandler()}, fontsize=FONT_SIZE-4)

    ax = plt.gca()

    if scale == 1:
        ax.set_xticks(np.arange(0, cutoff+.01, cutoff/5))
        ax.set_yticks(np.arange(0, cutoff+.01, cutoff/5))
    else:
        ax.set_xticks(np.arange(0, cutoff/scale+scale, cutoff/(scale*5)))
        ax.set_yticks(np.arange(0, cutoff/scale+scale, cutoff/(scale*5)))

    if cutoff == .5:
        round_val = 1
    else:
        round_val = 2
    labels = [r'$%s$' % str(val) for val in [round(val, round_val) for val in np.arange(0, cutoff+.01, cutoff/5)]]
    ax.set_xticklabels(labels, fontsize=FONT_SIZE-2)
    ax.set_yticklabels(labels, fontsize=FONT_SIZE-2)
    sns.despine()
    if plotbar:
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=FONT_SIZE-2)
    plt.savefig(plotname)
    plt.close()

class Obj(object):
    def __init__(self, plot_type, bias=False, generating_topology='invfels'):
        self.plot_type = plot_type
        self.bias = bias
        self.generating_topology = generating_topology
    def __call__(self, xy):
        return obj_fn(xy, self.plot_type, self.bias, self.generating_topology)

def main(args=sys.argv[1:]):
    args = parse_args()

    st_time = time.time()

    # Compute output or load previously computed output from pickle
    if args.in_pkl_name is None:
        px_range = np.arange(0., args.cutoff+args.delta, args.delta)
        py_range = np.arange(0., args.cutoff+args.delta, args.delta)
        PX, PY = np.meshgrid(px_range,py_range)
        X = 1-2*PX
        Y = 1-2*PY
        if args.n_jobs > 1:
            p = Pool(processes=args.n_jobs)
            Z_pool = p.map( Obj(args.plot_type, args.bias, args.generating_topology) , [(x, y) for x, y in zip(X.ravel(), Y.ravel())])
            Z = np.reshape(Z_pool, (int(args.cutoff / args.delta)+1, int(args.cutoff / args.delta)+1))
        else:
            Z = np.vectorize(lambda x, y: obj_fn((x, y), plot_type=args.plot_type, bias=args.bias, generating_topology=args.generating_topology))(X, Y)

        with open(args.out_pkl_name, 'w') as f:
            pickle.dump((X, Y, Z, args.delta, args.cutoff, args.bias, args.generating_topology, args.plot_type), f)
    else:
        with open(args.in_pkl_name, 'r') as f:
            X, Y, Z, args.delta, args.cutoff, args.bias, args.generating_topology, args.plot_type = pickle.load(f)
        px_range = np.arange(0., args.cutoff+args.delta, args.delta)
        py_range = np.arange(0., args.cutoff+args.delta, args.delta)
        PX, PY = np.meshgrid(px_range,py_range)

    # Print error range for bias
    if args.bias:
        print "Error range (p < %.2f) [%.0E, %.0E], Mean: %.2E" % (args.cutoff, np.nanmin(Z), np.nanmax(Z), np.nanmean(Z))

    # Plot output
    if args.plot_type == 'ancestral_state_conditions':
        uniques = np.unique(Z)
        plt.contour(PX, PY, Z, uniques, colors='k', linestyles='-', linewidths=.5)
        ct = plt.contourf(PX, PY, Z, uniques, colors='none', hatches=[None, 'x', '|', '\\', '/'])
    else:
        plt.imshow(Z, cmap=plt.cm.gray_r, origin='lower')
        ct = None

    # Plotting settings
    plotbar = False
    move_legend = False
    scale = 1
    legendtext = None
    if args.plot_type == 'ancestral_state_conditions':
        plottitle = 'Maximal ancestral state splits'
        legendtext = []
        cnt = 1
        for idx, anc_state in enumerate(product('012', repeat=4)):
            if idx in uniques:
                legendtext.append(r'$\hat{\xi}_%d$' % cnt)
                cnt += 1
        move_legend = True
    elif args.plot_type == 'joint_empirical':
        if args.bias:
            plottitle = r'Bias: $\hat{p}_w-p_{y^*}$'
        else:
            plottitle = r'Estimated $\hat{p}_w$'
        scale = args.delta
        plotbar = True
    elif args.plot_type == 'marginal_empirical':
        if args.bias:
            plottitle = r'Bias: $\hat{p}_w-p_{y^*}$ (integrated likelihood)'
        else:
            plottitle = r'Estimated $\hat{p}_w$ (integrated likelihood)'
        scale = args.delta
        plotbar = True

    make_plot(plottitle, args.plot_name, legendtext=legendtext, ct=ct, scale=scale, plotbar=plotbar, cutoff=args.cutoff, move_legend=move_legend)

    print "Completed! Time: %s" % str(time.time() - st_time)

if __name__ == "__main__":
    main(sys.argv[1:])
