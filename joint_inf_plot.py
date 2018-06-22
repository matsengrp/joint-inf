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

from multiprocessing import Pool
from scipy.optimize import minimize
from itertools import product

sns.set_style('white')
sns.set_style('ticks')

# ~~~~~~~~~
# global variables

FONT_SIZE = 20

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
    parser.add_argument(
        '--plot-curve',
        action='store_true',
        help='plot curve showing intuition for why this happens?',
    )

    args = parser.parse_args()

    return args

# ~~~~~~~~~
# Functions for exact likelihood bounds

def GRADIENTS(x, y):
    # w gradient
    w_output = .5
    theta = [x*y*y, y*y, x*y*y, 1., 1.]
    theta0 = [x, y, x, y, y]
    for pattern in ALL_PATTERNS:
        w_output += P_INVFELS(theta0, pattern) * A_INVFELS(theta, pattern) / (1 + A_INVFELS(theta, pattern))
    y2_output = .5
    for pattern in ALL_PATTERNS:
        y2_output += P_INVFELS(theta0, pattern) * B_INVFELS(theta, pattern) / (1 + B_INVFELS(theta, pattern))
    return np.minimum(w_output, y2_output)


def A_INVFELS(theta0, site_pattern):
    theta = [-t if s=='1' else t \
            for t,s in zip(theta0, site_pattern+'0')]
    return ((theta[0]+theta[2])*(theta[1]+theta[3])) / ((1+theta[0]*theta[2])*(1+theta[1]*theta[3]))

def B_INVFELS(theta0, site_pattern):
    theta = [-t if s=='1' else t \
            for t,s in zip(theta0, site_pattern+'0')]
    return (theta[1] + theta[4]*theta[0] + theta[4]*theta[2] + theta[0]*theta[1]*theta[2]) / \
           (1 + theta[0]*theta[2] + theta[0]*theta[1]*theta[4] + theta[1]*theta[2]*theta[4])

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

def bound(theta):
    """
    Table S4 but with w=1 and y_2=1 (and then x_1, y_1, x_2 their optima)
    """
    z = [theta[0]*theta[1]*theta[1], theta[1]*theta[1], theta[0]*theta[1]*theta[1], 1., 1.]
    probs = {pattern: P_INVFELS(theta, pattern) for pattern in ALL_PATTERNS}
    fit_probs = {pattern: P_INVFELS(z, pattern) for pattern in ALL_PATTERNS}
    all_terms = []
    for zval in z:
        all_terms.append([safe_log(1+zval), safe_log(1-zval)])

    likelihood = np.sum([safe_p_logq(p_gen, safe_log(fit_probs[pattern])) for pattern, p_gen in probs.iteritems()])
    likelihood += safe_p_logq(probs['0000'], all_terms[0][0] + all_terms[1][0] + all_terms[2][0]+ all_terms[3][0]+ all_terms[4][0] - LOG32)
    likelihood += safe_p_logq(probs['1000'], all_terms[0][1] + all_terms[1][0] + all_terms[2][0]+ all_terms[3][0]+ all_terms[4][0] - LOG32)
    likelihood += safe_p_logq(probs['0100'], all_terms[0][0] + all_terms[1][1] + all_terms[2][0]+ all_terms[3][0]+ all_terms[4][0] - LOG32)
    likelihood += safe_p_logq(probs['1100'], all_terms[0][1] + all_terms[1][1] + all_terms[2][0]+ all_terms[3][0]+ all_terms[4][0] - LOG32)
    likelihood += safe_p_logq(probs['0010'], all_terms[0][0] + all_terms[1][0] + all_terms[2][1]+ all_terms[3][0]+ all_terms[4][0] - LOG32)
    likelihood += safe_p_logq(probs['1110'], all_terms[0][1] + all_terms[1][1] + all_terms[2][1]+ all_terms[3][0]+ all_terms[4][0] - LOG32)
    likelihood += safe_p_logq(probs['1010'], all_terms[0][1] + all_terms[1][0] + all_terms[2][1]+ all_terms[3][0]+ all_terms[4][0] - LOG32)
    likelihood += safe_p_logq(probs['0110'], all_terms[0][0] + all_terms[1][1] + all_terms[2][1]+ all_terms[3][0]+ all_terms[4][0] - LOG32)

    return likelihood

def likelihood(theta, z, anc_states='0000'):
    """
    Table S4 in manuscript: we have 81 possible likelihood functions we can compute
    """
    probs = {pattern: P_INVFELS(theta, pattern) for pattern in ALL_PATTERNS}
    fit_probs = {pattern: P_INVFELS(z, pattern) for pattern in ALL_PATTERNS}
    all_terms = []
    for zval in z:
        all_terms.append([safe_log(1+zval), safe_log(1-zval)])
            
    likelihood = np.sum([safe_p_logq(p_gen, safe_log(fit_probs[pattern])) for pattern, p_gen in probs.iteritems()])
    likelihood += safe_p_logq(probs['0000'], all_terms[0][0] + all_terms[1][0] + all_terms[2][0]+ all_terms[3][0]+ all_terms[4][0] - LOG32)
    likelihood += safe_p_logq(probs['1000'], all_terms[0][1] + all_terms[1][0] + all_terms[2][0]+ all_terms[3][0]+ all_terms[4][0] - LOG32)
    likelihood += safe_p_logq(probs['0100'], all_terms[0][0] + all_terms[1][1] + all_terms[2][0]+ all_terms[3][0]+ all_terms[4][0] - LOG32)
    likelihood += safe_p_logq(probs['1100'], all_terms[0][1] + all_terms[1][1] + all_terms[2][0]+ all_terms[3][0]+ all_terms[4][0] - LOG32)
    if anc_states[0] == '0':
        likelihood += safe_p_logq(probs['0010'], all_terms[0][0] + all_terms[1][0] + all_terms[2][1]+ all_terms[3][0]+ all_terms[4][0] - LOG32)
    elif anc_states[0] == '1':
        likelihood += safe_p_logq(probs['0010'], all_terms[0][1] + all_terms[1][0] + all_terms[2][0]+ all_terms[3][0]+ all_terms[4][1] - LOG32)
    elif anc_states[0] == '2':
        likelihood += safe_p_logq(probs['0010'], all_terms[0][1] + all_terms[1][1] + all_terms[2][0]+ all_terms[3][1]+ all_terms[4][0] - LOG32)

    if anc_states[1] == '0':
        likelihood += safe_p_logq(probs['1110'], all_terms[0][1] + all_terms[1][1] + all_terms[2][1]+ all_terms[3][0]+ all_terms[4][0] - LOG32)
    elif anc_states[1] == '1':
        likelihood += safe_p_logq(probs['1110'], all_terms[0][0] + all_terms[1][1] + all_terms[2][0]+ all_terms[3][0]+ all_terms[4][1] - LOG32)
    elif anc_states[1] == '2':
        likelihood += safe_p_logq(probs['1110'], all_terms[0][0] + all_terms[1][0] + all_terms[2][0]+ all_terms[3][1]+ all_terms[4][0] - LOG32)

    if anc_states[2] == '0':
        likelihood += safe_p_logq(probs['1010'], all_terms[0][1] + all_terms[1][0] + all_terms[2][1]+ all_terms[3][0]+ all_terms[4][0] - LOG32)
    elif anc_states[2] == '1':
        likelihood += safe_p_logq(probs['1010'], all_terms[0][0] + all_terms[1][0] + all_terms[2][0]+ all_terms[3][0]+ all_terms[4][1] - LOG32)
    elif anc_states[2] == '2':
        likelihood += safe_p_logq(probs['1010'], all_terms[0][0] + all_terms[1][1] + all_terms[2][0]+ all_terms[3][1]+ all_terms[4][0] - LOG32)

    if anc_states[3] == '0':
        likelihood += safe_p_logq(probs['0110'], all_terms[0][0] + all_terms[1][1] + all_terms[2][1]+ all_terms[3][0]+ all_terms[4][0] - LOG32)
    elif anc_states[3] == '1':
        likelihood += safe_p_logq(probs['0110'], all_terms[0][1] + all_terms[1][1] + all_terms[2][0]+ all_terms[3][0]+ all_terms[4][1] - LOG32)
    elif anc_states[3] == '2':
        likelihood += safe_p_logq(probs['0110'], all_terms[0][1] + all_terms[1][0] + all_terms[2][0]+ all_terms[3][1]+ all_terms[4][0] - LOG32)

    return likelihood

def get_region(xy):
    """
    Plot region where likelihood of w=1 and y_2=1 is greater than or equal to all other maximized likelihoods
    """
    # if we're in an edge case, return .5 to gray out plot
    if xy[0] == 1. or xy[0] == 0. or xy[1] == 1. or xy[0] == 0.:
        return .5
    theta = [xy[0], xy[1], xy[0], xy[1], xy[1]]
    bnd = bound(theta)
    output = []
    for anc_state in product('012', repeat=4):
        output.append(-minimize(lambda z: -likelihood(theta, z, anc_state),
            x0=theta,
            method="L-BFGS-B",
            bounds=[(0.,1.)]*5,
        ).fun)
    max_out = max(output)
    if np.isclose(bnd, max_out) or bnd >= max_out:
        return 1.
    else:
        return 0.

def main(args=sys.argv[1:]):
    args = parse_args()

    st_time = time.time()
    px_range = np.arange(0., 0.5+args.delta, args.delta)
    py_range = np.arange(0., 0.5+args.delta, args.delta)
    PX, PY = np.meshgrid(px_range,py_range)
    X = 1-2*PX
    Y = 1-2*PY

    if args.analytic:
        plottitle = r'Region where gradient w.r.t. $w$ and $y_2$ is positive on boundary'
        region = GRADIENTS(X, Y)
        legendtext = r'positive''\n'r'gradient'

        ct = plt.contour(PX, PY, region, [0], colors='k', linewidths=1)
        plt.axis('scaled')
        plt.xlabel(r'$p_{x^*}$', fontsize=FONT_SIZE-4)
        plt.ylabel(r'$p_{y^*}$', fontsize=FONT_SIZE-4)
        ttl = plt.title(plottitle, fontsize=FONT_SIZE+2)
        ttl.set_position([.5, 1.05])
        ct.ax.tick_params(labelsize=FONT_SIZE-2)
        x = np.concatenate(([0.], [.5], [.5]))
        y = np.concatenate(([.5], [.5], [0.]))
        plt.plot(x, y, '-', alpha=.15, lw=.5)
        plt.legend([Legend()], [legendtext],
            handler_map={Legend: LegendHandler()},
            bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=FONT_SIZE-4)

        ax = plt.gca()
        ax.set_xticks(np.arange(0., 0.5+.1, .1))
        ax.set_yticks(np.arange(0., 0.5+.1, .1))
        ax.set_xticklabels([r'$0.0$', r'$0.1$', r'$0.2$', r'$0.3$', r'$0.4$', r'$0.5$'], fontsize=FONT_SIZE-2)
        ax.set_yticklabels([r'$0.0$', r'$0.1$', r'$0.2$', r'$0.3$', r'$0.4$', r'$0.5$'], fontsize=FONT_SIZE-2)
        sns.despine()
        plt.savefig(args.plot_name)
        plt.close()
    elif args.empirical:
        if args.n_jobs > 1:
            p = Pool(processes=args.n_jobs)
            Z_pool = p.map( get_region , [(x, y) for x, y in zip(X.ravel(), Y.ravel())])
            Z = np.reshape(Z_pool, (int(.5 / args.delta)+1, int(.5 / args.delta)+1))
        else:
            Z = np.vectorize(lambda x, y: get_region((x, y)))(X, Y)
        
        title = 'Region of inconsistency'
        im = plt.imshow(Z, cmap=plt.cm.gray, origin='lower')
        plt.xlabel(r'$p_{x^*}$', fontsize=FONT_SIZE)
        plt.ylabel(r'$p_{y^*}$', fontsize=FONT_SIZE)
        ttl = plt.title(title, fontsize=FONT_SIZE+2)
        ttl.set_position([.5, 1.05])
        ax = plt.gca()
        ax.set_xticks(np.arange(0, .5/args.delta+args.delta, .1/args.delta))
        ax.set_yticks(np.arange(0, .5/args.delta+args.delta, .1/args.delta))
        ax.set_xticklabels([r'$0.0$', r'$0.1$', r'$0.2$', r'$0.3$', r'$0.4$', r'$0.5$'], fontsize=FONT_SIZE-2)
        ax.set_yticklabels([r'$0.0$', r'$0.1$', r'$0.2$', r'$0.3$', r'$0.4$', r'$0.5$'], fontsize=FONT_SIZE-2)
        sns.despine()
        if args.plot_curve:
            # The intuition is that if the generating parameters will most likely generate \emptyset ancestral states, then we'll
            # be in this situation. It's not exact, but it's a start.
            xcurve = np.arange(0., 1./args.delta+args.delta, .001/args.delta)
            ycurve = (args.delta * 2*xcurve / (1+args.delta*args.delta*xcurve*xcurve)) / args.delta
            plt.plot(.5/args.delta-.5*xcurve, .5/args.delta-.5*ycurve)
            # The second curve comes from the condition on the gradients. Again, looks like it might not be exact, but it's close.
            crange = np.arange(.001, 1., .001)
            XC, YC = np.meshgrid(crange, crange)
            region = GRADIENTS(XC, YC)
            ct = plt.contour((1-XC)/(2*args.delta), (1-YC)/(2*args.delta), region, [0])
        plt.savefig(args.plot_name)
        plt.close()
    else:
        raise ValueError()

    print "Completed! Time: %s" % str(time.time() - st_time)

if __name__ == "__main__":
    main(sys.argv[1:])
