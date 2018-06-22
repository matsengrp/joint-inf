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
import collections

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

ALL_ANC_STATES = [
    '00',
    '10',
    '01',
    '11',
]

PATT2SPLIT = {
    '0000': '$\emptyset$',
    '1000': '$\{1\}$    ',
    '0100': '$\{2\}$    ',
    '0010': '$\{3\}$    ',
    '1110': '$\{1,2,3\}$',
    '1100': '$\{1,2\}$  ',
    '1010': '$\{1,3\}$  ',
    '0110': '$\{2,3\}$  ',
}

ANC2SPLIT = {
    '00': '$\emptyset$',
    '10': '$\{1\}$    ',
    '01': '$\{2\}$    ',
    '11': '$\{1,2\}$  ',
}

# ~~~~~~~~~

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

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

def P_INVFELS(theta, site_pattern):
    """
    Site pattern frequencies for inverse Felsenstein tree
    """
    # fifth branch will always be positive
    theta_sgn = [-1 if s=='1' else 1 for s in site_pattern+'0']
    prob_list = [
        [theta[0], theta[2]],
        [theta[1], theta[3]],
        [theta[0], theta[1], theta[4]],
        [theta[0], theta[3], theta[4]],
        [theta[1], theta[2], theta[4]],
        [theta[2], theta[3], theta[4]],
        [theta[0], theta[1], theta[2], theta[3]],
    ]
    sgn_list = [
        theta_sgn[0]*theta_sgn[2],
        theta_sgn[1]*theta_sgn[3],
        theta_sgn[0]*theta_sgn[1]*theta_sgn[4],
        theta_sgn[0]*theta_sgn[3]*theta_sgn[4],
        theta_sgn[1]*theta_sgn[2]*theta_sgn[4],
        theta_sgn[2]*theta_sgn[3]*theta_sgn[4],
        theta_sgn[0]*theta_sgn[1]*theta_sgn[2]*theta_sgn[3],
    ]
    new_prob_list = [resolve_exp(p, pad=False) for p in prob_list]
    out_str = ''.join(['+'+p if s > 0 else '-'+p for p,s in zip(new_prob_list, sgn_list)])

    return PATT2SPLIT[site_pattern] + '&' + '(1'+out_str+')' + '\\\\'

def L_INVFELS(theta, site_pattern):
    """
    Likelihood for inverse Felsenstein topology
    """
    all_likes = []
    all_likes += [''.join(['(1+'+t+')' if s=='0' else '(1-'+t+')' for t,s in zip(theta, site_pattern+'0')])]
    all_likes += [''.join(['(1+'+t+')' if s=='0' and i % 2 or s=='1' and not i % 2 else '(1-'+t+')' for i,(t,s) in enumerate(zip(theta, site_pattern+'0'))])]
    all_likes += [''.join(['(1+'+t+')' if s=='1' and i % 2 or s=='0' and not i % 2 else '(1-'+t+')' for i,(t,s) in enumerate(zip(theta, site_pattern+'1'))])]
    all_likes += [''.join(['(1+'+t+')' if s=='1' else '(1-'+t+')' for t,s in zip(theta, site_pattern+'1')])]

    out_str = PATT2SPLIT[site_pattern]
    for anc_state, split in zip(ALL_ANC_STATES, all_likes):
        out_str += '&' + ANC2SPLIT[anc_state] + '&$' + split + '$\\\\\n'
    return out_str

def resolve_exp(term_list, pad=True):
    """
    take list of things like ['x', 'y', 'y'] and turn it into 'xy^2'
    """
    counted = collections.Counter(term_list)
    out_str = ''
    for k in sorted(counted, key=len):
        if counted[k] > 1:
            out_str += k+'^'+str(counted[k])
            if pad:
                out_str += '   '
        else:
            out_str += k
    return out_str

def main(args=sys.argv[1:]):
    args = parse_args()

    if args.general_branch_lengths:
        pref = 'General'
        theta = ['x_1', 'y_1', 'x_2', 'y_2', 'w']
    elif args.restricted_branch_lengths:
        pref = 'Restricted'
        theta = ['x', 'y', 'x', 'y', 'w']
    else:
        pref = 'True'
        theta = ['x*', 'y*', 'x*', 'y*', 'y*']

    print pref + ' InvFels'
    print 'Probabilities:'
    for site_pattern in ALL_PATTERNS:
        print P_INVFELS(theta, site_pattern)
    print '\n'
    print 'Likelihoods:'
    for site_pattern in ALL_PATTERNS:
        print L_INVFELS(theta, site_pattern)

if __name__ == "__main__":
    main(sys.argv[1:])
