"""
Code to print generating probabilities and likelihoods for Felsenstein and
inverse Felsenstein trees.
"""

from __future__ import unicode_literals
import argparse
import sys
import collections

from sympy import sympify, expand
from operator import mul
from numpy import isclose

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
        '--general-branch-lengths',
        action="store_true",
        help='general branch lengths probabilities',
    )
    parser.add_argument(
        '--true-branch-lengths',
        action="store_true",
        help='true branch lengths probabilities',
    )
    parser.add_argument(
        '--generating-topology',
        type=str,
        choices=('invfels', 'fels'),
        default='invfels',
        help='which topology generated the data',
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='_output/out.txt',
        help='text file to output probabilities and likelihoods',
    )

    args = parser.parse_args()

    return args

# ~~~~~~~~~
# global functions for exact likelihood computation

def get_generating_probability(theta, site_pattern, generating_topology='invfels', symbolic=False):
    """
    Site pattern frequencies
    """
    # fifth branch will always be positive
    theta_sgn = [-1 if s=='1' else 1 for s in site_pattern+'0']
    # Hadamard representation; see Semple and Steel for where these groupings come from
    if generating_topology == 'invfels':
        groupings = [(0,2), (1,3), (0,1,4), (0,3,4), (1,2,4), (2,3,4), (0,1,2,3)]
    else:
        groupings = [(0,1), (2,3), (0,2,4), (1,2,4), (0,3,4), (1,3,4), (0,1,2,3)]

    prob_list = ['1']
    for idx_tuple in groupings:
        probs = '*'.join([str(theta[idx]) for idx in idx_tuple])
        if reduce(mul, [theta_sgn[idx] for idx in idx_tuple]) < 0:
            probs += '*(-1)'
        prob_list.append(probs)

    if not symbolic:
        return 0.125 * eval('+'.join(prob_list))
    else:
        return '+'.join(prob_list)

def get_likelihood(theta, site_pattern, ancestral_pattern, generating_topology='invfels', symbolic=False):
    """
    Likelihoods
    """
    # the not (i == 4) is needed to have the correct sign for the interior branch
    if generating_topology == 'invfels':
        pat_fn = lambda i: not (i % 2) and not (i == 4)
    else:
        pat_fn = lambda i: (i < 2) and not (i == 4)

    if ancestral_pattern == '00':
        out_like = \
            '*'.join(['(1+'+str(t)+')' if s=='0' \
            else '(1-'+str(t)+')' \
            for t,s in zip(theta, site_pattern+'0')])
    if ancestral_pattern == '01':
        out_like = \
            '*'.join(['(1+'+str(t)+')' if s=='0' and pat_fn(i) or s=='1' and not pat_fn(i) \
            else '(1-'+str(t)+')' \
            for i,(t,s) in enumerate(zip(theta, site_pattern+'0'))])
    if ancestral_pattern == '10':
        out_like = \
            '*'.join(['(1+'+str(t)+')' if s=='1' and pat_fn(i) or s=='0' and not pat_fn(i) \
            else '(1-'+str(t)+')' \
            for i,(t,s) in enumerate(zip(theta, site_pattern+'1'))])
    if ancestral_pattern == '11':
        out_like = \
            '*'.join(['(1+'+str(t)+')' if s=='1' \
            else '(1-'+str(t)+')' \
            for t,s in zip(theta, site_pattern+'1')])

    if not symbolic:
        return 0.03125 * eval(out_like)
    else:
        return out_like

def main(args=sys.argv[1:]):
    args = parse_args()

    if args.general_branch_lengths:
        theta = ['x_1', 'y_1', 'x_2', 'y_2', 'w']
        symbolic = True
    elif args.true_branch_lengths:
        theta = ['x', 'y', 'x', 'y', 'y']
        symbolic = True
    else:
        # for testing evaluation
        theta = [.9, .1, .9, .1, .1]
        symbolic = False

    with open(args.output_file, 'w') as f:
        f.write('Topology = ')
        f.write(args.generating_topology)
        f.write('\n')
        f.write('Theta = ')
        f.write(','.join([str(t) for t in theta]))
        f.write('\n')
        f.write('\n')
        f.write('Generating probabilities\n')
        all_ps = []
        for site_pattern in ALL_PATTERNS:
            p = get_generating_probability(theta, site_pattern, args.generating_topology, symbolic=symbolic)
            all_ps.append(p)
            f.write('\t'.join([PATT2SPLIT[site_pattern], str(sympify(p))]))
            f.write('\n')


        f.write('\n')
        f.write('Likelihoods\n')
        all_ells = []
        for site_pattern in ALL_PATTERNS:
            f.write(PATT2SPLIT[site_pattern] + '\n')
            for ancestral_pattern in ALL_ANC_STATES:
                ell = get_likelihood(theta, site_pattern, ancestral_pattern, args.generating_topology, symbolic=symbolic)
                f.write('\t'.join(['', ANC2SPLIT[ancestral_pattern], str(ell)]))
                f.write('\n')
                all_ells.append(ell)

    # verify they are probabilities
    if symbolic:
        assert(isclose(.125 * float(sympify('+'.join(all_ps))), 1.0))
        assert(isclose(.03125 * float(sympify(expand('+'.join(all_ells)))), 1.0))
    else:
        assert(isclose(sum(all_ps), 1.0))
        assert(isclose(sum(all_ells), 1.0))

if __name__ == "__main__":
    main(sys.argv[1:])
