import matplotlib.pyplot
from numpy import log, meshgrid, arange, maximum

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

PATTERN_TO_SPLIT_DICT = {
        '0000': '0',
        '1000': '1',
        '0100': '2',
        '0010': '3',
        '1110': '123',
        '1100': '12',
        '1010': '13',
        '0110': '23'
}

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

def split_to_gen(pattern, x, y):
    return P_FARRIS([x, y, x, y, y], pattern)

## Generating probabilities
# Left column of tab:sitepatprob
def gen_prob(split, x, y):
    return P_FARRIS([x, y, x, y, y], SPLIT_TO_PATTERN_DICT[split])

## Upper bound for Farris tree topology
# Constants from eq:a_const
def a_const_1(x, y):
    return 3. - gen_prob('1', x, y) - gen_prob('3', x, y) - gen_prob('12', x, y) - gen_prob('23', x, y)

def a_const_2(x, y):
    return 2. * (gen_prob('1', x, y) + gen_prob('3', x, y) + gen_prob('12', x, y) + gen_prob('23', x, y))

def a_const_3(x, y):
    return 3. - gen_prob('2', x, y) - gen_prob('12', x, y) - gen_prob('23', x, y) - gen_prob('123', x, y)

def a_const_4(x, y):
    return 2. * (gen_prob('2', x, y) + gen_prob('12', x, y) + gen_prob('23', x, y) + gen_prob('123', x, y))

# Constants from eq:a_const
def a_const_1_prime(x, y):
    return a_const_1(x, y) - 2.*gen_prob('13', x, y)

def a_const_2_prime(x, y):
    return a_const_2(x, y) + 2.*gen_prob('13', x, y)

def a_const_3_prime(x, y):
    return a_const_3(x, y) - 2.*gen_prob('13', x, y)

def a_const_4_prime(x, y):
    return a_const_4(x, y) + 2.*gen_prob('13', x, y)

# L^{1}_{\tau_1,\tau_1}(\mathbf{p}_{xy})
def farris_likelihood_1(x, y):
    return gen_prob('0', x, y)*log(2.) \
        + a_const_1(x, y)* log((2*a_const_1(x, y))/(a_const_1(x, y)+a_const_2(x, y))) \
        + a_const_2(x, y)* log((2*a_const_2(x, y))/(a_const_1(x, y)+a_const_2(x, y))) \
        + a_const_3(x, y)* log((2*a_const_3(x, y))/(a_const_3(x, y)+a_const_4(x, y))) \
        + a_const_4(x, y)* log((2*a_const_4(x, y))/(a_const_3(x, y)+a_const_4(x, y))) \
        + gen_prob('13', x, y)*log(2*gen_prob('13', x, y)) \
        + (1-gen_prob('13', x, y))*log(2*(1-gen_prob('13', x, y))) \
        - log(8.)
# !! either put the normalizing log(8) in joint_inf.tex or rewrite constants

# L^{2}_{\tau_1,\tau_1}(\mathbf{p}_{xy})
def farris_likelihood_2(x, y):
    return gen_prob('0', x, y)*log(2.) \
    + a_const_1_prime(x, y)* log((2*a_const_1_prime(x, y))/(a_const_1_prime(x, y)+a_const_2_prime(x, y))) \
    + a_const_2_prime(x, y)* log((2*a_const_2_prime(x, y))/(a_const_1_prime(x, y)+a_const_2_prime(x, y))) \
    + a_const_3(x, y)*log((2*a_const_3(x, y))/(a_const_3(x, y)+a_const_4(x, y))) \
    + a_const_4(x, y)*log((2*a_const_4(x, y))/(a_const_3(x, y)+a_const_4(x, y))) \
    + log(2.) \
    - log(8.)

# L^{3}_{\tau_1,\tau_1}(\mathbf{p}_{xy})
def farris_likelihood_3(x, y):
    return gen_prob('0', x, y)*log(2.) \
    + a_const_1(x, y)* log((2*a_const_1(x, y))/(a_const_1(x, y)+a_const_2(x, y))) \
    + a_const_2(x, y)* log((2*a_const_2(x, y))/(a_const_1(x, y)+a_const_2(x, y))) \
    + a_const_3_prime(x, y)*log((2*a_const_3_prime(x, y))/(a_const_3_prime(x, y)+a_const_4_prime(x, y))) \
    + a_const_4_prime(x, y)*log((2*a_const_4_prime(x, y))/(a_const_3_prime(x, y)+a_const_4_prime(x, y))) \
    + log(2.) \
    - log(8.)

## Lower bound for the Felsenstein tree topology
def b_const(x, y):
    return gen_prob('1', x, y) + gen_prob('2', x, y) + gen_prob('3', x, y) + gen_prob('123', x, y)

def felsenstein_likelihood(x, y):
    x_max = 1.
    w_max = 0.
    def y_max(u, v):
        return 1. - b_const(u, v)

    shannon_felsenstein = sum([gen_prob(PATTERN_TO_SPLIT_DICT[pattern], x, y) * log(P_FELSENSTEIN([x_max, y_max(x, y), x_max, y_max(x, y), w_max], pattern)) for pattern in ALL_PATTERNS])
    partial_felsenstein = 2*log(2.)+(2-b_const(x, y))*log(2-b_const(x, y))+b_const(x, y)*log(b_const(x, y))
    return shannon_felsenstein + partial_felsenstein


delta = 0.025
x_range = arange(0.0, 1.0, delta)
y_range = arange(0.0, 1.0, delta)
X, Y = meshgrid(x_range,y_range)

region = maximum(farris_likelihood_1(X, Y), farris_likelihood_2(X, Y), farris_likelihood_3(X, Y)) - felsenstein_likelihood(X, Y)

matplotlib.pyplot.contour(X, Y, region, [0])
matplotlib.pyplot.show()

