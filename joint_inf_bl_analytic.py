# Branch length inconsistency

import matplotlib.pyplot as plt
from numpy import sqrt, log, meshgrid, arange, maximum

def get_what(x, y):
    alpha_1 = 1+x**2+y**2+4*x*y**2+x**2*y**2
    alpha_2 = 1+x**2+y**2-4*x*y**2+x**2*y**2
    beta =    1+x**2+y**2+x**2*y**2
    gamma = 4*x*y

    a = -gamma**2*alpha_1 - gamma**2*alpha_2 - gamma**2
    b = gamma*alpha_1*beta - gamma**2*alpha_1 - gamma*alpha_2*beta - gamma**2*alpha_2
    c = gamma*alpha_1*beta - gamma*alpha_2*beta + beta**2

    what_1 = (-b + sqrt(b**2 - 4*a*c)) / (2*a)
    what_2 = (-b - sqrt(b**2 - 4*a*c)) / (2*a)

    return (what_2 >= 1)

# Want cases where what_2 is greater than one for x and y between 0 and 1

delta = 0.001
x_range = arange(0.0, 1.0, delta)
y_range = arange(0.0, 1.0, delta)
X, Y = meshgrid(x_range,y_range)

region = get_what(X, Y)

plt.contour(X, Y, region, [0])
plt.xlabel(r'x', fontsize=16)
plt.ylabel(r'y', fontsize=16)
plt.title(r'Branch length inconsistency')
plt.savefig('figures/branch-length-inconsistency.svg')

