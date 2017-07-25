import matplotlib
matplotlib.use('pdf')
from numpy import log, arange, vectorize
from scipy.optimize import basinhopping
from pylab import meshgrid,cm,imshow,title,savefig,xlabel,ylabel,colorbar

def p13(x, y, w):
    return 0.125 * (1 + x*x + y*y + x*x*y*y - 4*x*y*w)

def p0(x, y, w):
    return 0.125 * (1 + x*x + y*y + x*x*y*y + 4*x*y*w)

def likelihood(x, y, w):
    if (1 - x) * (1 + w) * (1 + y) > (1 + x) * (1 - w) * (1 - y) or \
        (1 + x) * (1 + w) * (1 - y) > (1 - x) * (1 - w) * (1 + y):
        return p0(x, y, y) * log(p0(x, y, w)) + p13(x, y, y) * log(p13(x, y, w)) + \
            log(1 + w)
    else:
        return p0(x, y, y) * log(p0(x, y, w)) + p13(x, y, y) * log(p13(x, y, w)) + \
            p13(x, y, y) * log(1 - w) + (1 - p13(x, y, y)) * log(1 + w)

def max_likelihood(x, y):
    """
    basin hopping to find maximum likelihood estimate of interior branch length
    """
    bnds = [(0,1)]
    minimizer_kwargs = dict(method="L-BFGS-B", bounds=bnds)
    what = basinhopping(lambda w: -likelihood(x, y, w),
        x0=[0.25],
        minimizer_kwargs=minimizer_kwargs)
    return(what.x)

delta = 0.01
x_range = arange(0.0, 1.0, delta)
y_range = arange(0.0, 1.0, delta)
X, Y = meshgrid(x_range,y_range)

Z = vectorize(lambda x, y: max_likelihood(x, y))(X, Y)
im = imshow(Z, cmap=cm.gray, origin='lower')
colorbar()
title('Value of $\hat{w}$')
xlabel('$x$')
ylabel('$y$')
savefig('figures/w_hat_heatmap.svg')

