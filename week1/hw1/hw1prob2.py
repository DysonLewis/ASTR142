# (c) 2023 Michael Fitzgerald (mpfitz@ucla.edu)
#
# Performing some number-crunching as a componnet of a comparison between Python and C..
#
# This code shows an example of the Python 'vectorize' decorator, as well as setting
# up calculation gris and saving an array to a binary file.
#

import numpy as np

max_iter = 100 # maximum number of iterations
r2_max = 1 << 16 # 2e8
log2 = np.log(2)

@np.vectorize # this allows us to pass arrays as inputs, even though the function is written for scalars
def calc_val(x0, y0):
    ii = 0 # iteration counter
    x, y = 0., 0.

    # iteration calculator
    while (x*x + y*y <= r2_max) and (ii < max_iter):
        xt = x*x - y*y + x0
        y = 2*x*y + y0
        x = xt
        ii += 1

    # if we hit the r2_max criterion, make an adjustment
    if ii < max_iter:
        log_zn = np.log(x*x + y*y) / 2.
        nu = np.log(log_zn / log2) / log2
        ii = ii + 1 - nu

    return ii

# set up calculation grid
xmin, xmax = -2.5, 1.
ymin, ymax = -1., 1.
ny, nx = 768, 1024
xx = np.linspace(xmin, xmax, nx, endpoint=True)
yy = np.linspace(ymin, ymax, ny, endpoint=True)
xx, yy = np.meshgrid(xx, yy)

# run the calculation
zz = calc_val(xx, yy) # uses vectorized version so can take arrays as inputs

# save output data to binary file
out_fn = 'hw1prob2.npy'
np.save(out_fn, zz)
