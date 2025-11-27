# (c) 2023 Michael Fitzgerald (mpfitz@ucla.edu)
#
# Code for calculating the acceleration of N bodies via Newtonian gravity.
#

def get_accel_py(R, M):
    """
    Compute the gravitational accelerations of masses.

    X is an N x 3 matrix of space positions, units of [cm].

    M is a length N vector of masses, units of [g].

    Returns an N x 3 matrix of accelerations, units of [cm/s^2].
    """
    import numpy as np

    # get N x 1 matrices for position
    X = R[:, 0:1]
    Y = R[:, 1:2]
    Z = R[:, 2:3]

    # compute deltas (N x N)
    DX = X.T - X
    DY = Y.T - Y
    DZ = Z.T - Z

    # compute 1/R^3 for each pair
    IR3 = (DX**2 + DY**2 + DZ**2)**(-1.5)

    # cleanup bad values
    IR3[np.isinf(IR3)] = 0.

    # gravitational constant
    G = 6.67259e-8 # [cm^3/g/s^2]
    # accelerations
    AX = (DX*IR3)@M # note @ is a matrix multiplication operator
    AY = (DY*IR3)@M
    AZ = (DZ*IR3)@M
    A = G * np.array((AX, AY, AZ)).T

    return A

# this is a C version of the above.
from _accel import get_accel

