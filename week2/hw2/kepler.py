import numpy as np
from scipy.optimize import newton


def solve_kepler(M, e, tol=1e-10, maxiter=50):
    # Ensure M is an array even if a single value is passed
    M = np.atleast_1d(M)
    
    # Initial guess for eccentric anomaly E
    # Heuristic: offset by 0.85*e in the direction of sin(M)
    E0 = M + 0.85 * e * np.sign(np.sin(M))

    # Define Kepler's equation: E - e*sin(E) = M
    f = lambda E, M: E - e*np.sin(E) - M
    # Derivative of Kepler's equation w.r.t E
    fprime = lambda E, M: 1 - e*np.cos(E)

    # Solve for each M using Newton-Raphson
    # E0i: initial guess for current M
    # Mi: current mean anomaly
    E = np.array([
        newton(f, E0i, fprime=fprime, args=(Mi,), tol=tol, maxiter=maxiter)
        for E0i, Mi in zip(E0, M)
    ])

    # Return a scalar if input was scalar, else array
    return E if E.size > 1 else E[0]


# solves for the space position of an object in a Keplerian orbit given:
#    a      semimajor axis [AU]
#    e      eccentricity
#    inc    inclination [rad]
#    omega  argument of periastron [rad]
#    Omega  longitude of ascending node [rad]  relative to x axis, counterclockwise
#    M      mean anomaly [rad]
# Below the variables x, y, z are space position [AU]
def kepler_position(a, e, inc, omega, Omega, M):
    # Compute 3D position of object in Keplerian orbit.
    E = solve_kepler(M, e)
    nu = 2.*np.arctan2(np.sqrt(1.+e)*np.sin(E/2.), np.sqrt(1.-e)*np.cos(E/2.))  # true anomaly
    r = a * (1.-e**2) / (1.+e*np.cos(nu))  # radius

    # compute various sin/cos factors
    so, co = np.sin(omega), np.cos(omega)
    sonu, conu = np.sin(omega+nu), np.cos(omega+nu)
    sO, cO = np.sin(Omega), np.cos(Omega)
    si, ci = np.sin(inc), np.cos(inc)

    # compute space position
    x = r * (cO*conu - sO*sonu*ci)
    y = r * (sO*conu + cO*sonu*ci)
    z = r * sonu*si
    return x, y, z


def mean_anomaly(t, T, T0):
    # Compute mean anomaly from time t, period T, and periastron T0.
    return (t - T0)/T * 2.*np.pi