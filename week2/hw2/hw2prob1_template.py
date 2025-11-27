# (c) 2023 Michael Fitzgerald (mpfitz@ucla.edu)
#
# Some code for computing some aspects of Keplerian orbits.  Students are expected to re-organize this code and do some example calculations
#
#


import numpy as np
from scipy.optimize import newton
from scipy.optimize import fmin
import matplotlib.pyplot as plt

# -----------------------------
# The following blocks of code are useful for calculating some aspects of Keplerian orbits.  However, they are not currently usable.  You should move this code to a module, and wrap them in function definitions.
#



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
    """Compute 3D position of object in Keplerian orbit."""
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
    """Compute mean anomaly from time t, period T, and periastron T0."""
    return (t - T0)/T * 2.*np.pi

# end of code snippets for Keplerian calculations
# -----------------------------



# parameters in order are:
#    a      semimajor axis [AU]
#    e      eccentricity
#    inc    inclination [rad]
#    omega  argument of periastron [rad]
#    Omega  longitude of ascending node [rad]  relative to x axis, counterclockwise
#    T      period [days]
#    T0     time of periastron [days]
# Keplerian parameters for object 1
obj1parms = (10., 0.02, 70. * np.pi/180., 20. * np.pi/180., -15. * np.pi/180., 30. * 365.25, 1.88)
# Keplerian parameters for object 2
obj2parms = (15., 0.3, 85. * np.pi/180., 15. * np.pi/180., 5. * np.pi/180., 55. * 365.25, 8.66)
# Note, normally the period of the objects would be directly related to the semimajor axis, given the mass of the central star.  Here I am just making up parameters.


# compute the positions of objects 1 and 2 every 10 days until object 2 begins to reverse direction along the line of sight (z axis).
dt = 10. # days
t = 0.
X1, X2 = [], [] # empty arrays to hold positions at each timestep

z_prev = None
while True: # what kind of loop?  what is the stopping condition?

    # mean anomaly for this timestep
    M1 = mean_anomaly(t, obj1parms[5], obj1parms[6])
    M2 = mean_anomaly(t, obj2parms[5], obj2parms[6])

    x1, y1, z1 = kepler_position(*obj1parms[:5], M1) # compute position using object 1 parameters and current mean anomaly
    x2, y2, z2 = kepler_position(*obj2parms[:5], M2) # compute position using object 2 parameters and current mean anomaly

    # add 3-tuple of position to end of list
    X1.append((x1, y1, z1))
    X2.append((x2, y2, z2))

    if z_prev is not None and z2 < z_prev:
        break
    z_prev = z2
    t += dt

X1 = np.array(X1) # converts list of 3-tuples to 2d array
X2 = np.array(X2) # converts list of 3-tuples to 2d array


# --- Helper: add triangular direction arrows along orbit ---
def add_direction_arrows(X, step=10, color='k', size=10):
    x, y = X[:,0], X[:,1]
    for i in range(0, len(x)-1, step):
        dx = x[i+1] - x[i]
        dy = y[i+1] - y[i]
        # normalize to get consistent arrow size
        L = np.hypot(dx, dy)
        if L == 0:
            continue
        dx /= L
        dy /= L
        # small arrow in direction of motion
        plt.arrow(x[i], y[i], dx*0.5, dy*0.5, 
                  shape='full', lw=0, length_includes_head=True,
                  head_width=0.2, head_length=0.2, color=color)


# dt is already defined (10 days)
# Compute velocities using finite differences
V1 = np.diff(X1, axis=0) / dt  # shape (N-1, 3)
V2 = np.diff(X2, axis=0) / dt

# --- Plot the orbits ---
plt.figure(figsize=(8,6))
plt.plot(X1[:,0], X1[:,1], label='Object 1')
plt.plot(X2[:,0], X2[:,1], label='Object 2')

# Add direction arrows along orbit
add_direction_arrows(X1, step=25, color='C0')
add_direction_arrows(X2, step=25, color='C1')

# Conversion factor AU/day → km/s
AU_per_day_to_kms = 1.496e8 / 86400  # ~1731.456

# Last velocity in km/s
vx1_kms, vy1_kms, vz1_kms = V1[-1] * AU_per_day_to_kms
vx2_kms, vy2_kms, vz2_kms = V2[-1] * AU_per_day_to_kms

# Format text
text1 = f'v = ({vx1_kms:.2f}, {vy1_kms:.2f}, {vz1_kms:.2f}) km/s'
text2 = f'v = ({vx2_kms:.2f}, {vy2_kms:.2f}, {vz2_kms:.2f}) km/s'

# Place text above last point
plt.text(X1[-1,0], X1[-1,1]+0.5, text1, color='C0', fontsize=10)
plt.text(X2[-1,0], X2[-1,1]+0.5, text2, color='C1', fontsize=10)

plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.savefig('kepler_orbits_with_velocity_text.pdf')
plt.show()

