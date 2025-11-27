# (c) 2023 Michael Fitzgerald (mpfitz@ucla.edu)
#
# Some code for computing some aspects of Keplerian orbits.  Students are expected to re-organize this code and do some example calculations
#
#


import numpy as np
import matplotlib.pyplot as plt
from kepler import kepler_position, mean_anomaly, solve_kepler


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

# dt is already defined (10 days)
# Compute velocities using finite differences
V1 = np.diff(X1, axis=0) / dt  # shape (N-1, 3)
V2 = np.diff(X2, axis=0) / dt


# Helper: add triangular direction arrows along orbit ---
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
plt.savefig('kepler_orbits.pdf')
plt.show()