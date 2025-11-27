import numpy as np
import matplotlib.pyplot as plt

# ===========================================
# ============== Activity 1 =================
# ===========================================

# ==== Set up ====
B_field = np.array([0, 0, 1])  # Magnetic field along Z (Tesla)
dt = 1e-10
steps = int(1e4)
t = np.linspace(0, steps*dt, steps)

# add a deuteron that starts at the origin and give it a initial velocity you like
particles = [{
        "label": "Proton",
        "q": 1.6e-19,
        "m": 1.67e-27,
        "v0": np.array([1e7, 1e7, 0.5e7]),
        "r0": np.array([0, 0, 0])
    },
    {
        "label": "Alpha Particle",
        "q": 2*1.6e-19,
        "m": 4*1.67e-27,
        "v0": np.array([1e7, -1e7, 1e7]),
        "r0": np.array([0, 0, 0])
    }] 


# use RK4 numerical integration to calculate 
def rk4_integration(q, m, r0, v0):
    r = np.zeros((steps, 3))
    v = np.zeros((steps, 3))
    r[0], v[0] = r0, v0

    def acceleration(v):
        return (q/m) * np.cross(v, B_field)

    for i in range(steps-1):
        k1_v = acceleration(v[i])
        k1_r = v[i]

        k2_v = acceleration(v[i]+0.5*dt*k1_v)
        k2_r = v[i]+0.5*dt*k1_v

        k3_v = acceleration(v[i]+0.5*dt*k2_v)
        k3_r = v[i]+0.5*dt*k2_v

        k4_v = acceleration(v[i]+dt*k3_v)
        k4_r = v[i]+dt*k3_v

        v[i+1] = v[i]+(dt/6)*(k1_v+2*k2_v+2*k3_v+k4_v)
        r[i+1] = r[i]+(dt/6)*(k1_r+2*k2_r+2*k3_r+k4_r)

    return r # r includes position information [x,y,z] at each step


# ==== Calculate trajactory ====
for p in particles:
        r_traj = rk4_integration(p["q"], p["m"], p["r0"], p["v0"])
        p['r'] = np.array(r_traj) # store trajactory as 3D np array
    
# ==== Plot trajactory ====
# modify the following code to plot trajectory of all three particles
# create a plot of trajectory in XY
fig_xy = plt.figure(figsize=(5,5)) 
p1 = particles[0]
plt.plot(p1['r'][:,0],p1['r'][:,1],
         label=f'{p1['label']}')
plt.title(f'Trajactory of {p1['label']} - XY Plane')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# create a plot of trajectory in XZ




# ===========================================
# ============== Activity 2 =================
# ===========================================

# Create proper data structure to store the following 
# multi-wavelength flux density data of M87.
# Then, complete the plotting code to plot 
# the flux density vs. band center in log-log scale.
# Ref: The EHT MWL Science Working Group et al 2021 ApJL 911 L11


# Columns: band minimum (Hz), band center (Hz), band maximum (Hz), flux density (Jy), flux density uncertainty (Jy)

# --- ALMA -------------------------------
2.21e+11,2.21e+11,2.21e+11,1.30e+00,1.30e-01

# ---  EHT --------------------------------
2.29e+11,2.29e+11,2.29e+11,6.60e-01,1.60e-01

# --- HST --------------------------------
5.18e+14,5.19e+14,5.20e+14,4.91e-04,0.40e-04
1.09e+15,1.10e+15,1.11e+15,1.64e-04,0.24e-04

# --- Swift/XRT --------------------------
4.84e+17,1.08e+18,2.42e+18,2.08e-07,0.00e+00,13

# --- Fermi/LAT --------------------------
2.42e+22,7.65e+22,2.42e+23,4.27e-12,1.32e-12
2.42e+23,7.65e+23,2.42e+24,2.45e-13,9.83e-14
2.42e+24,7.65e+24,2.42e+25,6.49e-14,3.87e-14
2.42e+25,7.65e+25,2.42e+26,6.88e-14,0.00e+00


# Plot the spectrum
# fig = plt.figure(figsize=(10, 6))

# plt.scatter(?, ?)

# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Flux Density (Jy)')
# plt.title('M87')
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.legend()
# plt.tight_layout()
# plt.show()
