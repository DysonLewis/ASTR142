import argparse
import numpy as np
import matplotlib.pyplot as plt

### We are further expanding on the particle trajectory example from week 2.
### Modify the script below to allow command line inputs 
### to set the initial parameters of the simulation, such as particle type, charge, mass, initial velocity components, 
### time step, and number of steps.

# ---- Argument Parser Setup ----
parser = argparse.ArgumentParser(description="Calculate and plot the trajectory of charged particles in a magnetic field.")
parser.add_argument("-particle", type=str, default="proton", help="Type of particle: proton, alpha, deuteron")
parser.add_argument("-q", type=float, default=1.6e-19, help="charge of the particle (Coulombs)")
parser.add_argument("-m", type=float, default=1.67e-27, help="mass of the particle (kg)")
parser.add_argument("-vx", type=float, default=1e5, help="initial velocity in x direction (m/s)")
parser.add_argument("-vy", type=float, default=0.0, help="initial velocity in y direction (m/s)")
parser.add_argument("-vz", type=float, default=1e5, help="initial velocity in z direction (m/s)")
parser.add_argument("-x0", type=float, default=0.0, help="initial x position (m)")
parser.add_argument("-y0", type=float, default=0.0, help="initial y position (m)")
parser.add_argument("-z0", type=float, default=0.0, help="initial z position (m)")
parser.add_argument("-Bx", type=float, default=0.0, help="magnetic field in x direction (Tesla)")
parser.add_argument("-By", type=float, default=0.0, help="magnetic field in y direction (Tesla)")
parser.add_argument("-Bz", type=float, default=1.0, help="magnetic field in z direction (Tesla)")
parser.add_argument("-dt", type=float, default=1e-8, help="time step (seconds)")
parser.add_argument("-steps", type=int, default=10000, help="number of time steps")
parser.add_argument("-plot", action="store_true", help="display plot interactively")
parser.add_argument("-nosave", action="store_true", help="don't save plot to file")

# ---- Parse Arguments ----
args = parser.parse_args()

# ---- Function Definition ----

def acceleration(q, m, v, B_field):
    return (q/m) * np.cross(v, B_field)

# use RK4 numerical integration to calculate position at each time step
def rk4_integration(q, m, r0, v0, B_field, dt, steps):
    r = np.zeros((steps, 3))
    v = np.zeros((steps, 3))
    r[0], v[0] = r0, v0

    for i in range(steps-1):
        k1_v = acceleration(q, m, v[i], B_field)
        k1_r = v[i]

        k2_v = acceleration(q, m, v[i]+0.5*dt*k1_v, B_field)
        k2_r = v[i]+0.5*dt*k1_v

        k3_v = acceleration(q, m, v[i]+0.5*dt*k2_v, B_field)
        k3_r = v[i]+0.5*dt*k2_v

        k4_v = acceleration(q, m, v[i]+dt*k3_v, B_field)
        k4_r = v[i]+dt*k3_v

        v[i+1] = v[i]+(dt/6)*(k1_v+2*k2_v+2*k3_v+k4_v)
        r[i+1] = r[i]+(dt/6)*(k1_r+2*k2_r+2*k3_r+k4_r)

    return r # r includes position information [x,y,z] at each step

def plot_trajectory(r, label, save=True, plot=False):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(r[:,0], r[:,1], r[:,2], label=label)
    ax.set_title(f'Trajectory of {label} in Magnetic Field')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.legend()
    if save:
        plt.savefig(f'{label}_trajectory.png')
        print(f"Saved plot to {label}_trajectory.png")
    if plot:
        plt.show()

# ---- Main Execution ----

if __name__ == "__main__":
    # Update particle parameters based on command line arguments
    particle_types = {
        "proton": {"q": 1.6e-19, "m": 1.67e-27},
        "alpha": {"q": 2*1.6e-19, "m": 4*1.67e-27},
        "deuteron": {"q": 1.6e-19, "m": 2*1.67e-27}
    }

    if args.particle in particle_types:
        q = particle_types[args.particle]["q"]
        m = particle_types[args.particle]["m"]
        print(f"Using predefined {args.particle} parameters: q={q:.2e} C, m={m:.2e} kg")
    else:
        print(f"Using custom particle parameters: q={args.q:.2e} C, m={args.m:.2e} kg")
        q = args.q
        m = args.m

    # Set other parameters from args
    r0 = np.array([args.x0, args.y0, args.z0])
    v0 = np.array([args.vx, args.vy, args.vz])
    B_field = np.array([args.Bx, args.By, args.Bz])
    dt = args.dt
    steps = args.steps
    
    # Print simulation parameters
    print(f"Initial position: {r0}")
    print(f"Initial velocity: {v0} m/s")
    print(f"Magnetic field: {B_field} T")
    print(f"Time step: {dt} s")
    print(f"Number of steps: {steps}")
    print(f"Total simulation time: {dt*steps} s")
    
    # Run simulation
    r_traj = rk4_integration(q, m, r0, v0, B_field, dt, steps)
    
    # Create filename reflecting parameters
    label = f"{args.particle}_q{q:.2e}_m{m:.2e}_B{np.linalg.norm(B_field):.2f}T"
    
    # Save trajectory data
    np.save(f'{label}_trajectory.npy', r_traj)
    print(f"Saved trajectory data to {label}_trajectory.npy")
    
    # Plot trajectory
    plot_trajectory(r_traj, label, save=not args.nosave, plot=args.plot)