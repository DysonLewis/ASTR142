import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import get_body_barycentric_posvel
from astropy import units as u
from astropy.time import Time
from accel import get_accel  # returns N x 3 accelerations in cm/s²

# Constants
AU = 1.496e13      # cm
Msol = 1.989e33
Mjup = 1.899e30
Mearth = 5.9742e27
yr = 3.15576e7     # seconds
G = 6.6743e-8      # cm^3 g^-1 s^-2

# Bodies
bodies = ["sun","mercury","venus","earth","mars","jupiter","saturn","uranus","neptune"]
N = len(bodies)

# Sublists
rocky = bodies[1:5]
gas_giants = bodies[5:9]

# Initial positions/velocities
X0 = np.zeros((N,3))
V0 = np.zeros((N,3))
M = np.zeros(N)

apt = Time('2025-10-15 00:00')
for i, body in enumerate(bodies):
    pos, vel = get_body_barycentric_posvel(body, apt)
    X0[i,:] = pos.xyz.to(u.cm).value
    V0[i,:] = vel.xyz.to(u.cm/u.s).value

# Masses
M[0] = Msol
M[1] = 0.0553*Mearth
M[2] = 0.815*Mearth
M[3] = Mearth
M[4] = 0.107*Mearth
M[5] = Mjup
M[6] = 0.299*Mjup
M[7] = 0.046*Mjup
M[8] = 0.054*Mjup

# Perturbation preset
perturb_type = input(
    "Choose perturbation type ('small (10-1e3cm', 'large (1e3-1e7)', 'custom'): "
    # Small/large presets perturbe all planets, custom allows the user to specify which planets to perturbe
).lower().strip()

if perturb_type == "small":
    perturb_planets = bodies[1:]
    perturb_pos_scale = rng.uniform(10,1e3)   # 1 km in cm
    perturb_vel_scale = rng.uniform(10,1e3)
    print(f"Small perturbation applied to all planets: pos={perturb_pos_scale} cm, vel={perturb_vel_scale} cm/s")
elif perturb_type == "large":
    perturb_planets = bodies[1:]
    perturb_pos_scale = rng.uniform(1e3, 1e7)   # 1e7 = 100 km in cm, this also flings planets out of the solar system
    perturb_vel_scale = rng.uniform(1e3, 1e7)
    print(f"Large perturbation applied to all planets: pos={perturb_pos_scale} cm, vel={perturb_vel_scale} cm/s")
elif perturb_type == "custom":
    perturb_input = input(
        "Enter planets to perturb (comma-separated, e.g., earth,jupiter) or 'all': "
    ).lower().strip()
    if perturb_input == "all":
        perturb_planets = bodies[1:]
    else:
        perturb_planets = [p.strip() for p in perturb_input.split(",")]

    perturb_pos_scale = float(input("Enter position perturbation scale [cm]: "))
    perturb_vel_scale = float(input("Enter velocity perturbation scale [cm/s]: "))
else:
    raise ValueError("Invalid perturbation type. Please enter 'small', 'large', or 'custom'.")


# --- Determine timestep dt ---
if set(perturb_planets).issubset(rocky) and "mercury" not in perturb_planets:
    dt = 0.01 * yr
elif perturb_planets == ["mercury"]:
    ''' 0.01yr is still not super accurate for mercury. if we're 
    choosing mercury as one of the focused planets, we may as well 
    simulate it better (takes more processing time though) 
    '''
    dt = 0.001 * yr
elif set(perturb_planets).issubset(gas_giants):
    dt = 0.1 * yr
else:
    dt = 0.01 * yr

print(f"Using dt = {dt/yr} yr")

# Random generator
seed = 259469
rng = np.random.default_rng(seed=seed)

''' the following gets number of orbits to simulate, this is based off
    the planet with the longest period ex. if input is earth,jupiter and 
    orbit is 10, it will simulate 10 orbits of jupiter, which is a lot 
    more orbits for earth. if only rocky planets are selected and orbit number 
    is low 1-50, some of the outer gas giants might not make a complete orbit 
    (neptune orbt period is ~170 yr)
    '''
# Determine outermost perturbed planet
if perturb_planets:
    outermost_planet = max(perturb_planets, key=lambda p: bodies.index(p))
    n_orbits = int(input(f"Enter number of orbits for {outermost_planet.capitalize()}: "))
else:
    outermost_planet = None
    n_orbits = 1

# Index map
planet_idx = {body: i for i, body in enumerate(bodies)}

# Determine simulation years 
outer_idx = planet_idx[outermost_planet]
r0 = np.linalg.norm(X0[outer_idx] - X0[0])
T_sec = 2 * np.pi * np.sqrt(r0**3 / (G * M[0]))
T_yr = T_sec / yr
n_years = n_orbits * T_yr
print(f"Simulating {n_years:.2f} years (~{n_orbits} orbits of {outermost_planet.capitalize()})")

n_step = int((n_years * yr) / dt)

# Number of simulations
n_simulations = int(input("Enter number of simulations to run: "))

# Run simulations
all_dfs = []

for sim in range(n_simulations):
    print(f"\n=== Simulation {sim+1}/{n_simulations} ===")
    
    X = X0.copy()
    V = V0.copy()
    
    # Apply perturbations
    for planet in perturb_planets:
        if planet in planet_idx:
            idx = planet_idx[planet]
            X[idx] += rng.normal(scale=perturb_pos_scale, size=3)
            V[idx] += rng.normal(scale=perturb_vel_scale, size=3)
        else:
            print(f"Warning: '{planet}' not recognized. Skipping.")
    
    # Integration
    acc = get_accel(X, M)
    records = []

    for step in range(n_step):
        t = step * dt
        V += acc * dt / 2
        X += V * dt
        acc = get_accel(X, M)
        V += acc * dt / 2
        
        KE = 0.5 * M * np.sum(V**2, axis=1)
        PE = np.zeros(N)
        for i in range(N):
            for j in range(i+1, N):
                r = np.linalg.norm(X[i] - X[j])
                PE[i] += -G * M[i] * M[j] / r
                PE[j] += -G * M[i] * M[j] / r
        
        for i, body in enumerate(bodies):
            records.append({
                "simulation": sim+1,
                "time_yr": t/yr,
                "body": body,
                "x_cm": X[i,0],
                "y_cm": X[i,1],
                "z_cm": X[i,2],
                "vx_cm_s": V[i,0],
                "vy_cm_s": V[i,1],
                "vz_cm_s": V[i,2],
                "KE": KE[i],
                "PE": PE[i],
                "E_tot": KE[i]+PE[i]
            })
    
    df_sim = pd.DataFrame(records)
    all_dfs.append(df_sim)

# Combine all simulations
df = pd.concat(all_dfs, ignore_index=True)
df.to_csv("solar_system_simulations.csv", index=False)
print("All simulations complete. Data saved to solar_system_simulations.csv")

# --- Plot gas giants ---
plt.figure(figsize=(10,10))
for body in gas_giants:
    for sim in range(n_simulations):
        planet_df = df[(df["body"]==body) & (df["simulation"]==sim+1)]
        plt.plot(planet_df["x_cm"]/AU, planet_df["y_cm"]/AU, label=f"{body.capitalize()} (Sim {sim+1})")
plt.xlabel("X [AU]")
plt.ylabel("Y [AU]")
plt.title("Gas Giant Orbits with Random Perturbations")
plt.legend()
plt.axis("equal")
plt.show()

# --- Plot perturbed planets ---
plt.figure(figsize=(10,10))
for planet in perturb_planets:
    if planet in planet_idx:
        for sim in range(n_simulations):
            planet_df = df[(df["body"]==planet) & (df["simulation"]==sim+1)]
            plt.plot(planet_df["x_cm"]/AU, planet_df["y_cm"]/AU, label=f"{planet.capitalize()} (Sim {sim+1})")
plt.xlabel("X [AU]")
plt.ylabel("Y [AU]")
plt.title("Orbits of Perturbed Planets")
plt.legend()
plt.axis("equal")
plt.show()
