"""
particle_simulation.py

Simulates charged particle motion in a uniform magnetic field using RK4 integration.
Positions are saved to a CSV file. Logging tracks execution and errors.
"""

import numpy as np
import pandas as pd
import logging
import os

# =========================================
# ============== Logging ==================
# =========================================
LOG_FILE = "particle_simulation.log"
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)  # start fresh each run

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# =========================================
# ============== Constants ================
# =========================================
B_FIELD = np.array([0, 0, 1.0])  # Tesla
DT = 1e-10
STEPS = int(1e4)
TIME = np.linspace(0, STEPS * DT, STEPS)


# =========================================
# ============== Classes ==================
# =========================================
class Particle:
    """Represents a charged particle in a magnetic field."""

    def __init__(self, label: str, q: float, m: float, r0: np.ndarray, v0: np.ndarray):
        self.label = label
        self.q = q
        self.m = m
        self.r0 = r0
        self.v0 = v0
        self.r_traj = None  # Will store trajectory later
        logging.info(f"Initialized particle: {self.label}")

    def _acceleration(self, v: np.ndarray) -> np.ndarray:
        """Lorentz acceleration q/m * (v × B)."""
        return (self.q / self.m) * np.cross(v, B_FIELD)

    def integrate(self):
        """Integrate motion using RK4, store trajectory in self.r_traj."""
        try:
            r = np.zeros((STEPS, 3))
            v = np.zeros((STEPS, 3))
            r[0], v[0] = self.r0, self.v0

            for i in range(STEPS - 1):
                k1_v = self._acceleration(v[i])
                k1_r = v[i]

                k2_v = self._acceleration(v[i] + 0.5 * DT * k1_v)
                k2_r = v[i] + 0.5 * DT * k1_v

                k3_v = self._acceleration(v[i] + 0.5 * DT * k2_v)
                k3_r = v[i] + 0.5 * DT * k2_v

                k4_v = self._acceleration(v[i] + DT * k3_v)
                k4_r = v[i] + DT * k3_v

                v[i + 1] = v[i] + (DT / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
                r[i + 1] = r[i] + (DT / 6) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)

            self.r_traj = r
            logging.info(f"Integration successful for {self.label}")

        except Exception as e:
            logging.error(f"Error during integration for {self.label}: {e}", exc_info=True)


# =========================================
# ============== Simulation ===============
# =========================================
def run_simulation():
    particles = [
        Particle("Proton", q=1.6e-19, m=1.67e-27, r0=np.array([0, 0, 0]), v0=np.array([1e7, 1e7, 0.5e7])),

        Particle("Alpha Particle", q=2 * 1.6e-19, m=4 * 1.67e-27, r0=np.array([0, 0, 0]), v0=np.array([1e7, -1e7, 1e7])),

        Particle("Deuteron", q=1.6e-19, m=2 * 1.67e-27, r0=np.array([0, 0, 0]), v0=np.array([5e6, 1e7, -0.5e7])),

        #Error particles
        Particle("Zero Mass", 1.6e-19, 0.0, np.array([0,0,0]), np.array([1e7,0,0])),
        
        Particle("NaN Velocity", 1.6e-19, 1.67e-27, np.array([0,0,0]), np.array([np.nan, 1e7, 0])),
    
        Particle("Bad r0 Type", 1.6e-19, 1.67e-27, np.array(["a","b","c"]), np.array([1e7,1e7,0])),
    
        Particle("Short Velocity Vector", 1.6e-19, 1.67e-27, np.array([0,0,0]), np.array([1e7,0])),
    
        Particle("Huge Numbers", 1e40, 1e-40, np.array([0,0,0]), np.array([1e20,1e20,1e20])),
    ]

    all_data = []

    for p in particles:
        p.integrate()
        if p.r_traj is not None:
            # Convert trajectory to DataFrame and label columns
            df = pd.DataFrame(
                p.r_traj,
                columns=["x (m)", "y (m)", "z (m)"]
            )
            df.insert(0, "t (s)", TIME)
            df["particle"] = p.label
            all_data.append(df)
        else:
            logging.warning(f"No trajectory stored for {p.label}")

    # Combine all and export to CSV
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        output_file = "particle_positions.csv"
        final_df.to_csv(output_file, index=False)
        logging.info(f"Trajectory data exported to {output_file}")
    else:
        logging.error("No valid trajectories to save.")


# =========================================
# ============== Main =====================
# =========================================
if __name__ == "__main__":
    logging.info("=== Particle simulation started ===")
    run_simulation()
    logging.info("=== Particle simulation completed ===")