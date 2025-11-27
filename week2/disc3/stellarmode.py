"""This is a demo for classes and objects"""

import math

# ============================================================
# Class: Star
# ============================================================
class Star:
    """Base class for all star types."""
    G = 6.67430e-11
    SIGMA = 5.670374419e-8
    L_SUN = 3.828e26
    R_SUN = 6.957e8
    M_SUN = 1.989e30
    T_SUN = 5778

    def __init__(self, name, mass, radius, temperature, luminosity=None):
        self.name = name
        self.mass = mass          # in solar masses
        self.radius = radius      # in solar radii
        self.temperature = temperature
        self.luminosity = luminosity or self.compute_luminosity()

    def compute_luminosity(self):
        lum_watts = 4 * math.pi * (self.radius * Star.R_SUN)**2 * Star.SIGMA * self.temperature**4
        return lum_watts / Star.L_SUN

    def mean_density(self):
        return self.mass / self.radius**3

    def surface_gravity(self):
        return (self.mass * Star.M_SUN * Star.G) / ((self.radius * Star.R_SUN)**2)

    def summary(self):
        print(f"Name: {self.name}")
        print(f"  Mass        = {self.mass:.3f} M☉")
        print(f"  Radius      = {self.radius:.3f} R☉")
        print(f"  Temperature = {self.temperature:.0f} K")
        print(f"  Luminosity  = {self.luminosity:.3f} L☉")
        print(f"  Surface gravity = {self.surface_gravity():.2e} m/s²")

# ============================================================
# Subclass: Main Sequence Star
# ============================================================
class MainSequenceStar(Star):
    """Represents a hydrogen-burning main sequence star."""
    def __init__(self, name, mass):
        if mass <= 0.43:
            luminosity = 0.23 * mass**2.3
        elif mass <= 2.0:
            luminosity = mass**4.0
        else:
            luminosity = 1.5 * mass**3.5

        radius = mass**0.8
        temperature = 5778 * (luminosity**0.25) / (radius**0.5)
        super().__init__(name, mass, radius, temperature, luminosity)

    def summary(self):
        super().summary()
        print("  Type: Main Sequence Star")

# ============================================================
# Class: Black Hole (Schwarzschild)
# ============================================================
class BlackHole:
    """Simple Schwarzschild black hole with name, mass, and radius."""
    G = Star.G
    c = 2.99792458e8
    M_SUN = Star.M_SUN

    def __init__(self, name, mass):
        self.name = name
        self.mass = mass  # solar masses
        self.radius = self.schwarzschild_radius()

    def schwarzschild_radius(self):
        return (2 * BlackHole.G * self.mass * BlackHole.M_SUN) / (BlackHole.c**2)

    def summary(self):
        print(f"Name: {self.name}")
        print(f"  Type        = Black Hole")
        print(f"  Mass        = {self.mass:.3f} M☉")
        print(f"  R_s         = {self.radius:.3e} m")

# ============================================================
# Class: Binary Star
# ============================================================
class BinaryStar:
    """Represents a binary system containing two stars."""
    def __init__(self, star1, star2, separation=None, period=None):
        self.star1 = star1
        self.star2 = star2
        self.separation = separation  # AU
        self.period = period          # years
        if separation and not period:
            self.period = self.compute_period()
        elif period and not separation:
            self.separation = self.compute_separation()

    def total_mass(self):
        return self.star1.mass + self.star2.mass

    def compute_period(self):
        return math.sqrt((self.separation**3) / self.total_mass())

    def compute_separation(self):
        return (self.period ** (2/3)) * (self.total_mass() ** (1/3))

    def summary(self):
        print(f"Binary System: {self.star1.name} + {self.star2.name}")
        print(f"  Total Mass = {self.total_mass():.3f} M☉")
        if self.separation:
            print(f"  Separation = {self.separation:.3f} AU")
        if self.period:
            print(f"  Orbital Period = {self.period:.3f} years")
        print("  --- Components ---")
        self.star1.summary()
        print("  ------------------")
        self.star2.summary()
        print("  ------------------")

# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("\n=== Example: Main Sequence ===")
    sun = MainSequenceStar("Sun", 1.0)
    sun.summary()

    print("\n=== Example: New Star ===")
    newstar = MainSequenceStar("Alpha", 2.0)
    newstar.summary()

    print("\n=== Example: Black Hole ===")
    bh = BlackHole("Ton 618", 66*(10**9))
    bh.summary()

    print("\n=== Example: Binary System ===")
    binary = BinaryStar(newstar, bh, separation=5.0)
    binary.summary()
