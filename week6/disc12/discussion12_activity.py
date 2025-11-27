import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import astropy.units as u

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy.coordinates import get_sun



# --------------- Task 1 ----------------------


# Load the FITS file
hdul = fits.open('gc_2mass_k.fits')
data = hdul[0].data
header = hdul[0].header

# Extract WCS information
wcs = WCS(header)

# Create the figure and axis with WCS projection
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111, projection=wcs)

# Display the image
im = ax.imshow(data, cmap='gray', origin='lower', vmin=np.percentile(data, 1), 
               vmax=np.percentile(data, 99))

# Set up the coordinate display
ax.coords[0].set_axislabel('Right Ascension (J2000)')
ax.coords[1].set_axislabel('Declination (J2000)')

# Overlay galactic coordinate grid
overlay = ax.get_coords_overlay('galactic')
overlay.grid(color='cyan', ls='--', alpha=0.7, linewidth=1.5)
overlay[0].set_axislabel('Galactic Longitude')
overlay[1].set_axislabel('Galactic Latitude')
overlay[0].set_ticklabel(color='cyan')
overlay[1].set_ticklabel(color='cyan')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Intensity', rotation=270, labelpad=20)

# Set title
ax.set_title('2MASS K-band Image with Galactic Coordinate Grid', fontsize=14, pad=20)

plt.tight_layout()
plt.show()

# Close the FITS file
hdul.close()


# --------------- Task 2 ----------------------

# Target names
tgtnames = ['HD 189733', 'HD 209458', 'HD 149026', 'WASP-33', 'KELT-9', 'TOI-1518']

location = EarthLocation.of_site('Keck Observatory')

# Set up the time range for tonight
midnight = Time.now()
delta_midnight = np.linspace(-12, 12, 100) * u.hour
times = midnight + delta_midnight

# Create AltAz frame
frame = AltAz(obstime=times, location=location)

# Resolve target coordinates and calculate altitudes
fig, ax = plt.subplots(figsize=(6, 4))

colors = plt.cm.tab10(np.linspace(0, 1, len(tgtnames)))

for i, name in enumerate(tgtnames):
    try:
        # Resolve target coordinates
        coord = SkyCoord.from_name(name)
        
        # Transform to AltAz
        altaz = coord.transform_to(frame)
        
        # Plot altitude vs time
        ax.plot(delta_midnight, altaz.alt, label=name, color=colors[i], linewidth=2)
    except Exception as e:
        print(f"Could not resolve {name}: {e}")

# Plot sun position
sun = get_sun(times)
sun_altaz = sun.transform_to(frame)
ax.plot(delta_midnight, sun_altaz.alt, 'yo-', label='Sun', linewidth=2, markersize=4, alpha=0.7)

# Add twilight lines
ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Horizon')
ax.axhline(y=-6, color='b', linestyle=':', alpha=0.3, label='Civil Twilight')
ax.axhline(y=-12, color='b', linestyle=':', alpha=0.5, label='Nautical Twilight')
ax.axhline(y=-18, color='b', linestyle=':', alpha=0.7, label='Astronomical Twilight')

# Formatting
ax.fill_between(delta_midnight.value, 0, 90, sun_altaz.alt.value < -18, 
                color='0.5', zorder=0, alpha=0.3)
ax.set_xlim(-12, 12)
ax.set_ylim(0, 90)
ax.set_xlabel('Hours from Midnight')
ax.set_ylabel('Altitude (degrees)')
ax.set_title(f'Target Observability for Tonight\n{midnight.iso} UTC', fontsize=14)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('target_observability.png', dpi=150, bbox_inches='tight')
print("Observability plot saved as 'target_observability.png'")
plt.show()
plt.close()