import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd
from astropy.table import vstack

# Task 1 
# Load the NGC7469 spectrum and convert flux units
hdul = fits.open('NGC7469_SingleExt_r0.4as_cube.fits')

flux_jy = hdul['FLUX_ST'].data
wavelength_micron = hdul['WAVE'].data

flux_jy_units = flux_jy << u.Jy
flux_cgs = flux_jy_units.to(u.erg / u.cm**2 / u.s / u.Hz)

plt.figure(figsize=(12, 6))
plt.plot(wavelength_micron, flux_cgs.value, 'b-', linewidth=1)
plt.xlabel('Wavelength (μm)', fontsize=12)
plt.ylabel('Flux (erg cm⁻² s⁻¹ Hz⁻¹)', fontsize=12)
plt.title('NGC7469 Spectrum', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ngc7469_spectrum.png', dpi=300)
print("\nSpectrum plot saved as 'ngc7469_spectrum.png'")
plt.show()

hdul.close()

# Task 2
star_table = Table.read('star_list.csv', format='ascii.csv')

print("\nOriginal star table:")
print(star_table)

coords = SkyCoord(ra=star_table['RA (deg)'] << u.deg, 
                  dec=star_table['DEC (deg)'] << u.deg, 
                  frame='icrs')

ra_hms = coords.ra.to_string(unit=u.hour, sep=':', precision=2)
dec_dms = coords.dec.to_string(unit=u.degree, sep=':', precision=2)

star_table['RA (HMS)'] = ra_hms
star_table['DEC (DMS)'] = dec_dms


sigma_sb = 5.670374419e-5
L_sun = 3.828e33
R_sun = 6.957e10

luminosity_erg_s = star_table['Luminosity (Lsun)'] * L_sun
radius_cm = star_table['Radius (Rsun)'] * R_sun

temperature_K = (luminosity_erg_s / (4 * np.pi * radius_cm**2 * sigma_sb))**0.25

star_table['Temperature (K)'] = temperature_K


# Add two more stars
new_stars = Table({
    'Star': ['Proxima Centauri', 'Polaris'],
    'RA (deg)': [217.429, 37.955],
    'DEC (deg)': [-62.679, 89.264],
    'Luminosity (Lsun)': [0.0017, 2500.0],
    'Radius (Rsun)': [0.154, 46.0],
    'Mass (Msun)': [0.122, 5.4],
    'Distance (pc)': [1.301, 133.0]
})

new_coords = SkyCoord(ra=new_stars['RA (deg)'] << u.deg,
                      dec=new_stars['DEC (deg)'] << u.deg,
                      frame='icrs')

new_stars['RA (HMS)'] = new_coords.ra.to_string(unit=u.hour, sep=':', precision=2)
new_stars['DEC (DMS)'] = new_coords.dec.to_string(unit=u.degree, sep=':', precision=2)

new_luminosity = new_stars['Luminosity (Lsun)'] * L_sun
new_radius = new_stars['Radius (Rsun)'] * R_sun
new_temperature = (new_luminosity / (4 * np.pi * new_radius**2 * sigma_sb))**0.25

new_stars['Temperature (K)'] = new_temperature
complete_table = vstack([star_table, new_stars])

# Save the updated table
complete_table.write('star_list_updated.csv', format='ascii.csv', overwrite=True)


plt.figure(figsize=(10, 8))
marker_sizes = 50 + 450 * (np.log10(complete_table['Radius (Rsun)']) - np.log10(complete_table['Radius (Rsun)'].min())) / \
               (np.log10(complete_table['Radius (Rsun)'].max()) - np.log10(complete_table['Radius (Rsun)'].min()))

plt.scatter(complete_table['Temperature (K)'], complete_table['Luminosity (Lsun)'],
           s=marker_sizes, alpha=0.6, 
           c=complete_table['Temperature (K)'], cmap='hot', edgecolors='black', linewidth=0.5)
plt.xlabel('Temperature (K)', fontsize=12)
plt.ylabel('Luminosity (L☉)', fontsize=12)
plt.yscale('log')
plt.gca().invert_xaxis()
plt.title('Hertzsprung-Russell Diagram', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
for i, star in enumerate(complete_table['Star']):
    plt.annotate(star, (complete_table['Temperature (K)'][i],
                        complete_table['Luminosity (Lsun)'][i]),
                fontsize=9, alpha=0.8, xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig('hr_diagram.png', dpi=300)
plt.show()