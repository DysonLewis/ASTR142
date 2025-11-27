import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Load the FITS file
filename = './jw02733-o001_t001_nircam_f444w-f470n_i2d.fits'
with fits.open(filename) as hdul:
    # Load the science data
    image_data = hdul['SCI'].data

# Convert 2D array to 1D array
flux_values = image_data.ravel()

# Remove NaN and non-positive values for logarithmic binning
flux_positive = flux_values[(~np.isnan(flux_values)) & (flux_values > 0)]

# Determine vmin and vmax using specified percentiles
vmin = np.nanpercentile(image_data[image_data > 0], 0.1)
vmax = np.nanpercentile(image_data[image_data > 0], 99.9)

# Create logarithmically spaced bins from 10^-3 to 10^4
log_min = -3
log_max = 4
n_bins = 100
log_bins = np.logspace(log_min, log_max, n_bins)

# Create figure with custom axes
fig = plt.figure(figsize=(8, 5))
ax = fig.add_axes([0.12, 0.12, 0.83, 0.83])

# Create histogram with logarithmically spaced bins
counts, bin_edges, patches = ax.hist(flux_positive, bins=log_bins, 
                                      color='steelblue', alpha=0.7, 
                                      edgecolor='black', linewidth=0.5)

# Set logarithmic scale on x-axis and limit range
ax.set_xscale('log')
ax.set_xlim(1e-3, 1e4)

# Add vertical lines for vmin and vmax
ax.axvline(vmin, color='red', linestyle='--', linewidth=2, label='vmin (0.1%)')
ax.axvline(vmax, color='green', linestyle='--', linewidth=2, label='vmax (99.9%)')

# Add text annotations for vmin and vmax
# Position text at center height of the histogram
y_max = counts.max()
y_center = y_max * 0.5
ax.text(vmin, y_center, f'vmin (0.1%)\n{vmin:.2e}', 
        color='red', fontsize=10, ha='right', va='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.text(vmax, y_center, f'vmax (99.9%)\n{vmax:.2e}', 
        color='green', fontsize=10, ha='left', va='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Labels and title
ax.set_xlabel('Flux (MJy/sr)', fontsize=12)
ax.set_ylabel('Number of Pixels', fontsize=12)
ax.set_title('Histogram of NGC 3132 Flux Values (Log-spaced bins)', fontsize=13)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3, which='both')
ax.tick_params(labelsize=10)

plt.savefig('flux_histogram.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Total pixels: {len(flux_values)}")
print(f"Positive flux pixels: {len(flux_positive)}")
print(f"vmin: {vmin:.4e}")
print(f"vmax: {vmax:.4e}")
print(f"Flux range: {flux_positive.min():.4e} to {flux_positive.max():.4e}")