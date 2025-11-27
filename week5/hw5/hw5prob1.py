import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits

# Load the FITS file
filename = './jw02733-o001_t001_nircam_f444w-f470n_i2d.fits'

with fits.open(filename) as hdul:
    # Display FITS structure to understand the file
    # hdul.info()
    # Load the science data (typically in the first or 'SCI' extension)
    image_data = hdul['SCI'].data

# Get center region (central 30% of the image)
ny, nx = image_data.shape
y_center, x_center = ny // 2, nx // 2
zoom_size = int(min(ny, nx) * 0.15)  # 15% of image size for zoom
y_slice = slice(y_center - zoom_size, y_center + zoom_size)
x_slice = slice(x_center - zoom_size, x_center + zoom_size)
zoom_data = image_data[y_slice, x_slice]

# Create figure with specified dimensions
fig = plt.figure(figsize=(6.5, 4))

# Determine appropriate vmin and vmax for logarithmic scaling
# Use percentiles to avoid extreme outliers
vmin = np.nanpercentile(image_data[image_data > 0], 0.1)
vmax = np.nanpercentile(image_data[image_data > 0], 99.9)

# Create main axes for the full image (left panel)
ax = fig.add_axes([0.08, 0.15, 0.35, 0.8])
cax = fig.add_axes([0.88, 0.15, 0.025, 0.8])
ax_zoom = fig.add_axes([0.5, 0.15, 0.35, 0.8])

# Display the full image with logarithmic normalization
im = ax.imshow(image_data, cmap='inferno', norm=LogNorm(vmin=vmin, vmax=vmax), 
               origin='lower', interpolation='nearest')

# Add rectangle to show zoom region on full image
from matplotlib.patches import Rectangle
rect = Rectangle((x_center - zoom_size, y_center - zoom_size), 
                 2 * zoom_size, 2 * zoom_size,
                 linewidth=1, edgecolor='cyan', facecolor='none')
ax.add_patch(rect)

# Display zoomed image
im_zoom = ax_zoom.imshow(zoom_data, cmap='inferno', norm=LogNorm(vmin=vmin, vmax=vmax),
                         origin='lower', interpolation='nearest',
                         extent=[x_center - zoom_size, x_center + zoom_size,
                                y_center - zoom_size, y_center + zoom_size])

# Add labels
ax.set_xlabel('X Pixel', fontsize=10)
ax.set_ylabel('Y Pixel', fontsize=10)
ax.set_title('NGC 3132 - Full Field', fontsize=11, pad=10)
ax.tick_params(labelsize=9)
ax_zoom.set_xlabel('X Pixel', fontsize=10)
ax_zoom.set_ylabel('Y Pixel', fontsize=10)
ax_zoom.set_title('Central Region (Zoomed)', fontsize=11, pad=10)
ax_zoom.tick_params(labelsize=9)


# Add colorbar
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Flux (MJy/sr)', fontsize=10)
cax.tick_params(labelsize=9)

plt.savefig('ngc3132_image.png', dpi=500, bbox_inches='tight')
plt.show()

print(f"Image shape: {image_data.shape}")
print(f"Zoom region: [{y_center - zoom_size}:{y_center + zoom_size}, {x_center - zoom_size}:{x_center + zoom_size}]")
print(f"vmin: {vmin:.4e}, vmax: {vmax:.4e}")