import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from datetime import datetime

# Image size
width, height = 512, 512
sigma_x, sigma_y = 50, 50
x0, y0 = width // 2, height // 2

# Create sinusoidal interference pattern
x = np.linspace(0, 4 * np.pi, width)
y = np.linspace(0, 4 * np.pi, height)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) + np.sin(Y)
Z_normalized = (Z - Z.min()) / (Z.max() - Z.min())

# Display interference pattern
plt.imshow(Z_normalized, cmap='gray', origin='upper')
plt.colorbar()
plt.title('2D Interference Pattern')
plt.show()

# Create 2D Gaussian
X_grid, Y_grid = np.meshgrid(np.arange(width), np.arange(height))
gaussian = np.exp(-(((X_grid - x0) ** 2) / (2 * sigma_x ** 2) + ((Y_grid - y0) ** 2) / (2 * sigma_y ** 2)))

# Display Gaussian
plt.imshow(gaussian, cmap='hot', origin='upper')
plt.colorbar()
plt.title('2D Gaussian')
plt.show()

# Create primary FITS HDU
primary_hdu = fits.PrimaryHDU(data=Z_normalized)

# Create second extension with Gaussian and metadata
header = fits.Header()
header['AUTHOR'] = 'Your Name'
header['DATE'] = datetime.now().isoformat()
header['COMMENT'] = "2D Gaussian added as second extension"
gaussian_hdu = fits.ImageHDU(data=gaussian, header=header)

# Combine HDUs into an HDU list and save
hdul = fits.HDUList([primary_hdu, gaussian_hdu])
hdul.writeto('interference_and_gaussian.fits', overwrite=True)

# Open and display original FITS file
with fits.open('discussion08_example.fits') as hdulist:
    original_data = hdulist[0].data
    plt.imshow(original_data, cmap='gray', origin='upper')
    plt.colorbar()
    plt.title('Original discussion08_example.fits')
    plt.show()

# Scale the original data by 10 and save
with fits.open('discussion08_example.fits') as hdulist:
    hdulist[0].data *= 10
    hdulist.writeto('discussion08_example_scaled.fits', overwrite=True)

# Open and display scaled FITS file
with fits.open('discussion08_example_scaled.fits') as hdulist:
    scaled_data = hdulist[0].data
    plt.imshow(scaled_data, cmap='gray', origin='upper')
    plt.colorbar()
    plt.title('Scaled discussion08_example.fits (10x)')
    plt.show()