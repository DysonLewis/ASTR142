import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from reproject import reproject_interp
import warnings
warnings.filterwarnings('ignore')

primary_hdul = fits.open('input.fits')
primary_data = primary_hdul[0].data

mask_hdul = fits.open('input_mask.fits')
uncertainty_data = mask_hdul[0].data

miri_hdul = fits.open('F2550W_finalsc_DC.fits')
miri_data = miri_hdul[0].data
miri_header = miri_hdul[0].header
miri_wcs = WCS(miri_header)

# Find brightest source in primary image
primary_max = np.nanmax(primary_data)
primary_max_loc = np.where(primary_data == primary_max)
primary_bright_y, primary_bright_x = primary_max_loc[0][0], primary_max_loc[1][0]
print(f"\nBrightest source in primary image:")
print(f"  Pixel: ({primary_bright_x}, {primary_bright_y})")
print(f"  Value: {primary_max:.2f}")

# Find brightest sources in MIRI image
miri_bright_locs = []
miri_flat = miri_data.flatten()
sorted_indices = np.argsort(miri_flat)[::-1]
for idx in sorted_indices[:100]:
    y, x = np.unravel_index(idx, miri_data.shape)
    if np.isfinite(miri_data[y, x]) and miri_data[y, x] > 0:
        too_close = False
        for my, mx, _ in miri_bright_locs:
            if np.sqrt((x-mx)**2 + (y-my)**2) < 20:
                too_close = True
                break
        if not too_close:
            miri_bright_locs.append((y, x, miri_data[y, x]))
        if len(miri_bright_locs) >= 5:
            break

print(f"\nTop 5 brightest sources in MIRI:")
for i, (y, x, val) in enumerate(miri_bright_locs, 1):
    coord = miri_wcs.pixel_to_world(x, y)
    print(f"  {i}. Pixel ({x:6.1f}, {y:6.1f}): {val:.2e} -> "
          f"RA={coord.ra.deg:.6f}°, Dec={coord.dec.deg:.6f}°")
    
miri_bright_y, miri_bright_x, miri_bright_val = miri_bright_locs[0]
ref_coord = miri_wcs.pixel_to_world(miri_bright_x, miri_bright_y)

print(f"Primary pixel ({primary_bright_x}, {primary_bright_y}) = "
      f"MIRI pixel ({miri_bright_x:.1f}, {miri_bright_y:.1f})")
print(f"This corresponds to sky position:")
print(f"  RA  = {ref_coord.ra.to_string(unit=u.hourangle, sep=':', precision=4)}")

pixel_scale = 0.039686  # arcsec/pixel
w = WCS(naxis=2)
w.wcs.crpix = [primary_bright_x, primary_bright_y]  # Brightest source as reference
w.wcs.cdelt = [-pixel_scale / 3600., pixel_scale / 3600.]  # degrees/pixel
w.wcs.crval = [ref_coord.ra.deg, ref_coord.dec.deg]
w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

primary_hdu = fits.PrimaryHDU(data=primary_data)
primary_hdu.header.update(w.to_header())

uncertainty_hdu = fits.ImageHDU(data=uncertainty_data, name='UNCERTAINTY')
uncertainty_hdu.header['EXTVER'] = 1

hdul_out = fits.HDUList([primary_hdu, uncertainty_hdu])
hdul_out.writeto('fomalhaut_multiext.fits', overwrite=True)

miri_reprojected, footprint = reproject_interp((miri_data, miri_wcs), w, 
                                                 shape_out=primary_data.shape)

print(f"Reprojection complete:")
print(f"  Output shape: {miri_reprojected.shape}")
print(f"  Finite pixels: {np.sum(np.isfinite(miri_reprojected))}")
print(f"  Footprint coverage: {np.sum(footprint > 0.5)/footprint.size*100:.1f}%")

miri_finite = miri_reprojected[np.isfinite(miri_reprojected)]
if len(miri_finite) > 0:
    print(f"  Value range: {np.nanmin(miri_finite):.2e} to {np.nanmax(miri_finite):.2e}")

# Find top 10 brightest regions
sorted_indices = np.argsort(primary_data.flatten())[::-1]
top_coords = []
for idx in sorted_indices[:100]:  # Check top 100 pixels
    y, x = np.unravel_index(idx, primary_data.shape)
    # Skip if too close to already found sources
    too_close = False
    for ty, tx in top_coords:
        if np.sqrt((x-tx)**2 + (y-ty)**2) < 50:  # 50 pixel separation
            too_close = True
            break
    if not too_close:
        top_coords.append((y, x))
    if len(top_coords) >= 10:
        break

fig = plt.figure(figsize=(18, 8))

# Calculate display ranges
vmin = np.nanpercentile(primary_data, 1)
vmax = np.nanpercentile(primary_data, 99.5)
miri_vmin = np.nanpercentile(miri_data, 1)
miri_vmax = np.nanpercentile(miri_data, 99.5)

# Left: Primary image with marked sources
ax1 = plt.subplot(1, 3, 1)
im1 = ax1.imshow(primary_data, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
plt.colorbar(im1, ax=ax1, label='Flux (ADU)')

# Mark the top sources
for i, (y, x) in enumerate(top_coords[:5], 1):
    circle = Circle((x, y), 30, color='red', fill=False, linewidth=2)
    ax1.add_patch(circle)
    ax1.text(x, y+40, f'{i}', color='red', fontsize=12, ha='center', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax1.set_title('Primary Image (2010)\nTop 5 brightest sources marked')
ax1.set_xlabel('X pixel')
ax1.set_ylabel('Y pixel')
ax1.set_xlim(0, primary_data.shape[1])
ax1.set_ylim(0, primary_data.shape[0])

# Middle: MIRI image
ax2 = plt.subplot(1, 3, 2)
im2 = ax2.imshow(miri_data, cmap='hot', vmin=miri_vmin, vmax=miri_vmax, origin='lower')
plt.colorbar(im2, ax=ax2, label='MIRI Flux')

# Mark where reference star should be
miri_x, miri_y = miri_bright_x, miri_bright_y
if 0 <= miri_x < miri_data.shape[1] and 0 <= miri_y < miri_data.shape[0]:
    circle = Circle((miri_x, miri_y), 10, color='cyan', fill=False, linewidth=2)
    ax2.add_patch(circle)
    ax2.text(miri_x, miri_y+15, 'Ref star', color='cyan', fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

ax2.set_title('JWST/MIRI F2550W\nReference star marked in cyan')
ax2.set_xlabel('X pixel')
ax2.set_ylabel('Y pixel')

# Right: Zoomed view of primary image around likely star location
ax3 = plt.subplot(1, 3, 3)
# Zoom around the brightest source
zoom_size = 200
y_bright, x_bright = top_coords[0]
y_min = max(0, y_bright - zoom_size)
y_max = min(primary_data.shape[0], y_bright + zoom_size)
x_min = max(0, x_bright - zoom_size)
x_max = min(primary_data.shape[1], x_bright + zoom_size)

zoom_data = primary_data[y_min:y_max, x_min:x_max]
im3 = ax3.imshow(zoom_data, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
plt.colorbar(im3, ax=ax3, label='Flux (ADU)')

# Mark sources in zoomed view
for i, (y, x) in enumerate(top_coords[:5], 1):
    if x_min <= x < x_max and y_min <= y < y_max:
        circle = Circle((x-x_min, y-y_min), 20, color='red', fill=False, linewidth=2)
        ax3.add_patch(circle)
        ax3.text(x-x_min, y-y_min+25, f'{i}', color='red', fontsize=12, ha='center',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax3.set_title(f'Zoomed: Brightest Source\nCentered on pixel ({x_bright}, {y_bright})')
ax3.set_xlabel(f'X pixel (offset from {x_min})')
ax3.set_ylabel(f'Y pixel (offset from {y_min})')

plt.tight_layout()
plt.savefig('hw6prob3_extra.pdf', dpi=300, bbox_inches='tight')
print("Saved: hw6prob3_extra.pdf")

fig, ax = plt.subplots(figsize=(14, 12), subplot_kw=dict(projection=w))

# Plot primary image
norm = plt.matplotlib.colors.AsinhNorm(vmin=vmin, vmax=vmax, linear_width=vmax/10)
im = ax.imshow(primary_data, cmap='viridis', norm=norm, origin='lower')
cbar = plt.colorbar(im, ax=ax, label='Flux (ADU)', pad=0.02)

# Add MIRI contours if data exists
miri_positive = miri_finite[miri_finite > 0] if len(miri_finite) > 0 else []

if len(miri_positive) > 10:
    
    # Calculate contour levels
    percentiles = [50, 70, 85, 92, 97, 99]
    miri_levels = np.percentile(miri_positive, percentiles)
    miri_levels = np.unique(miri_levels)
    
    print(f"  Contour levels: {miri_levels}")
    
    # Plot contours
    contours = ax.contour(miri_reprojected, levels=miri_levels, 
                          colors='cyan', linewidths=2.5, alpha=0.95, linestyles='solid')
    ax.clabel(contours, inline=True, fontsize=9, fmt='%.1e', colors='cyan')
    
    # Add filled contours for visibility
    ax.contourf(miri_reprojected, levels=miri_levels, 
                colors='red', alpha=0.15)

# Formatting
ax.set_xlabel('RA (J2000)', fontsize=13)
ax.set_ylabel('Dec (J2000)', fontsize=13)
ax.set_title('Fomalhaut Deep Image (2010) with JWST/MIRI F2550W Overlay', 
             fontsize=15, weight='bold', pad=20)

# Set view limits
ax.set_xlim(300, 1200)
ax.set_ylim(300, 1200)

# Grid and legend
ax.grid(alpha=0.3, color='white', linestyle='--', linewidth=0.5)

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elements = [
    Line2D([0], [0], color='cyan', lw=2.5, label='MIRI F2550W contours'),
    Patch(facecolor='red', alpha=0.15, label='MIRI filled regions')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)

plt.tight_layout()
plt.savefig('hw6prob3.pdf', dpi=300, bbox_inches='tight')
print("Saved: hw6prob3.pdf")

plt.show()

miri_scale = np.sqrt(miri_wcs.pixel_scale_matrix[0,0]**2 + 
                     miri_wcs.pixel_scale_matrix[0,1]**2) * 3600
print(f"Primary pixel scale: {pixel_scale:.6f} arcsec/pixel")
print(f"MIRI pixel scale:    {miri_scale:.6f} arcsec/pixel")
print(f"Scale ratio:         {miri_scale/pixel_scale:.2f}x")

# Check if calibration source aligns
primary_cal_sky = w.pixel_to_world(primary_bright_x, primary_bright_y)
miri_cal_sky = miri_wcs.pixel_to_world(miri_bright_x, miri_bright_y)
separation = primary_cal_sky.separation(miri_cal_sky).arcsec

primary_hdul.close()
mask_hdul.close()
miri_hdul.close()