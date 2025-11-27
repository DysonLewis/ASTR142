import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Load the binary data file
data = np.load('hw1prob2.npy')

# Create figure and axes
fig = Figure(figsize=(12, 9))
ax = fig.add_subplot(111)

# Display the data using imshow with an interesting colormap
im = ax.imshow(data, extent=[-2.5, 1., -1., 1.], 
               cmap='inferno', origin='lower', interpolation='bilinear')

# Add colorbar
cbar = fig.colorbar(im, ax=ax, label='Iterations')

# Set labels and title
ax.set_xlabel('Real axis', fontsize=12)
ax.set_ylabel('Imaginary axis', fontsize=12)
ax.set_title('Mandelbrot Set Visualization', fontsize=14, fontweight='bold')

# Add grid for better readability
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Save as PDF
fig.savefig('hw6prob1.pdf', bbox_inches='tight', dpi=600)