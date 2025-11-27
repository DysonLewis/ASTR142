import numpy as np
import matplotlib.pyplot as plt

# Generate 2D Gaussian
size, sigma = 100, 10
x = np.linspace(-size//2, size//2, size)
X, Y = np.meshgrid(x, x)
gaussian = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
gaussian /= gaussian.sum()

# Save data to .dat file
np.savetxt("2d_gaussian.dat", gaussian, fmt="%.6e")

# Plot and save image
plt.figure(figsize=(6, 5))
im = plt.imshow(gaussian, cmap="viridis", 
                extent=[x[0], x[-1], x[0], x[-1]])
plt.title("2D Gaussian")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(im, label="value")
plt.tight_layout()
plt.savefig("2d_gaussian.pdf")
plt.show()