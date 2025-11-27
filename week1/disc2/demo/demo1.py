import numpy as np
import matplotlib.pyplot as plt

# Create a meshgrid of (x, y) coordinates
x = np.linspace(-10, 10, 800)
y = np.linspace(-10, 10, 800)
X, Y = np.meshgrid(x, y)

# Define the function: moire/interference-like pattern
Z = np.sin(X**2 + Y**2) * np.cos(X - Y)

# Plot the result
plt.figure(figsize=(8, 6))
plt.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()],
           origin='lower', cmap='viridis', interpolation='bilinear')
plt.colorbar(label='Function Value')
plt.title(r'$f(x, y) = \sin(x^2 + y^2) \cdot \cos(x - y)$')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('moire.pdf',bbox_inches='tight')
plt.show()

