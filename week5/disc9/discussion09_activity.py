import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm

def example_2D(x, y):
	return (np.sin(x)+np.cos(5*y)) * np.exp(-np.sqrt(x**2+y**2))

x = np.linspace(-5, 5, 300)
y = np.linspace(-5, 5, 300)
X, Y = np.meshgrid(x, y)
Z = example_2D(X, Y)

plt.figure(figsize=(7, 5))
plt.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()],
           origin='lower', cmap='plasma', aspect='auto')
plt.colorbar(label='f(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Function Example')
plt.show()


### Now lets do some histograms. First let's make some fake data
x1 = np.random.normal(loc=2,scale=3,size=20)
x2 = np.random.normal(loc=4, scale=0.5,size=20)

mu, sigma = 2, 3
x_vals = np.linspace(-10, 10, 300)
true_pdf = norm.pdf(x_vals, mu, sigma)

plt.figure(figsize=(7, 5))
plt.hist(x1, bins=15, density=True, alpha=0.6, label='Samples')
plt.plot(x_vals, true_pdf, 'r-', label='True distribution')
plt.xlabel('x1 values')
plt.ylabel('Probability density')
plt.legend()
plt.title('Sample vs True Distribution')
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                      wspace=0.05, hspace=0.05)

ax_main = fig.add_subplot(gs[1, 0])
ax_xhist = fig.add_subplot(gs[0, 0], sharex=ax_main)
ax_yhist = fig.add_subplot(gs[1, 1], sharey=ax_main)

h = ax_main.hist2d(x1, x2, bins=20, cmap='viridis')
ax_main.set_xlabel('x1')
ax_main.set_ylabel('x2')

ax_xhist.hist(x1, bins=20, density=True, alpha=0.6)
ax_yhist.hist(x2, bins=20, density=True, orientation='horizontal', alpha=0.6)

ax_xhist.set_title('2D + 1D Histograms')
plt.setp(ax_xhist.get_xticklabels(), visible=False)
plt.setp(ax_yhist.get_yticklabels(), visible=False)
plt.colorbar(h[3], ax=ax_main, label='Counts')

plt.show()