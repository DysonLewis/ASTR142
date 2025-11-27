# (c) 2023 Michael Fitzgerald (mpfitz@ucla.edu)
#
# Using parametric data for a single cycle imported from another program, repeates the
# cycle and adds a scale factor to the parametric curve.  Plots the result.
#
# Demonstrates loading a tabular-formatted text file output from C, plotting a curve,
# annotating a plot with text, and exporting a figure to PDF.
#

import numpy as np
import matplotlib as mpl
import pylab
from astropy.io import ascii

n_cyc = 20 # number of cycles

# load previously generated data from C program
fn = 'hw1prob1.txt' # filename
dat = ascii.read(fn, names=('t','x','y'))

n_pt_cyc = len(dat) # number of points per cycle
n_pt = n_cyc*n_pt_cyc # total number of points

# replicate the x and y arrays n_cyc times, keep in single 1d array
xx = np.concatenate([dat['x']]*n_cyc, axis=0)
yy = np.concatenate([dat['y']]*n_cyc, axis=0)

# scale factor
a_min, a_max = 1., 1.3 # range
a_pow = 2. # power-law index
a = a_min*np.exp(np.linspace(0., 1., n_pt)**a_pow * np.log(a_max/a_min))

# 2d points along parametric curve (apply scale factor)
y = a*yy
x = a*xx

# plot points
fig = pylab.figure(figsize=(5.5,5.5))
ax = fig.add_subplot(111, aspect='equal')
ax.plot(x, y, c='r')

# add annotation
ax.text(-1.5, 1.5, "I")
ax.text(1.25, -1.25, "Astro 142!")

# show/save figure
#pylab.draw()
#pylab.show()
fig.savefig("hw1prob1.pdf")
