# (c) 2023 Michael Fitzgerald (mpfitz@ucla.edu)
#
# Some code for displaying 2d image comparisons between model and data.
#
#

import numpy as np
import matplotlib as mpl
import pylab

import logging
_log = logging.getLogger('MYLOG') # FIXME  rename this


# TEMPORARY -- remove when making into a module
ny, nx = 364, 512
seed = 23579
rng = np.random.default_rng(seed=seed)
data = rng.normal(size=(ny,nx))
model = rng.normal(size=(ny,nx))



# -- here is the code for displaying the 2-panel plot

# should check that data and model are the same size

fignum = 0 # default figure number

n_panel = 2
print('Creating {}-panel plot in figure {}'.format(n_panel, fignum))

ax_aspect = data.shape[1]/data.shape[0] # aspect ratio of axes (single panel)


# code for figuring out figure and axes dimensions given number of panels and axes aspect ratio
figwidth = 6.5 # [in]
t_margin = 0.4 # [in]  top margin
b_margin = 0.2 # [in]  bottom margin
l_margin = 0.2 # [in]  left margin
r_margin = 0.2 # [in]  right margin
left = l_margin/figwidth # normalized units
right = 1.-r_margin/figwidth # normalized units
dx = (right-left)*figwidth/n_panel # [in]  axes width
dy = dx/ax_aspect # [in]  axes height
figheight = b_margin+t_margin + dy # [in]
bottom = b_margin/figheight
top = 1.-t_margin/figheight
ax_dx = dx/figwidth # convert to normalized units
ax_dy = dy/figheight

# these are the critical dimensions for creating the figure
ax_dims = []
for i in range(n_panel):
    ax_dims.append((left+i*ax_dx, bottom, ax_dx, ax_dy))
figsize = (figwidth, figheight)

# display range of data
dm_vmin = np.min((data.min(), model.min()))
dm_vmax = np.max((data.max(), model.max()))

# check that the vmax is more than the vmin
# this should be an error
if dm_vmax < dm_vmin: print('vmax < vmin for data or model')


print("Creating figure {:.1f} in by {:.1f} in".format(figheight, figwidth))
if (figheight/figwidth < 0.2): print('figure might be too skinny') # warning
fig = pylab.figure(fignum, figsize=figsize)
ax1 = fig.add_axes(ax_dims[0])
ax2 = fig.add_axes(ax_dims[1])

dm_kw = {'interpolation':'nearest',
         'vmin':dm_vmin,
         'vmax':dm_vmax,
         'cmap':mpl.cm.jet,
         }
# display the arrays
ax1.imshow(data, **dm_kw)
ax2.imshow(model, **dm_kw)

# suppress the labels
for ax in (ax1, ax2):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

# set titles
ax1.set_title('data')
ax2.set_title('model')

# draw/show the plot
pylab.draw()
pylab.show()










# -- here is the code for displaying the 3-panel plot

# should check that data and model are the same size

fignum = 1 # default figure number
resid = data-model

n_panel = 3
print('Creating {}-panel plot in figure {}'.format(n_panel, fignum))

ax_aspect = data.shape[1]/data.shape[0] # aspect ratio of axes (single panel)


# code for figuring out figure and axes dimensions given number of panels and axes aspect ratio
figwidth = 6.5 # [in]
t_margin = 0.4 # [in]  top margin
b_margin = 0.2 # [in]  bottom margin
l_margin = 0.2 # [in]  left margin
r_margin = 0.2 # [in]  right margin
left = l_margin/figwidth # normalized units
right = 1.-r_margin/figwidth # normalized units
dx = (right-left)*figwidth/n_panel # [in]  axes width
dy = dx/ax_aspect # [in]  axes height
figheight = b_margin+t_margin + dy # [in]
bottom = b_margin/figheight
top = 1.-t_margin/figheight
ax_dx = dx/figwidth # convert to normalized units
ax_dy = dy/figheight

# these are the critical dimensions for creating the figure
ax_dims = []
for i in range(n_panel):
    ax_dims.append((left+i*ax_dx, bottom, ax_dx, ax_dy))
figsize = (figwidth, figheight)

# display range of data
dm_vmin = np.min((data.min(), model.min()))
dm_vmax = np.max((data.max(), model.max()))
r_vmin, r_vmax = resid.min(), resid.max()

# check that the vmax is more than the vmin
# these should be errors
if dm_vmax < dm_vmin: print('vmax < vmin for data or model')
if r_vmax < r_vmin: print('vmax < vmin for residual')


print("Creating figure {:.1f} in by {:.1f} in".format(figheight, figwidth))
if (figheight/figwidth < 0.2): print('figure might be too skinny') # warning
fig = pylab.figure(fignum, figsize=figsize)
ax1 = fig.add_axes(ax_dims[0])
ax2 = fig.add_axes(ax_dims[1])
ax3 = fig.add_axes(ax_dims[2])

dm_kw = {'interpolation':'nearest',
         'vmin':dm_vmin,
         'vmax':dm_vmax,
         'cmap':mpl.cm.jet,
         }
r_kw = {'interpolation':'nearest',
        'vmin':r_vmin,
        'vmax':r_vmax,
        'cmap':mpl.cm.RdBu,
        }
# display the arrays
ax1.imshow(data, **dm_kw)
ax2.imshow(model, **dm_kw)
ax3.imshow(resid, **r_kw)

# suppress the labels
for ax in (ax1, ax2, ax3):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

# set titles
ax1.set_title('data')
ax2.set_title('model')
ax3.set_title('residual')

# draw/show the plot
pylab.draw()
pylab.show()


if __name__ == '__main__':
    # this code only gets executed when this file is run as a script, not when it is imported as a module.  This is a good place for test function calls.

    #logging.basicConfig(level=logging.INFO,
    logging.basicConfig(level=logging.DEBUG,
                        format='%(name)-12s: %(levelname)-8s %(message)s',
                        )

    
    print('running tests')

    # test code here
    
