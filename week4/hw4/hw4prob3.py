#
# (c) 2023 Michael Fitzgerald (mpfitz@ucla.edu)
#
# Some code for querying Vizier for catalog to construct a CMD.
#


from astroquery.vizier import Vizier

catalog_name = 
cluster_name = 
Vizier.ROW_LIMIT = 100
#Vizier.ROW_LIMIT = 1000
#Vizier.ROW_LIMIT = -1 # get all sources
result = Vizier.query_constraints(????)

# parse the result to get the source list


vi = 
vmag = 


# plot the color-magnitude diagram
import matplotlib as mpl
import pylab
fig = pylab.figure(0)
fig.clear()
ax = fig.add_subplot(111)

ax.scatter(vi, vmag,
           marker='.',
           c='k',
           s=1., # experiment with marker size
           )

# plot title and axes labels

# invert the y axis


pylab.draw()
pylab.show()
fig.savefig('hw4prob2.pdf', dpi=300)

