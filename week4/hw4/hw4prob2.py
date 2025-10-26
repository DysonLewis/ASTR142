#
# (c) 2025 Michael Fitzgerald (mpfitz@ucla.edu)
#
# Some code for querying the virtual observatory to make finder charts
#

import numpy as np
import pyvo as vo
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.coordinates as coord
from astropy.io import fits
from astropy.wcs import WCS


# set up VO service for DSS image retrieval
# see https://nasa-navo.github.io/workshop-notebooks/CS_Image_Access.html
image_services = vo.regsearch(servicetype='sia',waveband='optical')
#  this next line will print out available services.  See which one is appropriate.
#image_services.to_table()['ivoid','short_name','res_title'].pprint(max_lines=-1, max_width=-1)
dss_service = image_services[''] # FIXME

# set up service for IRSA 2MASS point-source catalog retrieval
psc_services = vo.regsearch(servicetype='tap',waveband='infrared')
#  this next line will print out available services.  See which one is appropriate.
#psc_services.to_table()['ivoid','short_name','res_title'].pprint(max_lines=-1, max_width=-1)
psc_service = psc_services[''] # FIXME

# loop over sources
target_list = ['M2',
               'M45',
               'HD 189733',
               '3C 273',
               'NGC 1068',
               'AU Mic',
               'TRAPPIST-1',
               ]
for target in target_list:
    print('making finder chart for {}'.format(target))

    # compute object coordinates
    coords = coord.SkyCoord.from_name(target)
    
    # retrieve image
    #   Use the service to look up the URL to a FITS file for the image
    try:
        im_table = dss_service.search(pos=coords, size=0.2)
    except:
        print('error retrieving image for {}!  skipping...'.format(target))
        continue
    im_url = im_table[0].getdataurl()
    #   Open the fits file and retrieve the image data and header info
    with fits.open(im_url, ignore_missing_simple=True) as hl:
        imdata = hl[0].data
        hdr = hl[0].header
    w = WCS(hdr) # creates world coordinate system object for coordinate transformations

    # retrieve point sources
    rad = 3. / 60. # [deg]  cone search radius
    jlim = 15. # [mag]  J-band magnitude limit
    # NOTE  you will need to update this to retrieve only bright sources, with J-band magnitudes brighter than the limit defined above
    query_str = '''SELECT *
FROM fp_psc
WHERE CONTAINS(POINT('ICRS',ra, dec), CIRCLE('ICRS',{},{},{}))=1
AND ????? {}
'''.format(coords.ra.deg, coords.dec.deg, rad, jlim)
    sources = psc_service.service.run_async(query_str).to_table()  # performs the query and produces an astropy table
    #sources.pprint() # optional print out source list

    # get point source locations in image coordinates
    c = coord.SkyCoord(sources['ra'], sources['dec'], frame='icrs', unit='deg')
    x, y = w.world_to_pixel(c)
    
    # NOTE: This is not an ideal way to produce these plots.  Here
    # we're putting everything in pixel coordinates.  Later we'll see
    # how to plot things in equatorial or other coordinate systems.
    
    # set up figure
    fig = plt.figure(0)
    fig.clear()
    ax = fig.add_subplot(111)

    # display image
    ax.imshow(np.log(np.clip(imdata,1,np.inf)), # logarithmic scaling.  May need to adjust clip level.
              cmap=mpl.cm.gray_r,
              interpolation='nearest',
              )

    # display point sources
    ax.scatter(x, y,
               marker='o',
               edgecolor='k',
               facecolor='none',
               )

    # annotate
    ax.set_title(target)

    # save figure
    fn = 'finder-{}.pdf'.format(target.replace(' ', ''))
    fig.savefig(fn)
