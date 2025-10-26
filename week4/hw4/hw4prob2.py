import numpy as np
import pyvo as vo
import matplotlib.pyplot as plt
import astropy.coordinates as coord
from astropy.io import fits
from astropy.wcs import WCS

# Set up VO service for DSS image retrieval
image_services = vo.regsearch(servicetype='sia', waveband='optical')
dss_service = image_services['ivo://archive.stsci.edu/ghosts']

# Set up service for IRSA 2MASS point-source catalog retrieval
psc_services = vo.regsearch(servicetype='tap', waveband='infrared')
# pick the 2MASS PSC service
psc_service = psc_services[2]  # adjust index if needed

# Target list
target_list = ['M2', 'M45', 'HD 189733', '3C 273', 'NGC 1068', 'AU Mic', 'TRAPPIST-1']

for target in target_list:
    print(f'Making finder chart for {target}')
    try:
        # compute object coordinates
        coords = coord.SkyCoord.from_name(target)

        # retrieve DSS image
        im_table = dss_service.search(pos=coords, size=0.2)
        im_url = im_table[0].getdataurl()
        with fits.open(im_url, ignore_missing_simple=True) as hl:
            imdata = hl[0].data
            hdr = hl[0].header
        w = WCS(hdr)

        # retrieve bright 2MASS point sources
        rad = 3. / 60.  # [deg]
        jlim = 15.  # [mag]
        query_str = f'''
        SELECT *
        FROM fp_psc
        WHERE CONTAINS(POINT('ICRS',ra, dec), CIRCLE('ICRS',{coords.ra.deg},{coords.dec.deg},{rad}))=1
        AND j_m < {jlim}
        '''
        sources = psc_service.service.run_async(query_str).to_table()

        # convert point source coordinates to pixel coordinates
        c = coord.SkyCoord(sources['ra'], sources['dec'], frame='icrs', unit='deg')
        x, y = w.world_to_pixel(c)

        # set up figure
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.imshow(np.log(np.clip(imdata, 1, np.inf)),
                  cmap='gray_r',
                  origin='lower')
        ax.scatter(x, y, marker='o', edgecolor='k', facecolor='none')
        ax.set_title(target)

        # save figure
        fn = f'finder-{target.replace(" ", "")}.pdf'
        fig.savefig(fn)
        plt.close(fig)

    except Exception as e:
        print(f'Error retrieving data for {target}: {e}')
        continue
