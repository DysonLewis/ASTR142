#
# (c) 2023 Michael Fitzgerald (mpfitz@ucla.edu)
#
# Some code for querying Vizier for catalog to construct a CMD.
#

import numpy as np
import pyvo as vo
import matplotlib.pyplot as plt
import astropy.coordinates as coord
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.skyview import SkyView
import os

# Set up VO service for SDSS DR3 g-band
image_services = vo.regsearch(servicetype='sia', waveband='optical')
dss_service = image_services['ivo://sdss.jhu/services/siapdr3-g']

# Set up service for IRSA 2MASS point-source catalog retrieval
psc_services = vo.regsearch(servicetype='tap', waveband='infrared')
psc_service = psc_services[0]

# Target list
target_list = ['M2', 'M45', 'HD 189733', '3C 273', 'NGC 1068', 'AU Mic', 'TRAPPIST-1']

# Directory for images
output_dir = 'hw4prob2_images'

for target in target_list:
    print(f"Making finder chart for {target}")
    try:
        # Get coordinates
        coords = coord.SkyCoord.from_name(target)
        
        # Try SDSS first
        try:
            im_table = dss_service.search(pos=coords, size=0.2)
            if len(im_table) > 0:
                im_url = im_table[0].getdataurl()
                with fits.open(im_url, ignore_missing_simple=True) as hl:
                    imdata = hl[0].data
                    hdr = hl[0].header
            else:
                raise ValueError("No SDSS image found; using SkyView fallback")
        except Exception:
            # Fallback: SkyView DSS
            fits_files = SkyView.get_images(position=coords, survey='DSS', coordinates='J2000', pixels=300)
            if len(fits_files) == 0:
                print(f"No DSS image available for {target}, skipping...")
                continue
            imdata = fits_files[0][0].data
            hdr = fits_files[0][0].header

        w = WCS(hdr)

        # Query IRSA 2MASS bright sources
        rad = 3.0 / 60.0  # [deg]
        jlim = 15.0       # magnitude limit
        query_str = f"""
        SELECT *
        FROM fp_psc
        WHERE CONTAINS(POINT('ICRS',ra, dec), CIRCLE('ICRS',{coords.ra.deg},{coords.dec.deg},{rad}))=1
        AND j_m < {jlim}
        """
        sources = psc_service.service.run_async(query_str).to_table()
        if len(sources) == 0:
            print(f"No bright 2MASS sources found near {target}")
        
        # Convert to pixel coordinates
        if len(sources) > 0:
            c = coord.SkyCoord(sources['ra'], sources['dec'], frame='icrs', unit='deg')
            x, y = w.world_to_pixel(c)
        else:
            x, y = [], []

        # Plot
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.imshow(np.log(np.clip(imdata, 1, np.inf)), cmap='gray_r', origin='lower')
        if len(x) > 0:
            ax.scatter(x, y, marker='o', edgecolor='k', facecolor='none')
        ax.set_title(target)

        # Save figure to output directory
        fn = os.path.join(output_dir, f"finder-{target.replace(' ', '')}.pdf")
        fig.savefig(fn)
        plt.close(fig)

    except Exception as e:
        print(f"Error processing {target}: {e}")
        continue
