#
# (c) 2023 Michael Fitzgerald (mpfitz@ucla.edu)
#
# Some code for querying Simbad for making a target list.
#

from astroquery.simbad import Simbad
from astropy import units as u
from astropy.coordinates import SkyCoord

import logging
_log = logging.getLogger('hw4prob1')


def format_target_list(target_list):
    """Query Simbad for list of identifiers; returns dictionary with RA and Dec strings"""

    # Configure Simbad to return RA/DEC in degrees
    custom_simbad = Simbad()
    custom_simbad.add_votable_fields('ra(d)', 'dec(d)')

    target_info = {}

    for target_name in target_list:
        _log.debug(f'querying {target_name}')
        result_table = custom_simbad.query_object(target_name)

        n_result = 0 if result_table is None else len(result_table)
        _log.info(f'{target_name}: {n_result} objects found')

        if n_result == 0:
            _log.warning('skipping....')
            continue
        if n_result > 1:
            _log.warning('using first result')

        # get RA/DEC in degrees
        ra_deg = result_table['ra'][0]
        dec_deg = result_table['dec'][0]

        # build SkyCoord
        c = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)

        # construct sexagesimal strings with controlled precision (4 decimal places for seconds)
        # RA: HHhMMmSS.Ss
        ra_h = int(c.ra.hms.h)
        ra_m = int(c.ra.hms.m)
        ra_s = c.ra.hms.s
        ra_str = f"{ra_h:02d}h{ra_m:02d}m{ra_s:05.4f}s"   # e.g. 12h10m55.69s

        # DEC: +DDdMMmSS.Ss (always show sign)
        dec_sign = '+' if c.dec.deg >= 0 else '-'
        abs_deg = abs(int(c.dec.dms.d))
        dec_m = int(abs(int(c.dec.dms.m)))
        dec_s = abs(c.dec.dms.s)
        dec_str = f"{dec_sign}{abs_deg:02d}d{dec_m:02d}m{dec_s:05.4f}s"  # e.g. +22d42m39.07s

        # left-justify/pad to desired widths when storing
        target_info[target_name] = (ra_str, dec_str)

    # Sort by RA string for ordering
    target_info = dict(sorted(target_info.items(), key=lambda x: x[1]))

    return target_info


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(name)-12s: %(levelname)-8s %(message)s')

    target_list = ['M2', 'M45', 'HD 189733', '3C 273', 'NGC 1068', 'AU Mic', 'TRAPPIST-1']

    target_info = format_target_list(target_list)

    output_fn = 'target_list.txt'
    with open(output_fn, 'w') as f:
        for tn, (ra, dec) in target_info.items():
            print(f"{tn:<20}{ra:<15}{dec:<15}", file=f)

    print(f"Target list written to {output_fn}")
