from astroquery.simbad import Simbad
from astropy import units as u
from astropy.coordinates import SkyCoord

import logging
_log = logging.getLogger('hw4prob1')


def format_target_list(target_list):
    """Query Simbad for list of identifiers; returns dictionary with RA and Dec strings"""

    # Configure Simbad to return RA/DEC
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

        # Extract RA and DEC in degrees from the custom query
        ra_deg = result_table['ra'][0]
        dec_deg = result_table['dec'][0]

        # Convert to SkyCoord and reformat to consistent hms/dms string
        c = SkyCoord(ra=ra_deg*u.degree, dec=dec_deg*u.degree)
        ra_hms, dec_dms = c.to_string('hmsdms').split(' ')

        target_info[target_name] = (ra_hms, dec_dms)

    # Sort by RA
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
