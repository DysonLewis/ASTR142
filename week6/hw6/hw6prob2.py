import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord, AltAz, EarthLocation, get_sun
from astropy.time import Time
from astropy.visualization import quantity_support

# Keck Observatory location
keck = EarthLocation.of_site("Keck Observatory")

# Messier objects
MESSIER_OBJECTS = ["M1", "M13", "M31", "M33", "M42", "M45", "M51", "M57", "M81", "M101"]

def get_targets_coords(target_names):
    coords = {}
    for name in target_names:
        try:
            coords[name] = SkyCoord.from_name(name)
        except Exception as e:
            print(f"Skipping {name}: could not resolve ({e})")
    return coords

def split_at_wrap(x, y, wrap_point=24):
    # Split arrays where x jumps backwards across wrap_point (e.g., 24->0h).
    segments = []
    current_x, current_y = [x[0]], [y[0]]
    for xi, yi in zip(x[1:], y[1:]):
        if (xi - current_x[-1]) < -wrap_point/2:
            segments.append((np.array(current_x), np.array(current_y)))
            current_x, current_y = [xi], [yi]
        else:
            current_x.append(xi)
            current_y.append(yi)
    segments.append((np.array(current_x), np.array(current_y)))
    return segments

def plot_visibility(date_str="2025-11-09", targets=None, site=keck):
    if targets is None:
        targets = get_targets_coords(MESSIER_OBJECTS)

    # Time grid in UT
    midnight = Time(f"{date_str} 00:00:00", scale="utc")
    delta_hours = np.linspace(-6, 18, 500) * u.hour
    times = midnight + delta_hours

    # Alt/Az frame
    frame = AltAz(obstime=times, location=site)
    sun_altaz = get_sun(times).transform_to(frame)
    is_twilight = sun_altaz.alt < 0*u.deg
    is_night = sun_altaz.alt < -18*u.deg

    # LST in hours
    lst_hours = times.sidereal_time('apparent', longitude=site.lon).hour
    lst_hours_wrapped = np.mod(lst_hours, 24)

    # Target altitudes
    alt_dict = {}
    for name, coord in targets.items():
        altaz = coord.transform_to(frame)
        alt_dict[name] = altaz.alt

    with quantity_support():
        # UTC vs Elevation
        fig, ax = plt.subplots(figsize=(12, 6))
        for name, alt in alt_dict.items():
            mask = (alt > 0*u.deg) & is_night
            if np.any(mask):
                ax.plot(times[mask].datetime, alt[mask], label=name, lw=1.5)

        # Sun line
        ax.plot(times.datetime, sun_altaz.alt, color='gold', ls='--', lw=1.2, label='Sun')

        # Shading
        twilight_patch = ax.fill_between(times.datetime, 0, 90, is_twilight, color='0.5', alpha=0.25)
        night_patch = ax.fill_between(times.datetime, 0, 90, is_night, color='k', alpha=0.35)

        ax.set_xlabel("Time [UT]")
        ax.set_ylabel("Elevation [deg]")
        ax.set_ylim(0, 90)
        ax.grid(True, alpha=0.4)

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        handles += [twilight_patch, night_patch]
        labels += ["Twilight", "Night"]
        ax.legend(handles, labels, ncol=2, fontsize=8, loc='upper left')

        plt.tight_layout()
        plt.savefig("hw6prob2_UTC.pdf")
        plt.close()
        print("Saved UTC plot: hw6prob2_UTC.pdf")

        fig, ax = plt.subplots(figsize=(12, 6))

        # Sort LST for shading
        sort_idx = np.argsort(lst_hours_wrapped)
        lst_sorted = lst_hours_wrapped[sort_idx]
        is_twilight_sorted = is_twilight[sort_idx]
        is_night_sorted = is_night[sort_idx]

        # Plot targets with wrap handling
        for name, alt in alt_dict.items():
            mask = (alt > 0*u.deg) & is_night
            if np.any(mask):
                segments = split_at_wrap(lst_hours_wrapped[mask], alt[mask].value)
                for i, (seg_x, seg_y) in enumerate(segments):
                    label = name if i == 0 else None
                    ax.plot(seg_x, seg_y, label=label, lw=1.5)

        # Sun line
        sun_segments = split_at_wrap(lst_hours_wrapped, sun_altaz.alt.value)
        for i, (seg_x, seg_y) in enumerate(sun_segments):
            label = 'Sun' if i == 0 else None
            ax.plot(seg_x, seg_y, color='gold', ls='--', lw=1.2, label=label)

        # Shading by sorted LST
        ax.fill_between(lst_sorted, 0, 90, is_twilight_sorted, color='0.5', alpha=0.25)
        ax.fill_between(lst_sorted, 0, 90, is_night_sorted, color='k', alpha=0.35)

        ax.set_xlabel("Local Sidereal Time [hours]")
        ax.set_ylabel("Elevation [deg]")
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 90)
        ax.set_xticks(np.arange(0, 25, 1))
        ax.grid(True, alpha=0.4)

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        twilight_proxy = plt.Line2D([0], [0], color='0.5', lw=4, alpha=0.25)
        night_proxy = plt.Line2D([0], [0], color='k', lw=4, alpha=0.35)
        handles += [twilight_proxy, night_proxy]
        labels += ["Twilight", "Night"]
        ax.legend(handles, labels, ncol=2, fontsize=8, loc='upper left')

        plt.tight_layout()
        plt.savefig("hw6prob2_LST.pdf")
        plt.close()
        print("Saved LST plot: hw6prob2_LST.pdf")

if __name__ == "__main__":
    plot_visibility()