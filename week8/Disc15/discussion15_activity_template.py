import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from multiprocessing import Pool, cpu_count, current_process
from datetime import datetime
import logging
import os
import re
from functools import partial
import warnings

# suppress only the specific NaN warnings
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

def process_order(ordernumber,
                  basename='WASP33_nov21_rereduced_fluxes_coadded_order_',
                  plot=False):
    
    # Get process info
    proc = current_process()
    proc_name = proc.name
    proc_pid = proc.pid
    
    # Display which core is processing this order
    print(f"[{proc_name} - PID {proc_pid}] Processing Order {ordernumber}")
    
    fname = basename + str(ordernumber) + '.npy'
    data = np.load(fname)
    logging.info(f'[{proc_name}] Loaded file: {fname}')

    for i in range(data.shape[0]):
        med = np.nanmedian(data[i])
        if not np.isnan(med) and med != 0:
            data[i] /= med

    medspec = np.nanmedian(data, axis=0)

    # Identify invalid wavelength bins (all NaN)
    valid_cols = ~np.isnan(medspec)
    n_bad = np.sum(~valid_cols)

    if n_bad > 0:
        logging.info(f'[{proc_name}] Order {ordernumber}: removed {n_bad} invalid all-NaN columns')

    # Remove invalid columns from data and medspec
    data = data[:, valid_cols]
    medspec = medspec[valid_cols]
    for i in range(data.shape[0]):
        data[i] /= medspec

    flat = data.flatten()
    threshold = 6 * median_abs_deviation(flat, nan_policy='omit')

    if threshold == 0 or np.isnan(threshold):
        threshold = 0.1  # fallback safety valve

    data[np.abs(data - 1) > threshold] = np.nan
    if plot:
        plt.imshow(data, aspect=15, vmin=0.95, vmax=1.05)
        plt.title(f"Cleaned Order {ordernumber}")
        plt.show()

    print(f"[{proc_name} - PID {proc_pid}] ✓ Completed Order {ordernumber}")
    logging.info(f'[{proc_name}] Done with {fname}')
    return ordernumber, data


if __name__ == '__main__':
    logging.basicConfig(filename='logfile.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')

    # Resolve program directory and data directory (robust for interactive sessions)
    try:
        program_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        program_dir = os.getcwd()
    data_dir = os.path.join(program_dir, 'discussion15_activity_data')

    if not os.path.isdir(data_dir):
        logging.error(f'Data directory not found: {data_dir}')
        raise SystemExit(f'Data directory not found: {data_dir}')

    # Change working dir to data_dir so np.load can open relative filenames easily
    os.chdir(data_dir)
    logging.info(f'Processing .npy files in {data_dir}')

    # Default basename used by your function
    BASE = 'WASP33_nov21_rereduced_fluxes_coadded_order_'

    # Find .npy files
    all_npy = [f for f in os.listdir('.') if f.endswith('.npy')]
    if not all_npy:
        logging.error('No .npy files found in data directory')
        raise SystemExit('No .npy files found in data directory')

    # Prefer files that start with the expected BASE; otherwise attempt to infer order numbers
    matching = [f for f in all_npy if f.startswith(BASE)]
    orders = []
    if matching:
        for f in matching:
            m = re.search(r'(\d+)\.npy$', f)
            if m:
                orders.append(int(m.group(1)))
    else:
        # fallback: try to find digits before .npy and use unique numbers
        for f in all_npy:
            m = re.search(r'(\d+)\.npy$', f)
            if m:
                orders.append(int(m.group(1)))
        # if still nothing, try to parse any trailing number after last underscore
        if not orders:
            for f in all_npy:
                parts = os.path.splitext(f)[0].split('_')
                if parts and parts[-1].isdigit():
                    orders.append(int(parts[-1]))

    orders = sorted(set(orders))
    if not orders:
        logging.error('Could not infer order numbers from filenames.')
        raise SystemExit('Could not infer order numbers from filenames.')

    logging.info(f'Found orders: {orders}')

    # Use multiprocessing Pool to process orders in parallel
    nprocs = min(cpu_count(), len(orders))
    print(f'\n{"="*60}')
    print(f'PARALLEL PROCESSING SUMMARY')
    print(f'{"="*60}')
    print(f'Total CPU cores available: {cpu_count()}')
    print(f'Workers to use: {nprocs}')
    print(f'Orders to process: {len(orders)} -> {orders}')
    print(f'{"="*60}\n')
    
    logging.info(f'Using {nprocs} parallel workers')
    with Pool(processes=nprocs) as pool:
        # We run process_order with the expected BASE; process_order expects files in current cwd (data_dir)
        worker = partial(process_order, basename=BASE, plot=False)
        results = pool.map(worker, orders)

    print(f'\n{"="*60}')
    print(f'All processing complete!')
    print(f'{"="*60}\n')

    # results is a list of (order, data)
    processed = {f'order_{order}': data for order, data in results}

    # Save to compressed npz in program_dir (so outputs are next to the script)
    outpath = os.path.join(program_dir, 'processed_orders.npz')
    # np.savez_compressed accepts keyword args, so expand a dict
    np.savez_compressed(outpath, **processed)
    logging.info(f'Saved processed arrays to {outpath}')

    # Make a simple plot: median spectrum per order (wavelength index on x-axis)
    medians = []
    order_labels = []
    for order in sorted(orders):
        arr = processed[f'order_{order}']
        # compute median spectrum along time axis
        med = np.nanmedian(arr, axis=0)
        medians.append(med)
        order_labels.append(str(order))

    plt.figure(figsize=(10, 6))
    for med, lbl in zip(medians, order_labels):
        x = np.arange(med.size)
        plt.plot(x, med, label=f'order {lbl}', linewidth=0.8)
    plt.xlabel('Pixel / Wavelength index')
    plt.ylabel('Median (normalized)')
    plt.title('Median spectrum per order (processed)')
    plt.legend(fontsize='small', ncol=2)
    figpath = os.path.join(program_dir, 'median_spectra.png')
    plt.tight_layout()
    plt.savefig(figpath, dpi=150)
    plt.close()
    logging.info(f'Saved median spectra plot to {figpath}')

    print('Done.')
    print(f'Processed {len(orders)} orders. Saved: {outpath} and {figpath}')