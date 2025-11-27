import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from datetime import datetime
import logging
import sys

def process_order(ordernumber, basename='/home/dyson/fall25/142ASTR/week8/Disc15/discussion15_activity_data/WASP33_nov21_rereduced_fluxes_coadded_order_', plot=False):
    """Process a single order and save the result."""
    data = np.load(basename+str(ordernumber)+'.npy')
    logging.info('Loaded file: ' + basename+str(ordernumber)+'.npy')
    
    if plot:
        plt.imshow(data, aspect=15, vmin=0, vmax=1e4)
        plt.show()
    
    ### Scale each exposure to a consistent median
    for i in range(data.shape[0]):
        data[i] = data[i]/np.nanmedian(data[i])
    
    ### Now we want to make the median over the time series (y-axis)
    medspec = np.nanmedian(data, axis=0)
    
    ### Now divide each spectrum in the time series by its median
    for i in range(data.shape[0]):
        data[i] = data[i]/medspec
    
    ### And let's mask outliers. Use a boolean index for speed
    threshold = 6*median_abs_deviation(data.flatten())
    data[np.abs(data-1)>threshold] = np.nan	

    if plot:
        plt.imshow(data, aspect=15, vmin=0.95, vmax=1.05)
        plt.show()
    
    logging.info('Done with ' + basename+str(ordernumber)+'.npy')
    return data

if __name__ == '__main__':
    # Check if order number is provided as command line argument
    if len(sys.argv) > 1:
        # Single order mode - for use with GNU parallel
        order = int(sys.argv[1])
        logging.basicConfig(
            filename=f'logfile_order_{order}.log',
            level=logging.INFO
        )
        
        t_start = datetime.now()
        logging.info(f'Starting order {order}')
        
        data = process_order(order)
        
        # Save the output
        output_file = f'WASP33_nov21_processed_order{order}.npy'
        np.save(output_file, data)
        logging.info(f'Saved {output_file}')
        
        t_end = datetime.now()
        logging.info(f'Order {order} time: {t_end - t_start}')
        
    else:
        # Run all orders sequentially for comparison
        logging.basicConfig(filename='logfile.log', level=logging.INFO)
        orders = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        
        t1 = datetime.now()
        logging.info('Starting sequential processing')
        for order in orders:
            data = process_order(order)
            np.save(f'WASP33_nov21_processed_order{order}.npy', data)
        t2 = datetime.now()
        logging.info(f'Sequential time: {t2-t1}')