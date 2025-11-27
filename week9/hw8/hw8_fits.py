import numpy as np
import logging
import os
import sys
import multiprocessing as mp
import gc
from matplotlib.colors import LinearSegmentedColormap
import pyvips

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    import mandelbrot
except ImportError:
    print("Error: mandelbrot module not found")
    print("Please run 'make' to compile the C++ extension first")
    sys.exit(1)

_log = logging.getLogger('hw8')

max_iter = 100
r2_max = 1 << 16

xmin, xmax = -2.5, 1.
ymin, ymax = -1., 1.
ny, nx = 7680, 10240
x = np.linspace(xmin, xmax, nx, endpoint=True)
y = np.linspace(ymin, ymax, ny, endpoint=True)

ncol = 16
fx = nx//ncol
nc = fx + (fx*ncol < nx)
bx = np.arange(nc, dtype=int)*ncol
ex = np.clip((np.arange(nc, dtype=int)+1)*ncol, 0, nx)


def worker(input, output):
    for i, args in iter(input.get, 'STOP'):
        coldata = mandelbrot.calc_val(*args)
        output.put((i, coldata))


def feeder(input):
    for i in range(nc):
        xx, yy = np.meshgrid(x[bx[i]:ex[i]], y)
        args = (xx, yy)
        input.put((i, args), True)
    _log.debug('feeder finished')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(name)-12s: %(levelname)-8s %(message)s',
                        )
    
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

    from astropy.io import fits
    
    out_fn = os.path.join(script_dir, 'output.fits')
    
    _log.info(f'Creating FITS file: {out_fn}')
    _log.info(f'Resolution: {ny} x {nx}')
    
    dum = np.zeros((100, 100), dtype=np.float64)
    hdu = fits.PrimaryHDU(data=dum)
    header = hdu.header
    while len(header) < (36 * 4 - 1):
        header.append()
    header['NAXIS1'] = nx
    header['NAXIS2'] = ny
    header.tofile(out_fn, overwrite=True)
    shape = tuple(header[f'NAXIS{ii}'] for ii in range(1, header['NAXIS']+1))
    with open(out_fn, 'rb+') as fobj:
        fobj.seek(len(header.tostring()) + (np.prod(shape) * np.abs(header['BITPIX']//8)) - 1)
        fobj.write(b'\0')

    n_process = mp.cpu_count()
    n_max = n_process*2
    inqueue = mp.Queue(n_max)
    outqueue = mp.Queue(n_max)

    for i in range(n_process):
        mp.Process(target=worker, args=(inqueue, outqueue)).start()

    feedp = mp.Process(target=feeder, args=(inqueue,))
    feedp.start()
    
    _log.info('Computing Mandelbrot values')
    
    with fits.open(out_fn, mode='update', memmap=True) as hdul:
        out_im = hdul[0].data

        for j in range(nc):
            i, coldata = outqueue.get()
            _log.debug("received chunk %d/%d" % (i, nc))
            out_im[:, bx[i]:ex[i]] = coldata
            
            '''
            # Don't need this tbh, creating the .png uses way more memory than this will ever use
            # This is mostly CPU bound
            if (j + 1) % 50 == 0:
                _log.debug(f"Flushing FITS data ({j+1}/{nc})")
                hdul.flush()
                gc.collect()
            '''
            
    _log.debug('received all chunks; killing workers')
    for i in range(n_process):
        inqueue.put('STOP')

    _log.debug('waiting for feeder to finish')
    feedp.join(1.)
    
    _log.info(f'FITS file created: {out_fn}')
    
    _log.info('Converting FITS to PNG')
    
    colors = ["#10001F", "#1A0E36", "#001E71", "#007D7D", "#006C7F", 
              "#00B129", "#F2FF00", "#FF6600", "#D60000", "#757575FF"]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('mandelbrot', colors, N=n_bins)
    lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    
    png_fn = os.path.join(script_dir, 'mandelbrot.png')
    
    _log.info('Using pyvips for PNG conversion')
    
    pyvips.cache_set_max(0)
    pyvips.cache_set_max_mem(512 * 1024 * 1024)
    pyvips.cache_set_max_files(0)
    
    with fits.open(out_fn, memmap=True) as hdul:
        data = hdul[0].data
        
        chunk_size = 256
        temp_raw = os.path.join(script_dir, 'temp_rgb.raw')
        
        _log.info('Converting to RGB in chunks')
        with open(temp_raw, 'wb') as f:
            for row_start in range(0, ny, chunk_size):
                row_end = min(row_start + chunk_size, ny)
                chunk = data[row_start:row_end, :]
                
                normalized = np.clip((chunk / max_iter) * 255, 0, 255).astype(np.uint8)
                rgb_chunk = lut[normalized]
                
                f.write(rgb_chunk.tobytes())
                
                del chunk, normalized, rgb_chunk
                
                '''
                # Also don't need this, only need for like >4x resolution
                if (row_start // chunk_size + 1) % 10 == 0:
                    _log.debug(f"Processed {row_end}/{ny} rows")
                    gc.collect()
                '''
                
    _log.info('Creating PNG from RGB data')
    rgb_data = np.memmap(temp_raw, dtype=np.uint8, mode='r', shape=(ny, nx, 3))
    
    img = pyvips.Image.new_from_memory(
        rgb_data.data,
        nx, ny, 3, 'uchar'
    )
    
    img = img.flip(pyvips.Direction.VERTICAL)
    
    _log.info(f'Writing PNG file: {png_fn}')
    img.write_to_file(png_fn, compression=6)
    
    del img, rgb_data
    os.remove(temp_raw)
    gc.collect()
    
    _log.info(f'PNG file created: {png_fn}')
    _log.info('Done')