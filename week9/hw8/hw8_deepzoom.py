import numpy as np
import logging
import os
import sys
import multiprocessing as mp
import pyvips
from matplotlib.colors import LinearSegmentedColormap
import webbrowser
import http.server
import socketserver
import threading
import shutil
import gc

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    import mandelbrot
except ImportError:
    print("Error: mandelbrot module not found")
    print("Please run 'make' to compile the C++ extension first")
    sys.exit(1)

_log = logging.getLogger('hw8')

# Mandelbrot calculation parameters
max_iter = 300
r2_max = 1 << 16

# Reference iteration count for color normalization
# Colors will always match this scale regardless of max_iter
# This only changes the color, tbh I just liked how it looks at 100
color_reference = 100

# Define calculation domain and resolution
xmin, xmax = -2.5, 1.
ymin, ymax = -1., 1.
ny, nx = 4*7680, 4*10240
x = np.linspace(xmin, xmax, nx, endpoint=True)
y = np.linspace(ymin, ymax, ny, endpoint=True)

# Setup for chunking the x-axis into columns
ncol = 64
fx = nx//ncol
nc = fx + (fx*ncol < nx)
bx = np.arange(nc, dtype=int)*ncol
ex = np.clip((np.arange(nc, dtype=int)+1)*ncol, 0, nx)


def worker(input, output):
    '''Worker process that computes Mandelbrot values for column chunks'''
    for i, args in iter(input.get, 'STOP'):
        coldata = mandelbrot.calc_val(*args)
        output.put((i, coldata))


def feeder(input):
    '''Feeder process that creates work chunks and queues them for workers'''
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

    _log.info(f'Generating Mandelbrot set at {ny} x {nx} resolution')
    
    # Setup colormap early so workers can use it
    # VScode is great I could just select a color on the graph thingy
    colors = ["#10001F", "#1A0E36", "#001E71", "#007D7D", "#006C7F", 
              "#00B129", "#F2FF00", "#FF6600", "#D60000", "#757575FF"]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('mandelbrot', colors, N=n_bins)
    lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    color_max = float(color_reference)
    
    # Write directly to raw RGB file
    temp_raw = os.path.join(script_dir, 'temp_mandelbrot.raw')
    
    # Pre-allocate the raw file with correct size
    _log.info(f'Pre-allocating raw RGB file ({ny * nx * 3 / (1024**3):.2f} GB)')
    with open(temp_raw, 'wb') as f:
        f.seek(ny * nx * 3 - 1)
        f.write(b'\0')
    
    # Open as memmap for random access writes
    _log.info('Opening raw file as memory-mapped array')
    raw_output = np.memmap(temp_raw, dtype=np.uint8, mode='r+', shape=(ny, nx, 3))

    
    # Setup multiprocessing queues and processes
    n_process = mp.cpu_count()
    n_max = n_process*2
    inqueue = mp.Queue(n_max)
    outqueue = mp.Queue(n_max)


    # Start worker processes to compute Mandelbrot chunks
    for i in range(n_process):
        mp.Process(target=worker, args=(inqueue, outqueue)).start()

    # Start feeder process to generate work
    feedp = mp.Process(target=feeder, args=(inqueue,))
    feedp.start()
    
    # Collect results and write directly to RGB memmap
    _log.info('Computing Mandelbrot and writing RGB data directly')
    
    for j in range(nc):
        i, coldata = outqueue.get()
        _log.debug("received chunk %d/%d" % (i, nc))
        
        # Convert to RGB immediately and write to memmap
        chunk_normalized = np.clip((coldata / color_max) * 255, 0, 255).astype(np.uint8)
        rgb_chunk = lut[chunk_normalized]
        raw_output[:, bx[i]:ex[i], :] = rgb_chunk
        
        del coldata, chunk_normalized, rgb_chunk
        
        # Flush memmap every 200 chunks, probably not needed? At least for <8x res
        if (j + 1) % 200 == 0:
            _log.debug(f"Flushing memmap ({j+1}/{nc})")
            raw_output.flush()
            gc.collect()
    
    # Final flush and close
    _log.info('Flushing final data to disk')
    raw_output.flush()
    del raw_output
    gc.collect()
    
    # Clean up worker processes
    _log.debug('received all chunks; killing workers')
    for i in range(n_process):
        inqueue.put('STOP')
    
    _log.debug('waiting for feeder to finish')
    feedp.join(1.)
    
    _log.info('Creating DeepZoom visualization')
    
    # Configure pyvips to use less memory
    pyvips.cache_set_max(0)  # Disable operation cache
    pyvips.cache_set_max_mem(512 * 1024 * 1024)  # Limit cache to 512MB
    pyvips.cache_set_max_files(0)  # Don't cache file descriptors
    
    # Create temporary TIFF file that we'll stream to
    temp_tiff = os.path.join(script_dir, 'temp_mandelbrot.tiff')
    
    _log.info('Converting raw data to TIFF')
    
    # Load the raw file as memory-mapped array (doesn't load into RAM)
    raw_data = np.memmap(temp_raw, dtype=np.uint8, mode='r', shape=(ny, nx, 3))
    
    # Create pyvips image from the memmap (streams, doesn't copy)
    result_img = pyvips.Image.new_from_memory(
        raw_data.data,
        nx, ny, 3, 'uchar'
    )
    
    # Flip vertically
    _log.info('Flipping image vertically')
    result_img = result_img.flip(pyvips.Direction.VERTICAL)
    
    _log.info(f'Writing temporary TIFF file: {temp_tiff}')
    result_img.write_to_file(temp_tiff, compression='deflate', tile=True, 
                              tile_width=256, tile_height=256, pyramid=True)
    
    del result_img, raw_data
    os.remove(temp_raw)
    gc.collect()
    
    _log.info('TIFF file created successfully')
    
    # Now create DeepZoom from the TIFF file (streaming read)
    _log.info('Creating DeepZoom pyramid from TIFF')
    
    # Clean up existing DeepZoom files if they exist
    dz_dir = os.path.join(script_dir, 'mandelbrot_deepzoom')
    dz_files_dir = dz_dir + '_files'
    if os.path.exists(dz_files_dir):
        _log.info(f'Removing existing DeepZoom directory: {dz_files_dir}')
        shutil.rmtree(dz_files_dir)
    dzi_file = dz_dir + '.dzi'
    if os.path.exists(dzi_file):
        _log.info(f'Removing existing .dzi file: {dzi_file}')
        os.remove(dzi_file)
    
    _log.info(f'Converting TIFF to DeepZoom pyramid: {dz_dir}')
    _log.info('This may take a while...')
    
    # Load TIFF with sequential access (streaming) and convert to DeepZoom
    final_img = pyvips.Image.new_from_file(temp_tiff, access='sequential')
    final_img.dzsave(dz_dir, suffix='.png')
    
    del final_img
    gc.collect()
    
    # Clean up temporary TIFF
    _log.info('Cleaning up temporary TIFF file')
    os.remove(temp_tiff)
    
    # Generate HTML viewer for DeepZoom
    # Lowk had to look this up, idk if it's any good
    _log.info('Generating HTML viewer')
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mandelbrot Set Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background: #000;
        }}
        #viewer {{
            width: 100vw;
            height: 100vh;
            background: #000;
        }}
        .info {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.7);
            color: #fff;
            padding: 10px 15px;
            border-radius: 5px;
            font-size: 14px;
            z-index: 1000;
        }}
    </style>
</head>
<body>
    <div class="info">
        Mandelbrot Set ({nx} × {ny} pixels, {max_iter} iterations)<br>
        Use mouse wheel to zoom, drag to pan
    </div>
    <div id="viewer"></div>
    <script>
        OpenSeadragon({{
            id: "viewer",
            prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
            tileSources: "mandelbrot_deepzoom.dzi",
            showNavigationControl: true,
            navigationControlAnchor: OpenSeadragon.ControlAnchor.TOP_RIGHT,
            animationTime: 0.5,
            blendTime: 0.1,
            constrainDuringPan: false,
            maxZoomPixelRatio: 1000,
            minZoomLevel: 0.8,
            visibilityRatio: 1,
            zoomPerScroll: 1.2,
            timeout: 120000
        }});
    </script>
</body>
</html>"""

    html_fn = os.path.join(script_dir, 'mandelbrot_viewer.html')
    with open(html_fn, 'w') as f:
        f.write(html_content)

    _log.info('DeepZoom pyramid saved successfully')
    _log.info(f'HTML viewer created: {html_fn}')
    _log.info(f'Open {html_fn} in your browser to view the Mandelbrot set')

    # Start local web server and open browser
    _log.info('Starting local web server')
    PORT = 8000
    
    class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Suppress HTTP request logs
    
    os.chdir(script_dir)
    Handler = QuietHTTPRequestHandler
    
    try:
        httpd = socketserver.TCPServer(("", PORT), Handler)
        _log.info(f'Web server running at http://localhost:{PORT}')
        
        # Start server in background thread
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        
        # Open browser
        url = f'http://localhost:{PORT}/mandelbrot_viewer.html'
        _log.info(f'Opening browser to {url}')
        webbrowser.open(url)
        
        _log.info('Press Ctrl+C to stop the server and exit')
        
        # Keep server running
        try:
            server_thread.join()
        except KeyboardInterrupt:
            _log.info('Shutting down server')
            httpd.shutdown()
            
    except OSError as e:
        _log.warning(f'Could not start server on port {PORT}: {e}')
        _log.info(f'You can manually run: python3 -m http.server {PORT}')
        _log.info(f'Then open: http://localhost:{PORT}/mandelbrot_viewer.html')