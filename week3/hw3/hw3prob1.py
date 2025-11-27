# (c) 2023 Michael Fitzgerald (mpfitz@ucla.edu)
#
# Some code for displaying 2d image comparisons between model and data.
#
#
'''
This module generates 2-panel and 3-panel comparison plots for 2D numerical datasets
(such as images or simulation outputs) using Matplotlib. It includes functions to:

- Generate test data (or load external data)
- Plot data vs model in a 2-panel layout
- Plot data, model, and residual in a 3-panel layout
- Validate data/model shapes and value ranges
- Log informative messages and handle errors gracefully

Intended as a module but includes a main section for testing.
'''

import numpy as np
import matplotlib as mpl
import pylab
import logging
from logging.handlers import RotatingFileHandler

# Configure module-level logger
logger = logging.getLogger(__name__)

# ----------------------------- Helper Functions ----------------------------- #

def generate_test_data(ny=364, nx=512, seed=23579):
    '''
    Generate deterministic test data and model arrays using a normal distribution.

    Parameters
    ----------
    ny : int
        Number of rows.
    nx : int
        Number of columns.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    data : ndarray
        Generated data array.
    model : ndarray
        Generated model array.
    '''
    rng = np.random.default_rng(seed=seed)
    data = rng.normal(size=(ny, nx))
    model = rng.normal(size=(ny, nx))
    logger.debug(f"Generated test data and model with shape ({ny}, {nx})")
    return data, model


def validate_data_shapes(data, model):
    '''
    Ensure data and model arrays have the same shape.

    Raises
    ------
    ValueError
        If the shapes of the arrays differ.
    '''
    if data.shape != model.shape:
        logger.error(f"Shape mismatch: data {data.shape}, model {model.shape}")
        raise ValueError("Data and model must have the same shape")
    logger.debug(f"Validated data/model shape: {data.shape}")


def compute_display_range(*arrays):
    '''
    Compute the overall min and max display values across multiple arrays.

    Returns
    -------
    vmin, vmax : float
        Minimum and maximum values.

    Raises
    ------
    ValueError
        If vmax < vmin or if arrays contain only NaNs.
    '''
    try:
        vmin = min(np.nanmin(arr) for arr in arrays)
        vmax = max(np.nanmax(arr) for arr in arrays)
    except ValueError as e:
        logger.error(f"Failed to compute display range: {e}")
        raise

    if np.isnan(vmin) or np.isnan(vmax):
        logger.error("NaN detected in display range")
        raise ValueError("Cannot compute display range: array contains only NaNs")

    if vmax < vmin:
        logger.error(f"Invalid display range: vmax={vmax} < vmin={vmin}")
        raise ValueError("vmax < vmin for provided arrays")

    logger.debug(f"Computed display range: vmin={vmin:.3f}, vmax={vmax:.3f}")
    return vmin, vmax



def compute_figure_layout(n_panel, ax_aspect, figwidth=6.5,
                          t_margin=0.4, b_margin=0.2, l_margin=0.2, r_margin=0.2):
    '''
    Compute figure size and axis dimensions for a panel layout.

    Returns
    -------
    figsize : tuple
        Figure size (width, height) in inches.
    ax_dims : list of tuple
        List of normalized axis dimensions.
    '''
    left = l_margin / figwidth
    right = 1. - r_margin / figwidth
    dx = (right - left) * figwidth / n_panel
    dy = dx / ax_aspect
    figheight = b_margin + t_margin + dy
    bottom = b_margin / figheight
    ax_dx = dx / figwidth
    ax_dy = dy / figheight

    ax_dims = [(left + i * ax_dx, bottom, ax_dx, ax_dy) for i in range(n_panel)]
    figsize = (figwidth, figheight)

    if figheight / figwidth < 0.2:
        logger.warning(f"Figure might be too skinny: height/width = {figheight/figwidth:.2f}")

    logger.debug(f"Figure layout computed for {n_panel} panels, figsize={figsize}")
    return figsize, ax_dims


# ----------------------------- Plotting Functions ----------------------------- #

def plot_two_panel(data, model, fignum=0):
    '''
    Plot data and model side-by-side in a 2-panel figure.
    '''
    logger.info("Creating 2-panel plot")
    validate_data_shapes(data, model)
    ax_aspect = data.shape[1] / data.shape[0]
    figsize, ax_dims = compute_figure_layout(2, ax_aspect)
    vmin, vmax = compute_display_range(data, model)

    fig = pylab.figure(fignum, figsize=figsize)
    axes = [fig.add_axes(ax_dims[i]) for i in range(2)]

    dm_kw = {'interpolation': 'nearest', 'vmin': vmin, 'vmax': vmax, 'cmap': mpl.cm.jet}
    axes[0].imshow(data, **dm_kw)
    axes[1].imshow(model, **dm_kw)

    for ax in axes:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    axes[0].set_title('data')
    axes[1].set_title('model')

    pylab.draw()
    pylab.show()
    logger.debug("2-panel plot displayed successfully")


def plot_three_panel(data, model, fignum=1):
    '''
    Plot data, model, and residuals in a 3-panel figure.
    '''
    logger.info("Creating 3-panel plot")
    validate_data_shapes(data, model)
    resid = data - model
    ax_aspect = data.shape[1] / data.shape[0]
    figsize, ax_dims = compute_figure_layout(3, ax_aspect)

    dm_vmin, dm_vmax = compute_display_range(data, model)
    r_vmin, r_vmax = compute_display_range(resid)

    fig = pylab.figure(fignum, figsize=figsize)
    axes = [fig.add_axes(ax_dims[i]) for i in range(3)]

    dm_kw = {'interpolation': 'nearest', 'vmin': dm_vmin, 'vmax': dm_vmax, 'cmap': mpl.cm.jet}
    r_kw = {'interpolation': 'nearest', 'vmin': r_vmin, 'vmax': r_vmax, 'cmap': mpl.cm.RdBu}

    axes[0].imshow(data, **dm_kw)
    axes[1].imshow(model, **dm_kw)
    axes[2].imshow(resid, **r_kw)

    for ax in axes:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    axes[0].set_title('data')
    axes[1].set_title('model')
    axes[2].set_title('residual')

    pylab.draw()
    pylab.show()
    logger.debug("3-panel plot displayed successfully")


# ----------------------------- Main Section ----------------------------- #

if __name__ == '__main__':

    # ------------------ Logging Configuration ------------------ #
    log_filename = "plotter.log"

    # Create root logger
    logger.setLevel(logging.DEBUG)

    # Console handler (prints to terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Less verbose in console

    # File handler (rotating)
    file_handler = RotatingFileHandler(log_filename, maxBytes=1_000_000, backupCount=3)
    file_handler.setLevel(logging.DEBUG)  # Log everything to file

    # Common formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - line %(lineno)d - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Attach handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("Running module as a script — starting tests")
    

    # 1. Generate valid test data and run both plotting functions
    try:
        data, model = generate_test_data()
        plot_two_panel(data, model)
        plot_three_panel(data, model)
        logger.info("Successfully plotted valid data and model")
    except Exception as e:
        logger.exception(f"Unexpected error during valid plotting: {e}")

    # 2. Intentionally trigger a shape mismatch error to test exception handling
    try:
        bad_model = np.random.normal(size=(100, 100))  # Wrong shape
        plot_two_panel(data, bad_model)  # should raise ValueError
    except ValueError as ve:
        logger.warning(f"Expected ValueError caught: {ve}")
    except Exception as e:
        logger.exception(f"Unexpected exception type caught: {e}")

    # 3. Intentionally trigger vmax < vmin error by passing NaN array
    try:
        nan_array = np.full_like(data, np.nan)
        # compute_display_range will fail due to min/max on NaNs
        compute_display_range(nan_array)
    except ValueError as ve:
        logger.warning(f"Expected ValueError for display range: {ve}")
    except Exception as e:
        logger.exception(f"Unexpected exception during display range test: {e}")

    logger.info("All logging and exception tests completed")