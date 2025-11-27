import numpy as np
from astropy.io import fits
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os

def moffat_psf(x, y, x0, y0, flux, alpha, beta, background):
    """
    Calculate the Moffat PSF model.
    
    Parameters:
    x, y: 2D coordinate arrays
    x0, y0: center position of the star
    flux: total flux of the star
    alpha: scale parameter
    beta: power law index
    background: constant background level
    
    Returns:
    2D array of model values
    """
    # Convert to polar coordinates centered on the star
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    
    # Moffat profile (normalized)
    normalization = (beta - 1) / (np.pi * alpha**2)
    profile = normalization * (1 + (r / alpha)**2)**(-beta)
    
    # Model = flux * profile + background
    model = flux * profile + background
    
    return model

def residual_function(params, x, y, data, uncertainty):
    """
    Calculate the residual vector for least squares fitting.
    
    Parameters:
    params: [x0, y0, flux, alpha, beta, background]
    x, y: 2D coordinate arrays
    data: 2D data array
    uncertainty: 2D uncertainty array
    
    Returns:
    1D flattened residual vector
    """
    x0, y0, flux, alpha, beta, background = params
    
    # Calculate model
    model = moffat_psf(x, y, x0, y0, flux, alpha, beta, background)
    
    # Calculate normalized residuals
    residuals_2d = (data - model) / uncertainty
    
    # Flatten to 1D
    residuals_1d = residuals_2d.flatten()
    
    return residuals_1d

def perform_fit(data, uncertainty, initial_params, bounds=None):
    """
    Perform least-squares fitting.
    
    Parameters:
    data: 2D data array
    uncertainty: 2D uncertainty array
    initial_params: initial parameter guess [x0, y0, flux, alpha, beta, background]
    bounds: optional parameter bounds
    
    Returns:
    result: optimization result object
    """
    # Create coordinate arrays
    ny, nx = data.shape
    x = np.arange(nx)
    y = np.arange(ny)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Perform least-squares fit
    result = least_squares(
        residual_function,
        initial_params,
        args=(x_grid, y_grid, data, uncertainty),
        bounds=bounds,
        verbose=1
    )
    
    return result, x_grid, y_grid

def calculate_uncertainties(result):
    """
    Calculate parameter uncertainties from the covariance matrix.
    
    Parameters:
    result: optimization result from least_squares
    
    Returns:
    uncertainties: 1-sigma uncertainties for each parameter
    """
    # Jacobian at the solution
    J = result.jac
    
    # Covariance matrix (approximation)
    # Cov = (J^T J)^-1
    try:
        cov = np.linalg.inv(J.T @ J)
        uncertainties = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        print("Warning: Could not compute covariance matrix")
        uncertainties = np.full(len(result.x), np.nan)
    
    return uncertainties

def plot_results(data, model, residuals, params, uncertainties, output_path):
    """
    Create a 3-panel figure showing data, model, and residuals.
    
    Parameters:
    output_path: full path where the figure should be saved
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Data
    im0 = axes[0].imshow(data, origin='lower', cmap='viridis')
    axes[0].set_title('Data')
    axes[0].set_xlabel('X (pixels)')
    axes[0].set_ylabel('Y (pixels)')
    plt.colorbar(im0, ax=axes[0])
    
    # Mark the fitted position
    axes[0].plot(params[0], params[1], 'r+', markersize=15, markeredgewidth=2)
    
    # Model
    im1 = axes[1].imshow(model, origin='lower', cmap='viridis')
    axes[1].set_title('Model')
    axes[1].set_xlabel('X (pixels)')
    axes[1].set_ylabel('Y (pixels)')
    plt.colorbar(im1, ax=axes[1])
    
    # Residuals
    im2 = axes[2].imshow(residuals, origin='lower', cmap='RdBu_r', 
                         vmin=-3, vmax=3)
    axes[2].set_title('Normalized Residuals')
    axes[2].set_xlabel('X (pixels)')
    axes[2].set_ylabel('Y (pixels)')
    plt.colorbar(im2, ax=axes[2], label='sigma')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nFigure saved as '{output_path}' (300 dpi)")

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fits_file = os.path.join(script_dir, 'simdata.fits')
    
    print(f"Script directory: {script_dir}")
    print(f"Looking for FITS file at: {fits_file}")
    
    if not os.path.exists(fits_file):
        print(f"Error: {fits_file} not found!")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in script directory: {os.listdir(script_dir)}")
        return
    
    with fits.open(fits_file) as hdul:
        print(f"\nFITS file structure:")
        hdul.info()
        print(f"\nNumber of extensions: {len(hdul)}")
        
        # The problem states data is in first extension, uncertainty in second
        # FITS indexing: 0 = primary HDU, 1 = first extension, 2 = second extension
        if len(hdul) < 3:
            print(f"\nError: Expected at least 3 HDUs (primary + 2 extensions), found {len(hdul)}")
            print("Checking if extensions are at indices 0 and 1 instead...")
            if len(hdul) >= 2:
                data = hdul[0].data.astype(float) if hdul[0].data is not None else hdul[1].data.astype(float)
                uncertainty = hdul[1].data.astype(float)
            else:
                print("Cannot proceed - insufficient extensions in FITS file")
                return
        else:
            data = hdul[1].data.astype(float)
            uncertainty = hdul[2].data.astype(float)
    
    print(f"\nData shape: {data.shape}")
    print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Uncertainty range: [{uncertainty.min():.2f}, {uncertainty.max():.2f}]")
    
    # Estimate initial parameters
    # Find approximate position of maximum
    ny, nx = data.shape
    y_max, x_max = np.unravel_index(np.argmax(data), data.shape)
    
    # Estimate background from corners
    corners = np.concatenate([
        data[:5, :5].flatten(),
        data[:5, -5:].flatten(),
        data[-5:, :5].flatten(),
        data[-5:, -5:].flatten()
    ])
    background_est = np.median(corners)
    
    # Estimate flux
    flux_est = np.sum(data - background_est)
    
    # Initial parameter guess: [x0, y0, flux, alpha, beta, background]
    initial_params = [x_max, y_max, flux_est, 3.0, 4.5, background_est]
    
    print("\n" + "="*60)
    print("INITIAL PARAMETER ESTIMATES:")
    print("="*60)
    print(f"Position (x0, y0): ({initial_params[0]:.2f}, {initial_params[1]:.2f})")
    print(f"Flux: {initial_params[2]:.2f}")
    print(f"Alpha: {initial_params[3]:.2f}")
    print(f"Beta: {initial_params[4]:.2f}")
    print(f"Background: {initial_params[5]:.2f}")
    
    # Set reasonable bounds for parameters
    bounds = (
        [0, 0, 0, 0.1, 1.1, -np.inf],  # lower bounds
        [nx, ny, np.inf, 20, 20, np.inf]  # upper bounds
    )
    
    # Perform fit
    print("\n" + "="*60)
    print("PERFORMING LEAST-SQUARES FIT...")
    print("="*60)
    result, x_grid, y_grid = perform_fit(data, uncertainty, initial_params, bounds)
    
    # Check convergence
    print("\n" + "="*60)
    print("FIT CONVERGENCE STATUS:")
    print("="*60)
    print(f"Success: {result.success}")
    print(f"Status: {result.status}")
    print(f"Message: {result.message}")
    print(f"Cost (final): {result.cost:.6e}")
    print(f"Number of iterations: {result.nfev}")
    
    # Calculate uncertainties
    uncertainties = calculate_uncertainties(result)
    
    # Report results
    param_names = ['x0', 'y0', 'flux', 'alpha', 'beta', 'background']
    print("\n" + "="*60)
    print("BEST-FIT PARAMETERS (with 1-sigma uncertainties):")
    print("="*60)
    for name, value, unc in zip(param_names, result.x, uncertainties):
        print(f"{name:12s}: {value:12.6f} ± {unc:.6f}")
    
    # Test sensitivity to initial conditions
    print("\n" + "="*60)
    print("TESTING SENSITIVITY TO INITIAL CONDITIONS:")
    print("="*60)
    
    test_cases = [
        [x_max + 2, y_max - 2, flux_est * 1.2, 2.5, 5.0, background_est * 0.9],
        [x_max - 1, y_max + 1, flux_est * 0.8, 3.5, 4.0, background_est * 1.1],
        [x_max + 1, y_max - 1, flux_est * 1.1, 2.0, 6.0, background_est]
    ]
    
    all_results = [result.x]
    
    for i, test_init in enumerate(test_cases, 1):
        print(f"\nTest case {i}:")
        test_result, _, _ = perform_fit(data, uncertainty, test_init, bounds)
        all_results.append(test_result.x)
        
        if test_result.success:
            diff = np.abs(test_result.x - result.x)
            rel_diff = diff / (np.abs(result.x) + 1e-10) * 100
            print(f"  Success: {test_result.success}")
            print(f"  Max relative difference: {np.max(rel_diff):.4f}%")
        else:
            print(f"  Success: {test_result.success} - {test_result.message}")
    
    # Calculate statistics across different initial conditions
    all_results = np.array(all_results)
    std_results = np.std(all_results, axis=0)
    
    print("\n" + "="*60)
    print("FIT STABILITY ASSESSMENT:")
    print("="*60)
    print("Standard deviation across different initial conditions:")
    for name, std_val, unc in zip(param_names, std_results, uncertainties):
        print(f"{name:12s}: {std_val:.6e} (uncertainty: {unc:.6e})")
    
    max_variation = np.max(std_results / (uncertainties + 1e-10))
    print(f"\nMaximum variation / uncertainty ratio: {max_variation:.2f}")
    
    if max_variation < 0.1:
        print("FIT IS HIGHLY STABLE - variations are much smaller than uncertainties")
    elif max_variation < 1.0:
        print("FIT IS STABLE - variations are comparable to or smaller than uncertainties")
    else:
        print("WARNING: FIT MAY BE UNSTABLE - variations exceed uncertainties")
    
    # Generate model and residuals for plotting
    best_params = result.x
    model = moffat_psf(x_grid, y_grid, *best_params)
    residuals_2d = (data - model) / uncertainty
    
    # Create plots - save to script directory
    print("\n" + "="*60)
    print("GENERATING PLOTS...")
    print("="*60)
    output_path = os.path.join(script_dir, 'fit_results.png')
    plot_results(data, model, residuals_2d, best_params, uncertainties, output_path)
    
    # Additional statistics
    print("\n" + "="*60)
    print("RESIDUAL STATISTICS:")
    print("="*60)
    print(f"Mean normalized residual: {np.mean(residuals_2d):.4f}")
    print(f"Std of normalized residuals: {np.std(residuals_2d):.4f}")
    print(f"Chi-squared (reduced): {result.cost / (data.size - len(result.x)):.4f}")

if __name__ == "__main__":
    main()