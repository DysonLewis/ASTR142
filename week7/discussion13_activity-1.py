import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from astropy.modeling.functional_models import AiryDisk2D, Gaussian2D
from astropy.io import ascii

import os

def logparabola(x, phi0, a, b):
    E0 = 1.0  # normalization at 1 TeV
    return phi0 * (x / E0) ** (a - b * np.log10(x / E0))


def fit_crab_spectrum(filename):
    # Make it always look in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
	
    data = ascii.read(filepath, format='ecsv')
    E = data['e_ref'].data
    dnde = data['dnde'].data
    err = (data['dnde_errn'].data + data['dnde_errp'].data) / 2  # symmetric error

    # Residual function for least squares
    def residuals(params):
        phi0, a, b = params
        model = logparabola(E, phi0, a, b)
        return (dnde - model) / err

    # Initial guess
    p0 = [1e-10, -3, 0.1]

    # Fit using least squares
    result = least_squares(residuals, p0, jac='2-point')

    phi0, a, b = result.x
    print("Fit results:")
    print(f"phi0 = {phi0:.3e}")
    print(f"a    = {a:.3f}")
    print(f"b    = {b:.3f}")

    # Estimate parameter uncertainties from covariance matrix
    _, s, VT = np.linalg.svd(result.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(result.jac.shape) * s[0]
    s = s[s > threshold]
    cov = VT.T @ np.diag(1 / s**2) @ VT
    perr = np.sqrt(np.diag(cov))
    print("\nParameter errors (1σ):")
    print(f"σ_phi0 = {perr[0]:.3e}")
    print(f"σ_a    = {perr[1]:.3f}")
    print(f"σ_b    = {perr[2]:.3f}")

    # Plot data and fit
    E_fit = np.logspace(np.log10(min(E)), np.log10(max(E)), 200)
    model_fit = logparabola(E_fit, *result.x)

    plt.errorbar(E, dnde, yerr=err, fmt='o', label='Data', color='k')
    plt.plot(E_fit, model_fit, label='Best-fit logparabola', color='red')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Energy [TeV]')
    plt.ylabel('dN/dE [cm$^{-2}$ s$^{-1}$ TeV$^{-1}$]')
    plt.legend()
    plt.title('Crab Spectrum Fit (Log-Parabola)')
    plt.show()

    return result


def fit_psf(model_type='Airy'):
    xx, yy = np.meshgrid(np.arange(512), np.arange(512))
    true_model = AiryDisk2D(amplitude=100, x_0=246.3, y_0=230.8, radius=25)
    clean_data = true_model(xx, yy)
    noise = np.random.normal(scale=np.sqrt(clean_data.flatten()), size=512**2).reshape((512, 512))
    noisy_data = clean_data + noise

    plt.imshow(noisy_data, origin='lower', cmap='inferno')
    plt.title('Noisy PSF Data')
    plt.colorbar(label='Counts')
    plt.show()

    # Residual function to fit either Airy or Gaussian
    def residual_2d(params):
        if model_type == 'Airy':
            amp, x0, y0, radius = params
            model = AiryDisk2D(amplitude=amp, x_0=x0, y_0=y0, radius=radius)
        elif model_type == 'Gaussian':
            amp, x0, y0, sigma_x, sigma_y = params
            model = Gaussian2D(amplitude=amp, x_mean=x0, y_mean=y0, x_stddev=sigma_x, y_stddev=sigma_y)
        else:
            raise ValueError("model_type must be 'Airy' or 'Gaussian'")
        return (model(xx, yy) - noisy_data).ravel()

    # Initial guesses
    if model_type == 'Airy':
        p0 = [90, 250, 230, 20]
    else:
        p0 = [90, 250, 230, 10, 10]

    result = least_squares(residual_2d, p0)
    print(f"\n2D {model_type} Fit Results:")
    print(result.x)

    # Plot fitted model
    if model_type == 'Airy':
        fitted = AiryDisk2D(amplitude=result.x[0], x_0=result.x[1], y_0=result.x[2], radius=result.x[3])
    else:
        fitted = Gaussian2D(amplitude=result.x[0], x_mean=result.x[1], y_mean=result.x[2],
                            x_stddev=result.x[3], y_stddev=result.x[4])

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(noisy_data, origin='lower', cmap='inferno')
    plt.title('Data')
    plt.subplot(1, 3, 2)
    plt.imshow(fitted(xx, yy), origin='lower', cmap='inferno')
    plt.title(f'Fitted {model_type}')
    plt.subplot(1, 3, 3)
    plt.imshow(noisy_data - fitted(xx, yy), origin='lower', cmap='bwr')
    plt.title('Residuals')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 1. Fit the Crab spectrum
    result_spectrum = fit_crab_spectrum('crab_18-19_veritas.ecsv')

    # 2. Fit 2D model
    fit_psf('Airy')