import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
mat_file = os.path.join(script_dir, 'demo_data.mat')

# Load the .mat file
print(f"Loading data from {mat_file}...")
data = loadmat(mat_file)

# Print available variables in the .mat file
print("\nAvailable variables in .mat file:")
for key in data.keys():
    if not key.startswith('__'):
        print(f"  {key}: shape {data[key].shape}")

# Extract the faces data
X = data['faces']
print(f"\nUsing data matrix: faces")
print(f"Data shape: {X.shape}")
print(f"Number of face images: {X.shape[0]}")
print(f"Pixels per face: {X.shape[1]} (likely 48x75 or similar)")

# Center the data
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

# Compute covariance matrix and eigenvalues/eigenvectors
cov_matrix = np.cov(X_centered.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Create visualizations
fig = plt.figure(figsize=(20, 14))

# 1. Eigenvalue spectrum
ax1 = plt.subplot(3, 4, 1)
plt.plot(eigenvalues, 'bo-', linewidth=2, markersize=6)
plt.xlabel('Component Number', fontsize=12)
plt.ylabel('Eigenvalue', fontsize=12)
plt.title('Eigenvalue Spectrum (Scree Plot)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')

# 2. Cumulative explained variance
ax2 = plt.subplot(3, 4, 2)
explained_var = eigenvalues / np.sum(eigenvalues) * 100
cumulative_var = np.cumsum(explained_var)
plt.plot(cumulative_var, 'ro-', linewidth=2, markersize=6)
plt.xlabel('Number of Components', fontsize=12)
plt.ylabel('Cumulative Variance Explained (%)', fontsize=12)
plt.title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axhline(y=90, color='g', linestyle='--', label='90% threshold', alpha=0.7)
plt.axhline(y=95, color='orange', linestyle='--', label='95% threshold', alpha=0.7)
plt.legend()

# 3. Eigenface visualization (mean face)
ax3 = plt.subplot(3, 4, 3)
# Assume 48 pixels might be 6x8 or 8x6
img_h, img_w = 6, 8
mean_face = X_mean.reshape(img_h, img_w)
plt.imshow(mean_face, cmap='gray', interpolation='nearest')
plt.title('Mean Face', fontsize=14, fontweight='bold')
plt.axis('off')

# 4. First eigenvector as eigenface
ax4 = plt.subplot(3, 4, 4)
eigenface_1 = eigenvectors[:, 0].reshape(img_h, img_w)
plt.imshow(eigenface_1, cmap='gray', interpolation='nearest')
plt.title('1st Eigenface', fontsize=14, fontweight='bold')
plt.axis('off')

# Show more eigenfaces
for i in range(4):
    ax = plt.subplot(3, 4, 5 + i)
    eigenface = eigenvectors[:, i+1].reshape(img_h, img_w)
    plt.imshow(eigenface, cmap='gray', interpolation='nearest')
    plt.title(f'Eigenface {i+2}', fontsize=12, fontweight='bold')
    plt.axis('off')

# Reconstruction with different numbers of components
n_components_list = [1, 2, 5, 10]
sample_idx = 0  # First face

for idx, n_comp in enumerate(n_components_list):
    # Project data onto first n components
    eigenvectors_subset = eigenvectors[:, :n_comp]
    X_projected = X_centered @ eigenvectors_subset
    
    # Reconstruct data
    X_reconstructed = X_projected @ eigenvectors_subset.T + X_mean
    
    # Calculate reconstruction error
    reconstruction_error = np.mean((X[sample_idx] - X_reconstructed[sample_idx]) ** 2)
    var_explained = cumulative_var[n_comp-1]
    
    # Plot reconstructed face
    ax = plt.subplot(3, 4, 9 + idx)
    reconstructed_face = X_reconstructed[sample_idx].reshape(img_h, img_w)
    plt.imshow(reconstructed_face, cmap='gray', interpolation='nearest')
    plt.title(f'{n_comp} Components\nMSE: {reconstruction_error:.4f}\nVar: {var_explained:.1f}%', 
              fontsize=10, fontweight='bold')
    plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'eigenvalue_analysis.png'), dpi=600, bbox_inches='tight')
print(f"\nPlot saved to: {os.path.join(script_dir, 'eigenvalue_analysis.png')}")
plt.show()

# Create a separate figure showing the original face and reconstructions side by side
fig2 = plt.figure(figsize=(18, 6))

# Original face
ax_orig = plt.subplot(1, 6, 1)
original_face = X[sample_idx].reshape(img_h, img_w)
plt.imshow(original_face, cmap='gray', interpolation='nearest')
plt.title('Original Face', fontsize=14, fontweight='bold')
plt.axis('off')

# Reconstructions with different numbers of components
n_components_full = [1, 2, 5, 10, 20]

for idx, n_comp in enumerate(n_components_full):
    if n_comp > len(eigenvalues):
        n_comp = len(eigenvalues)
    
    # Project data onto first n components
    eigenvectors_subset = eigenvectors[:, :n_comp]
    X_projected = X_centered @ eigenvectors_subset
    
    # Reconstruct data
    X_reconstructed = X_projected @ eigenvectors_subset.T + X_mean
    
    # Calculate reconstruction error
    reconstruction_error = np.mean((X[sample_idx] - X_reconstructed[sample_idx]) ** 2)
    var_explained = cumulative_var[n_comp-1]
    
    # Plot reconstructed face
    ax = plt.subplot(1, 6, idx + 2)
    reconstructed_face = X_reconstructed[sample_idx].reshape(img_h, img_w)
    plt.imshow(reconstructed_face, cmap='gray', interpolation='nearest')
    plt.title(f'{n_comp} Components\nMSE: {reconstruction_error:.4f}\nVar: {var_explained:.1f}%', 
              fontsize=12, fontweight='bold')
    plt.axis('off')

plt.suptitle(f'Face Reconstruction Comparison (Face #{sample_idx})', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'face_reconstruction_comparison.png'), dpi=600, bbox_inches='tight')
print(f"Reconstruction comparison saved to: {os.path.join(script_dir, 'face_reconstruction_comparison.png')}")
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total components: {len(eigenvalues)}")
print(f"\nTop 5 eigenvalues: {eigenvalues[:5]}")
print(f"\nVariance explained by top 5 components:")
for i in range(min(5, len(explained_var))):
    print(f"  PC{i+1}: {explained_var[i]:.2f}%")
print(f"\nComponents needed for 90% variance: {np.argmax(cumulative_var >= 90) + 1}")
print(f"Components needed for 95% variance: {np.argmax(cumulative_var >= 95) + 1}")
print("="*60)