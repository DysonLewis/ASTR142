#
# Color-Magnitude Diagram for M2 using HST data
# Data from Sarajedini et al. (2007)
#
#
# (c) 2023 Michael Fitzgerald (mpfitz@ucla.edu)
#
# Some code for querying Vizier for catalog to construct a CMD.
#

from astroquery.vizier import Vizier
import matplotlib.pyplot as plt
import numpy as np

# Catalog information
catalog_name = 'J/AJ/133/1658'  # Sarajedini et al. (2007)
cluster_name = 'NGC7089'  # M2's NGC designation

# Vizier.ROW_LIMIT = 100
# Vizier.ROW_LIMIT = 1000
Vizier.ROW_LIMIT = -1

print(f"Querying Vizier for catalog {catalog_name}...")

# see what tables are available
v = Vizier(row_limit=100)
catalogs = v.get_catalogs(catalog_name)

print(f"Found {len(catalogs)} table(s) in catalog")
for i, cat in enumerate(catalogs):
    print(f"\nTable {i}: {cat}")
    print(f"Columns: {cat.colnames}")
    print(f"Number of rows: {len(cat)}")

# Now query for M2 data
# Try querying with the cluster designation
print(f"\n{'='*60}")
print(f"Querying for cluster {cluster_name}...")

# Method 1: Query by object name (M2)
result = Vizier.query_object("M2", catalog=catalog_name)

if len(result) == 0:
    print("Method 1 (M2) failed, trying NGC 7089...")
    result = Vizier.query_object("NGC 7089", catalog=catalog_name)

if len(result) == 0:
    print("Object query failed. Trying to get all data and filter...")
    # Method 2: Get the photometry table directly
    result = Vizier(row_limit=100).get_catalogs(catalog_name)

# Check if we got results
if len(result) == 0:
    print("No results found!")
else:
    print(f"Retrieved {len(result)} table(s)")
    
    # Find the table with photometry data
    # Look for table with Vmag and Imag columns
    data = None
    for table in result:
        print(f"\nTable columns: {table.colnames}")
        if 'Vmag' in table.colnames or 'V' in table.colnames:
            data = table
            break
    
    if data is None:
        print("Could not find photometry table!")
        data = result[0]  # Use first table as fallback
    
    print(f"Using table with {len(data)} sources")
    print(f"Columns: {data.colnames}")
    
    # Use the actual magnitude columns
    # The catalog has Vmag, Imag, and a pre-computed V-I column
    v_col = 'Vmag'
    vi_col = 'V-I'
    
    if v_col in data.colnames and vi_col in data.colnames:
        print(f"Using columns: {v_col} and {vi_col}")
        
        # Extract V magnitude and V-I color
        vmag = data[v_col]
        vi = data[vi_col]
        
        # Remove any masked/invalid values
        if hasattr(vmag, 'mask') and hasattr(vi, 'mask'):
            mask = ~vmag.mask & ~vi.mask
            vi = vi[mask]
            vmag = vmag[mask]
        else:
            # No masked values, use all data
            pass
        
        print(f"\nValid sources after masking: {len(vmag)}")
        print(f"V magnitude range: {vmag.min():.2f} to {vmag.max():.2f}")
        print(f"V-I color range: {vi.min():.2f} to {vi.max():.2f}")
        
        # Create the color-magnitude diagram
        fig = plt.figure(figsize=(8, 10))
        fig.clear()
        ax = fig.add_subplot(111)
        
        ax.scatter(vi, vmag,
                   marker='.',
                   c='k',
                   s=1.,  # experiment with marker size
                   alpha=0.5)
        
        # Add plot title and axes labels
        ax.set_xlabel('V - I (mag)', fontsize=12)
        ax.set_ylabel('V (mag)', fontsize=12)
        ax.set_title(f'Color-Magnitude Diagram: M2 (NGC 7089)\nSarajedini et al. (2007)', 
                     fontsize=14)
        
        # Invert the y-axis (fainter stars at bottom)
        ax.invert_yaxis()
        
        # Add grid for easier reading
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.draw()
        plt.show()
        
        # Save the figure
        fig.savefig('hw4prob3.pdf', dpi=200, bbox_inches='tight')
        print("\nPlot saved as 'hw4prob3.pdf'")
    else:
        print(f"Could not find Vmag and V-I columns!")
        print(f"Available columns: {data.colnames}")