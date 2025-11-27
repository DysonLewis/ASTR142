import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

### If you want to define functions, those should be here (not in main)

if __name__ == '__main__':
    ### Load the array you made in homework 1
    data = np.load('hw1prob2.npy')
    
    ### Make a figure and add a main panel. 
    ### Leave some space for adding two colorbars later
    fig = plt.figure(figsize=(12, 8))
    # Main panel with space for colorbars on the right
    ax_main = fig.add_axes([0.1, 0.1, 0.65, 0.8])
    
    ### Plot with imshow (consider playing with the colormap/scaling) in the main panel
    im = ax_main.imshow(data, cmap='viridis', origin='lower', aspect='auto')
    
    ### Overlay the image with a contour plot. Experiment with 
    ### color maps/levels so that the contours are clear. Add labels to your 
    ### contours
    # Create contour levels based on data range
    levels = np.linspace(np.nanmin(data), np.nanmax(data), 8)
    contours = ax_main.contour(data, levels=levels, cmap='autumn', 
                                linewidths=1.5, alpha=0.8)
    ax_main.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
    
    ### Create two new axes. In each one, add a color bar, one for 
    ### the contour plot and one for the image. Make sure the scales
    ### match and the axis labels are big enough to read
    # Colorbar for imshow
    cax1 = fig.add_axes([0.78, 0.55, 0.03, 0.35])
    cbar1 = plt.colorbar(im, cax=cax1)
    cbar1.set_label('Image Intensity', fontsize=12)
    cbar1.ax.tick_params(labelsize=10)
    
    # Colorbar for contours
    cax2 = fig.add_axes([0.78, 0.1, 0.03, 0.35])
    cbar2 = plt.colorbar(contours, cax=cax2)
    cbar2.set_label('Contour Levels', fontsize=12)
    cbar2.ax.tick_params(labelsize=10)
    
    ### Add a semi-transparent circular patch inscribing the central region. Name
    ### the patch something and add a legend in the upper-left corner with
    ### no bounding box
    # Calculate center and radius based on data shape
    center_y, center_x = np.array(data.shape) / 2
    radius = min(data.shape) / 3
    
    central_region = Circle((center_x, center_y), radius, 
                           fill=False, edgecolor='cyan', 
                           linewidth=2, alpha=0.6, 
                           label='Central Region')
    ax_main.add_patch(central_region)
    
    ### ZOOM INSET ###
    # Add an inset zoom of the central region
    axins = inset_axes(ax_main, width="35%", height="35%", 
                       loc='lower right', borderpad=1.5)
    
    # Calculate zoom region
    zoom_size = int(radius * 0.4)
    y_min, y_max = int(center_y - zoom_size), int(center_y + zoom_size)
    x_min, x_max = int(center_x - zoom_size) - 200, int(center_x + zoom_size) - 200
    
    # Plot zoomed region
    axins.imshow(data[y_min:y_max, x_min:x_max], 
                cmap='viridis', origin='lower', aspect='auto')
    axins.set_title('Zoom: Central Region', fontsize=9)
    axins.tick_params(labelsize=7)
    
    # Add rectangle on main plot showing zoom location
    zoom_box = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                         fill=False, edgecolor='red', linewidth=2, 
                         linestyle='--', alpha=0.7, label='Zoom Region')
    ax_main.add_patch(zoom_box)
    
    # Update legend to include zoom box
    ax_main.legend(loc='upper left', frameon=False, fontsize=11)
    
    # Add axis labels
    ax_main.set_xlabel('X Pixel', fontsize=12)
    ax_main.set_ylabel('Y Pixel', fontsize=12)
    ax_main.set_title('Data Visualization with Contours', fontsize=14)
    
    ### Save your plot as a pdf (however far you get) and upload it for the discussion assignment
    plt.savefig('homework_plot.png', bbox_inches='tight', dpi=300)
    print("Plot saved as homework_plot.pdf")
    plt.show()