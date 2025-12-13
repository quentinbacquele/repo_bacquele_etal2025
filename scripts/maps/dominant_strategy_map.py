#!/usr/bin/env python3

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import argparse
import ast  # Needed for parsing if used, but not in this version
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- Define Default 8-Color Palette ---
DEFAULT_8_COLORS = ['#785EF0', '#E69F00', '#009E73', '#F0E442',
                    '#0072B2', '#D55E00', '#CC79A7', '#444444']

# Set publication-quality defaults (minimal) - From original script
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 0.5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.0,  # No padding
})

def calculate_dominant_strategy_by_percentile(probs_df, num_clusters):
    """
    Calculates the dominant strategy for each cell based on the highest 
    percentile rank. (Identical calculation logic as before)
    """
    prob_cols = [f'prob_{i}' for i in range(num_clusters)]
    rank_cols = [f'pct_rank_{i}' for i in range(num_clusters)]

    missing_cols = [col for col in prob_cols if col not in probs_df.columns]
    if missing_cols:
        raise ValueError(f"Missing probability columns in DataFrame: {missing_cols}")

    work_df = probs_df[['grid_id'] + prob_cols].copy()

    for i in range(num_clusters):
        prob_col = prob_cols[i]
        rank_col = rank_cols[i]
        work_df[rank_col] = work_df[prob_col].rank(method='average', pct=True, na_option='keep') 

    rank_data = work_df[rank_cols].fillna(-1) 
    max_rank_col_name = rank_data.idxmax(axis=1)
    
    def get_index_from_colname(colname):
        if pd.isna(colname) or colname == -1: return np.nan
        try: return int(colname.split('_')[-1])
        except (AttributeError, ValueError): return np.nan

    work_df['dominant_strategy_pct'] = max_rank_col_name.apply(get_index_from_colname)
    return work_df[['grid_id', 'dominant_strategy_pct']]


def create_dominant_strategy_map_styled( # Renamed for clarity
    gdf, # GeoDataFrame merged with dominant_strategy_pct AND geometry
    output_path,
    num_clusters, 
    cmap_name="tab10", 
    strategy_colors=None, 
    figure_width=10.0,
    figure_height=5.0, # Matched original script default
    bg_color="#FFFFFF",
    land_color="#FFFFFF", # Matched original script default
    ocean_color="#E6ECF5", # Matched original script default
    legend_size=0.03, # Matched original script default
    alpha=1.0, # Added alpha from original script
    strategy_labels=None,
    legend_title="Dominant Strategy\n(Highest Percentile Rank)" # Kept specific title
):
    """
    Create a minimalist map showing the dominant strategy based on percentile rank,
    styled identically to the provided species richness script.
    Uses pcolormesh raster approach.
    """
    
    # --- Prepare Colors and Normalization (Identical logic as before) ---
    colors = None
    if strategy_colors:
        if len(strategy_colors) == num_clusters:
            print("Using provided custom strategy colors via --strategy-colors.")
            colors = strategy_colors
        else:
            print(f"Warning: Provided --strategy-colors invalid length. Ignoring custom colors.")
            strategy_colors = None 
    if colors is None and num_clusters == 8:
        print(f"Using default 8-color palette for {num_clusters} clusters.")
        colors = DEFAULT_8_COLORS
    if colors is None:
        print(f"Using fallback colormap name: '{cmap_name}' for {num_clusters} clusters.")
        try:
            if num_clusters <= 10 and cmap_name in ["tab10"]:
                 cmap_func = plt.get_cmap(cmap_name); colors = [cmap_func(i/10.0) for i in range(num_clusters)]
            elif num_clusters <= 20 and cmap_name in ["tab20", "tab20b", "tab20c"]:
                 cmap_func = plt.get_cmap(cmap_name); colors = [cmap_func(i) for i in range(num_clusters)]
            else: 
                 cmap_func = plt.get_cmap(cmap_name); colors = [cmap_func(i / (num_clusters - 1 if num_clusters > 1 else 1) ) for i in range(num_clusters)]
                 print(f"Warning: Sampling potentially non-qualitative colormap '{cmap_name}'.")
        except ValueError:
            print(f"Warning: Colormap '{cmap_name}' not found. Falling back to sampling 'viridis'.")
            cmap_func = plt.get_cmap('viridis'); colors = [cmap_func(i / (num_clusters - 1 if num_clusters > 1 else 1)) for i in range(num_clusters)]
    if colors is None or len(colors) != num_clusters:
         print("Error: Could not determine appropriate colors. Aborting map creation.")
         return
    cmap = ListedColormap(colors)
    boundaries = np.arange(-0.5, num_clusters, 1)
    norm = BoundaryNorm(boundaries, cmap.N)
    # Set color for NaN data in the colormap if needed (pcolormesh often handles this)
    # cmap.set_bad(color=land_color) # Optional: Explicitly set NaN color

    # --- Create Figure and GeoAxes (Styled like original) ---
    fig = plt.figure(figsize=(figure_width, figure_height))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson()) # Use Robinson projection
    
    # Set background colors (Styled like original)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color) # Set axes background
    
    # Make the map minimalist (Styled like original)
    ax.set_global()
    ax.axis('off')  # Turn off all axis elements, including spines

    # Add land and ocean features (Styled like original)
    land = cfeature.LAND.with_scale('110m')
    ocean = cfeature.OCEAN.with_scale('110m')
    ax.add_feature(land, edgecolor='none', facecolor=land_color, zorder=0)
    ax.add_feature(ocean, edgecolor='none', facecolor=ocean_color, zorder=0)
    # NOTE: Original script didn't add coastlines explicitly AFTER adding land/ocean
    # To add coastlines ON TOP of the data/land, add them later with higher zorder
    # ax.coastlines(linewidth=0.5, color="#666666", zorder=3) # Example if needed

    # --- Plotting the Categorical Data using Raster Approach (like original) ---
    print("Creating raster grid for seamless visualization without grid lines...")
    
    # Ensure data is in PlateCarree for grid mapping
    gdf_pc = gdf.to_crs(ccrs.PlateCarree().proj4_init)

    # Create a regular 1-degree global grid (like original)
    x_grid = np.arange(-180, 181, 1) # Longitude edges
    y_grid = np.arange(-90, 91, 1)  # Latitude edges
    
    # Create an empty grid for the dominant strategy data, fill with NaN
    strategy_grid = np.full((len(y_grid)-1, len(x_grid)-1), np.nan) 
    
    # Get coordinates (use centroids - assuming 1-degree cells centered near integer coords)
    # Important: Ensure centroids accurately represent the grid cell for index calculation
    gdf_pc['grid_lon'] = gdf_pc.geometry.centroid.x
    gdf_pc['grid_lat'] = gdf_pc.geometry.centroid.y

    # Fill the grid with dominant strategy index values
    # Iterate only through rows with valid strategy data
    data_to_plot = gdf_pc.dropna(subset=['dominant_strategy_pct']).copy()
    
    for _, row in data_to_plot.iterrows():
        # Calculate grid indices based on centroid coordinates
        # Adding 0.5 assumes centroids are near cell centers, flooring maps to bottom-left corner index
        # Adjust if grid coordinates represent something else (e.g., corners)
        i = int(np.floor(row['grid_lon'] + 180))  # Lon index (0-359)
        j = int(np.floor(row['grid_lat'] + 90))   # Lat index (0-179)
        
        # Only set values for valid indices within the grid bounds
        if 0 <= i < strategy_grid.shape[1] and 0 <= j < strategy_grid.shape[0]:
            strategy_grid[j, i] = row['dominant_strategy_pct'] # Assign the strategy index
            
    # Plot the grid using pcolormesh (Styled like original)
    mesh = ax.pcolormesh(
        x_grid, y_grid, strategy_grid, 
        cmap=cmap, 
        norm=norm,
        transform=ccrs.PlateCarree(), # Data is on PlateCarree grid
        alpha=alpha, # Use alpha from args
        shading='flat',  # Match original style
        rasterized=True  # Match original style
        # zorder=2 # Ensure data plots above land/ocean but below coastlines if added later
    )
    
    # --- Add Minimalist Colorbar (Styled like original) ---
    # Position: [left, bottom, width, height]
    legend_height = legend_size # Use argument
    legend_width = 0.5  # Half the figure width (like original)
    legend_left = 0.25  # Centered (like original)
    legend_bottom = 0.05  # Near bottom (like original)
    
    cax = fig.add_axes([legend_left, legend_bottom, legend_width, legend_height])
    
    # Create ScalarMappable needed for colorbar with categorical data
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) # Dummy array needed

    # Create a very minimal colorbar (Styled like original)
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=np.arange(num_clusters)) # Set ticks explicitly
    cbar.ax.tick_params(size=0, labelsize=8, color='#555555') # Style ticks
    cbar.outline.set_visible(False) # Hide outline

    # Set tick labels for categories
    if strategy_labels and len(strategy_labels) == num_clusters:
        cbar.set_ticklabels(strategy_labels)
    else:
        cbar.set_ticklabels(np.arange(num_clusters)) # Default to indices
        if strategy_labels: print("Warning: strategy_labels provided but length doesn't match num_clusters. Using default indices.")
    
    # Add legend title if specified (Styled like original)
    if legend_title:
        cbar.set_label(legend_title, size=9, color='#333333')
    
    # Save figure (Identical settings)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"Saved styled dominant strategy map to {output_path}")
    plt.close(fig)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calculate dominant strategy by percentile and generate minimalist map styled like species richness map.")
    # Arguments kept from dominant_strategy script
    parser.add_argument("--feature-type", type=str, choices=["pc", "umap3d"], default="pc", help="Feature type used in GMM")
    parser.add_argument("--n-components", type=int, default=8, help="Number of GMM components/strategies")
    parser.add_argument("--extreme", action="store_true", help="Use extreme GMM assignments data")
    parser.add_argument("--grid-file", type=str, default="grid_cells.gpkg", help="Path to the grid cells geospatial file")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for output files")
    parser.add_argument("--output-name", type=str, default="dominant_strategy_percentile_map", help="Base name for output files")
    
    # Style arguments matched to species richness script
    parser.add_argument("--cmap", type=str, default="tab10", 
                       help="Matplotlib colormap name IF n_components is NOT 8 AND --strategy-colors is not provided")
    parser.add_argument("--strategy-colors", nargs='+', default=None,
                        help="Optional: List of custom hex color codes. Overrides default 8-color palette and --cmap.")
    parser.add_argument("--figure-width", type=float, default=10.0, help="Figure width in inches")
    parser.add_argument("--figure-height", type=float, default=5.0, help="Figure height in inches") # Matched default
    parser.add_argument("--bg-color", type=str, default="#FFFFFF", help="Background color") # Matched default
    parser.add_argument("--land-color", type=str, default="#FFFFFF", help="Land color") # Matched default
    parser.add_argument("--ocean-color", type=str, default="#E6ECF5", help="Ocean color") # Matched default
    parser.add_argument("--legend-size", type=float, default=0.03, help="Relative height of the color legend") # Added from original
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha transparency for data overlay") # Added from original
    parser.add_argument("--legend-title", type=str, default="Dominant Strategy\n(Highest Percentile Rank)", help="Title for the colorbar legend") # Kept specific title
    parser.add_argument("--strategy-labels", nargs='+', default=None, help="Optional: List of labels for strategies")
    # Removed args not relevant to categorical map: --vmin, --vmax, --log-scale, --min-richness

    args = parser.parse_args()

    # --- Validation (Identical logic) ---
    if args.strategy_labels and len(args.strategy_labels) != args.n_components:
        print(f"Error: Number of strategy labels mismatch."); return
    if args.strategy_colors and len(args.strategy_colors) != args.n_components:
         print(f"Error: Number of strategy colors mismatch."); return 

    # --- Setup paths (Identical logic) ---
    feature_label = f"{args.feature_type}_{args.n_components}"
    if args.extreme:
        input_probs_file = os.path.join(args.output_dir, f"grid_gmm_probabilities_{feature_label}_extreme.csv")
        output_prefix = f"{args.output_name}_{feature_label}_extreme"
    else:
        input_probs_file = os.path.join(args.output_dir, f"grid_gmm_probabilities_{feature_label}.csv")
        output_prefix = f"{args.output_name}_{feature_label}"
    output_csv = os.path.join(args.output_dir, f"{output_prefix}_dominant_strategy.csv") 
    output_png = os.path.join(args.output_dir, f"{output_prefix}.png")
    output_pdf = os.path.join(args.output_dir, f"{output_prefix}.pdf") 
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Load GMM probabilities (Identical logic) ---
    print(f"Loading GMM probabilities from {input_probs_file}...")
    try: probs_df = pd.read_csv(input_probs_file)
    except FileNotFoundError: print(f"Error: Could not find {input_probs_file}"); return
    if 'status' in probs_df.columns: probs_df = probs_df[probs_df['status'] == 'success'].copy()
    else: print("Warning: 'status' column not found. Using all rows.")
    if 'grid_id' not in probs_df.columns: print("Error: 'grid_id' column not found."); return
    prob_cols = [f'prob_{i}' for i in range(args.n_components)]
    if not all(col in probs_df.columns for col in prob_cols): print(f"Error: Missing probability columns."); return
    if len(probs_df) == 0: print("Error: No valid grid cells found."); return
    print(f"Loaded {len(probs_df)} grid cells with probability data.")
    
    # --- Calculate Dominant Strategy (Identical logic) ---
    print("Calculating dominant strategy based on highest percentile rank...")
    try: dominant_strategy_df = calculate_dominant_strategy_by_percentile(probs_df, args.n_components)
    except ValueError as e: print(f"Error during calculation: {e}"); return

    # --- Print Statistics (Identical logic) ---
    print(f"\nDominant Strategy Statistics:")
    dominant_counts = dominant_strategy_df['dominant_strategy_pct'].value_counts().sort_index()
    print("Number of cells where each strategy is dominant:")
    for i in range(args.n_components):
        count = dominant_counts.get(i, 0)
        label = args.strategy_labels[i] if args.strategy_labels else f"Strategy {i}"
        print(f"  {label}: {count} cells")
    nan_count = dominant_strategy_df['dominant_strategy_pct'].isna().sum()
    if nan_count > 0: print(f"  Undefined/NaN: {nan_count} cells")

    # --- Save dominant strategy data (Identical logic) ---
    dominant_strategy_df.to_csv(output_csv, index=False)
    print(f"Saved dominant strategy data to {output_csv}")
    
    # --- Load grid cells (Identical logic) ---
    print(f"Loading grid cells from {args.grid_file}...")
    try: grid_gdf = gpd.read_file(args.grid_file)
    except Exception as e: print(f"Error loading grid file {args.grid_file}: {e}"); return
    if 'grid_id' not in grid_gdf.columns: print(f"Error: 'grid_id' column missing."); return

    # --- Merge Data (Identical logic) ---
    grid_dominant_strategy = grid_gdf.merge(dominant_strategy_df[['grid_id', 'dominant_strategy_pct']], 
                                            on='grid_id', how='left')
    
    # --- Create minimalist visualization (Using styled function) ---
    print(f"Creating minimalist dominant strategy map (styled)...")
    try:
        create_dominant_strategy_map_styled( # Call the styled function
            grid_dominant_strategy, output_png, args.n_components,
            cmap_name=args.cmap, strategy_colors=args.strategy_colors, 
            figure_width=args.figure_width, figure_height=args.figure_height,
            bg_color=args.bg_color, land_color=args.land_color, ocean_color=args.ocean_color,
            legend_size=args.legend_size, alpha=args.alpha, # Pass style args
            strategy_labels=args.strategy_labels, legend_title=args.legend_title
        )
        create_dominant_strategy_map_styled( # Call again for PDF
             grid_dominant_strategy, output_pdf, args.n_components,
             cmap_name=args.cmap, strategy_colors=args.strategy_colors, 
             figure_width=args.figure_width, figure_height=args.figure_height,
             bg_color=args.bg_color, land_color=args.land_color, ocean_color=args.ocean_color,
             legend_size=args.legend_size, alpha=args.alpha, # Pass style args
             strategy_labels=args.strategy_labels, legend_title=args.legend_title
        )
    except Exception as e:
         print(f"Error during map creation: {e}")
         # raise e # Uncomment for debugging
         
    print("Done!")

if __name__ == "__main__":
    main()