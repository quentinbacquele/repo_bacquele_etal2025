#!/usr/bin/env python3

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl # Import base matplotlib
# mpl.use('Agg') # Uncomment if needed for non-interactive environments
import matplotlib.pyplot as plt
import os
import sys
import argparse
import ast
from matplotlib.colors import ListedColormap, LogNorm, Normalize # Add ListedColormap, LogNorm, Normalize
import matplotlib.cm as cm # Import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import logging # Add logging for consistency
from tqdm.auto import tqdm # Add tqdm for progress bars

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# --- Constants ---
# Reference Species Data (REQUIRED)
REFERENCE_FILE = "./data/traits_data_pc_gmm_8components_proba.csv" # Default reference file
REFERENCE_SPECIES_COL = 'species' # Default column in reference file
# Grid Data
GRID_SPECIES_FILE = "./data/grid_1.0deg_species_lists.csv"
GRID_SPECIES_COL = 'sci_name_list'
GRID_GEO_FILE = "./data/grid_1.0deg_coordID.gpkg"
# Output & Processing
OUTPUT_BASE_DIR = "output"
OUTPUT_SUBDIR = "richness_map" # Specific subdir for this map type
OUTPUT_PREFIX = "richness_1deg" # Default output prefix
DEFAULT_MIN_SPECIES = 8 # Default threshold applied AFTER filtering

# --- Custom Colormap Configuration ---
CUSTOM_CMAP_NAME = "nipy_spectral_darkred"
CUSTOM_CMAP_BASE = "nipy_spectral"
CUSTOM_CMAP_TRANSITION_POINT = 0.95
DARK_RED_RGBA = np.array([0.5, 0.0, 0.0, 1.0])
# ---

# --- Helper Functions ---
def standardize_species_name(name):
    """Replaces space with underscore, strips whitespace, and lowercases."""
    if isinstance(name, str):
        return name.replace(' ', '_').strip().lower()
    return name

def load_reference_species_set(source_csv_path, species_col_name):
    """Loads the required reference species set (standardized)."""
    logger.info(f"Loading reference species list from: {source_csv_path}")
    try:
        ref_df = pd.read_csv(source_csv_path, usecols=[species_col_name])
        reference_species_set = {s for s in ref_df[species_col_name].dropna().apply(standardize_species_name) if s}
        if not reference_species_set:
            logger.error(f"No valid species found in reference file {source_csv_path} using column '{species_col_name}'. Exiting.")
            sys.exit(1)
        logger.info(f"Loaded {len(reference_species_set)} unique reference species.")
        return reference_species_set
    except FileNotFoundError: logger.error(f"Reference species file not found: {source_csv_path}"); sys.exit(1)
    except KeyError: logger.error(f"Column '{species_col_name}' not found in reference file {source_csv_path}"); sys.exit(1)
    except Exception as e: logger.error(f"Error reading reference species file '{source_csv_path}': {e}"); sys.exit(1)

# --- Custom Colormap Registration Function ---
def register_custom_nipy_spectral_darkred(n_colors=256):
    """Registers the custom nipy_spectral colormap ending in dark red."""
    if CUSTOM_CMAP_NAME in mpl.colormaps: return True # Already registered
    try: base_cmap = mpl.colormaps[CUSTOM_CMAP_BASE]
    except KeyError: logger.error(f"Base colormap '{CUSTOM_CMAP_BASE}' not found."); return False
    except Exception as e: logger.error(f"Error getting base cmap: {e}"); return False
    try:
        n_original = int(np.floor(n_colors * CUSTOM_CMAP_TRANSITION_POINT))
        n_transition = n_colors - n_original
        if n_original <= 0 or n_transition <= 0: logger.error("Invalid color split."); return False
        original_points = np.linspace(0, CUSTOM_CMAP_TRANSITION_POINT, n_original, endpoint=False)
        colors_original = base_cmap(original_points)
        color_at_transition = base_cmap(CUSTOM_CMAP_TRANSITION_POINT)
        transition_colors = np.zeros((n_transition, 4)) # RGBA
        for i in range(4): transition_colors[:, i] = np.linspace(color_at_transition[i], DARK_RED_RGBA[i], n_transition)
        all_colors = np.vstack((colors_original, transition_colors))
        custom_cmap = ListedColormap(all_colors, name=CUSTOM_CMAP_NAME)
        mpl.colormaps.register(cmap=custom_cmap); logger.info(f"Registered custom cmap: '{CUSTOM_CMAP_NAME}'"); return True
    except Exception as e: logger.error(f"Failed custom cmap registration: {e}", exc_info=True); return False

# --- Map Plotting Function (Generic) ---
def create_generic_map(gdf, output_path, value_col_name, map_title_nice, cmap_name="viridis", vmin=0.0, vmax=1.0, figure_width=10.0, figure_height=5.0, bg_color="#FFFFFF", land_color="#FFFFFF", ocean_color="#E6ECF5", legend_size=0.03, alpha=1.0, base_legend_title="Value", log_scale=False):
    """Creates a generic map visualizing a column in a GeoDataFrame."""
    try:
        if cmap_name not in plt.colormaps(): logger.warning(f"Colormap '{cmap_name}' not found. Using 'viridis'."); cmap_name = "viridis"
        cmap = plt.get_cmap(cmap_name)
    except Exception as e: logger.error(f"Error getting colormap '{cmap_name}': {e}. Using 'viridis'."); cmap = plt.get_cmap("viridis")

    # Validate vmin/vmax and adjust if necessary
    if not (np.isfinite(vmin) and np.isfinite(vmax)): logger.warning(f"Non-finite vmin/vmax ({vmin=},{vmax=}) for '{map_title_nice}'. Using [0, 1]."); vmin=0.0; vmax=1.0
    elif vmin >= vmax:
        if np.isclose(vmin, vmax): logger.warning(f"vmin ({vmin=:.3g}) close to vmax ({vmax=:.3g}) for '{map_title_nice}'. Expanding range slightly."); vmin_orig = vmin; vmax = vmin_orig + 1 # Simple expansion
        else: logger.warning(f"vmin ({vmin=:.3g}) > vmax ({vmax=:.3g}) for '{map_title_nice}'. Using [0, 1]."); vmin, vmax = 0.0, 1.0
        if np.isclose(vmin, vmax): vmin, vmax = 0.0, 1.0 # Final check

    # Setup normalization
    if log_scale:
        if vmin <= 0: logger.warning(f"vmin ({vmin:.2f}) <= 0 for log scale. Adjusting vmin to 1."); vmin = 1
        if vmin >= vmax: vmax = vmin * 10 # Ensure vmax > vmin for log
        norm = LogNorm(vmin=vmin, vmax=vmax); logger.info(f"Using Log scale ('{cmap_name}') {vmin:.3g} to {vmax:.3g} for '{map_title_nice}'")
    else:
        norm = Normalize(vmin=vmin, vmax=vmax); logger.info(f"Using Linear scale ('{cmap_name}') {vmin:.3g} to {vmax:.3g} for '{map_title_nice}'")

    # --- Map Creation ---
    fig=plt.figure(figsize=(figure_width,figure_height))
    ax=fig.add_subplot(1,1,1,projection=ccrs.Robinson())
    fig.patch.set_facecolor(bg_color); ax.set_facecolor(bg_color); ax.set_global(); ax.axis('off')
    land=cfeature.LAND.with_scale('110m'); ocean=cfeature.OCEAN.with_scale('110m')
    ax.add_feature(land,edgecolor='none',facecolor=land_color,zorder=0)
    ax.add_feature(ocean,edgecolor='none',facecolor=ocean_color,zorder=0)

    if value_col_name not in gdf.columns: logger.error(f"Value column '{value_col_name}' not found in GeoDataFrame. Cannot map."); plt.close(fig); return

    # Ensure geometry is valid
    gdf_valid_geom = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].copy()
    if gdf_valid_geom.empty: logger.warning(f"No valid geometries found for plotting '{map_title_nice}'. Skipping map."); plt.close(fig); return

    # Reproject and get centroids
    try:
        gdf_pc = gdf_valid_geom.to_crs(ccrs.PlateCarree().proj4_init)
        if gdf_pc.empty: logger.warning(f"GeoDataFrame empty after CRS conversion for '{map_title_nice}'."); plt.close(fig); return
        gdf_pc['grid_lon'] = gdf_pc.geometry.centroid.x
        gdf_pc['grid_lat'] = gdf_pc.geometry.centroid.y
    except Exception as e: logger.error(f"Projection/Centroid error for '{map_title_nice}': {e}"); plt.close(fig); return

    # Prepare data for rasterization (only non-NaN values for the target column)
    data_to_plot=gdf_pc.dropna(subset=[value_col_name]).copy()
    mesh=None
    if data_to_plot.empty: logger.warning(f"No non-NaN data to plot for '{map_title_nice}'. Skipping mesh plot.")
    else:
        logger.info(f"Plotting {len(data_to_plot)} grid cells for '{map_title_nice}'.")
        lon_step = 1.0; lat_step = 1.0 # Assuming 1-degree grid
        x_grid=np.arange(-180, 180 + lon_step, lon_step); y_grid=np.arange(-90, 90 + lat_step, lat_step)
        value_grid=np.full((len(y_grid)-1,len(x_grid)-1), np.nan) # Initialize with NaN

        # Populate the grid
        for _, row in data_to_plot.iterrows():
            i=int(np.floor((row['grid_lon']+180) / lon_step))
            j=int(np.floor((row['grid_lat']+90) / lat_step))
            if 0<=i<value_grid.shape[1] and 0<=j<value_grid.shape[0]:
                 value_grid[j,i]=row[value_col_name]

        # Plot the grid if it contains any finite values
        if np.any(np.isfinite(value_grid)):
            mesh=ax.pcolormesh(x_grid,y_grid,value_grid,cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),alpha=alpha,shading='flat',rasterized=True)
            logger.debug("Mesh plotted.")
        else: logger.warning(f"Raster grid for '{map_title_nice}' contained only NaN values. No mesh plotted.")

    # Add colorbar if mesh was created
    if mesh:
        legend_height=legend_size; legend_width=0.5; legend_left=0.25; legend_bottom=0.05
        cax=fig.add_axes([legend_left,legend_bottom,legend_width,legend_height])
        cbar=fig.colorbar(mesh,cax=cax,orientation='horizontal')
        cbar.ax.tick_params(size=0,labelsize=8,color='#555555'); cbar.outline.set_visible(False)
        legend_title=f"{base_legend_title}\n({map_title_nice})"
        cbar.set_label(legend_title,size=9,color='#333333')
    else: logger.info("No mesh plotted, skipping colorbar.")

    # Save figure
    try: plt.savefig(output_path,dpi=300,bbox_inches='tight',pad_inches=0.05); logger.info(f"  Saved map: {output_path}")
    except Exception as e: logger.error(f"  Failed to save map to {output_path}: {e}")
    finally: plt.close(fig) # Ensure figure is closed

# Set publication-quality defaults (minimal)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':10,'axes.linewidth':0.5,'savefig.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':0.0})

def main():
    # --- Register Custom Colormap ---
    if not register_custom_nipy_spectral_darkred():
        logger.warning("Custom colormap registration failed. Falling back to 'viridis'.")
        default_cmap = "viridis"
    else:
        default_cmap = CUSTOM_CMAP_NAME

    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description="Generate filtered species richness map based on a reference list.")
    # Input Files
    parser.add_argument("--grid-file", type=str, default=GRID_GEO_FILE, help="Path to grid geospatial file (e.g., GeoPackage)")
    parser.add_argument("--species-lists-file", type=str, default=GRID_SPECIES_FILE, help="Path to CSV with grid_id and species lists (sci_name_list column)")
    parser.add_argument("--reference-species-file", type=str, default=REFERENCE_FILE, help="Path to reference species list CSV (used for filtering)")
    parser.add_argument("--reference-species-col", type=str, default=REFERENCE_SPECIES_COL, help="Column name containing species in the reference file")
    # Output Control
    parser.add_argument("--output-dir", type=str, default=OUTPUT_BASE_DIR, help="Base output directory")
    parser.add_argument("--output-prefix", type=str, default=OUTPUT_PREFIX, help="Prefix for output file names")
    # Filtering Thresholds
    parser.add_argument("--min-species-threshold", type=int, default=DEFAULT_MIN_SPECIES, help="Min number of *reference-filtered* species required for a cell to be included in calculations and output CSV.")
    parser.add_argument("--min-richness-display", type=float, default=1.0, help="Min filtered richness value to *color* on the map (cells below this get background color). Set <= min-species-threshold to show all included cells.")
    # Map Styling
    parser.add_argument("--cmap", type=str, default=default_cmap, help="Matplotlib colormap")
    parser.add_argument("--vmin", type=float, default=None, help="Min value for color normalization (overrides auto-calc)")
    parser.add_argument("--vmax", type=float, default=None, help="Max value for color normalization (overrides auto-calc)")
    parser.add_argument("--figure-width", type=float, default=10.0, help="Figure width (inches)")
    parser.add_argument("--figure-height", type=float, default=5.0, help="Figure height (inches)")
    parser.add_argument("--bg-color", type=str, default="#FFFFFF", help="Map background color")
    parser.add_argument("--land-color", type=str, default="#FFFFFF", help="Map land color")
    parser.add_argument("--ocean-color", type=str, default="#E6ECF5", help="Map ocean color")
    parser.add_argument("--legend-size", type=float, default=0.03, help="Relative height of the colorbar legend")
    parser.add_argument("--alpha", type=float, default=1.0, help="Transparency level for plotted data (0=transparent, 1=opaque)")
    parser.add_argument("--legend-title", type=str, default="Filtered Species Richness", help="Title for the colorbar legend")
    parser.add_argument("--log-scale", action="store_true", help="Use logarithmic color scale for display")
    args = parser.parse_args()

    # --- Validate Arguments ---
    if args.min_richness_display < 0: logger.warning("--min-richness-display should be non-negative."); args.min_richness_display = 0
    if args.min_species_threshold <= 0: logger.warning("--min-species-threshold should be positive."); args.min_species_threshold = 1
    if args.alpha < 0 or args.alpha > 1: logger.error("--alpha must be between 0 and 1."); return 1

    # --- Setup Paths ---
    full_output_dir = os.path.join(args.output_dir, OUTPUT_SUBDIR)
    output_prefix_final = f"{args.output_prefix}" # Add threshold info to name
    output_csv = os.path.join(full_output_dir, f"{output_prefix_final}.csv")
    output_png = os.path.join(full_output_dir, f"{output_prefix_final}.png")
    output_pdf = os.path.join(full_output_dir, f"{output_prefix_final}.pdf")
    os.makedirs(full_output_dir, exist_ok=True)
    logger.info(f"Output directory: {full_output_dir}")
    logger.info(f"Output file prefix: {output_prefix_final}")
    logger.info(f"Using colormap: {args.cmap}")
    logger.info(f"Reference species file for filtering: {args.reference_species_file} (Column: {args.reference_species_col})")
    logger.info(f"Minimum *filtered* species threshold for inclusion: {args.min_species_threshold}")
    logger.info(f"Minimum richness threshold for display (coloring): {args.min_richness_display}")

    # --- Load Reference Species Set ---
    reference_species_set = load_reference_species_set(args.reference_species_file, args.reference_species_col)
    # Exit handled within function if loading fails

    # --- Load Species Lists ---
    logger.info(f"Loading species lists from {args.species_lists_file}...")
    try:
        # Ensure grid_id is read as string
        grid_species_df = pd.read_csv(args.species_lists_file, dtype={'grid_id': str})
        if 'grid_id' not in grid_species_df.columns or GRID_SPECIES_COL not in grid_species_df.columns:
             logger.error(f"Species lists file '{args.species_lists_file}' missing required columns 'grid_id' or '{GRID_SPECIES_COL}'."); return 1
    except FileNotFoundError: logger.error(f"Species lists file not found: {args.species_lists_file}"); return 1
    except Exception as e: logger.error(f"Error loading species lists file: {e}"); return 1
    logger.info(f"Loaded species data for {len(grid_species_df)} grid cells")

    # --- Calculate Filtered Richness & Apply Threshold ---
    logger.info("Calculating filtered species richness per cell...")
    richness_results = []
    skipped_parse_error = 0
    skipped_no_species_initially = 0
    skipped_min_species_count = 0
    processed_count = 0
    total_cells_input = len(grid_species_df)

    for _, row in tqdm(grid_species_df.iterrows(), total=total_cells_input, desc="Processing Cells"):
        grid_id = row['grid_id'] # Already string

        # 1. Parse Raw Species List
        species_list_str = row[GRID_SPECIES_COL]
        raw_species_set = set()
        if isinstance(species_list_str, str) and species_list_str.startswith('[') and species_list_str.endswith(']'):
            try:
                parsed_list = ast.literal_eval(species_list_str)
                raw_species_set = {standardize_species_name(sp) for sp in parsed_list if isinstance(sp, (str, int, float)) and standardize_species_name(sp)}
            except (ValueError, SyntaxError): skipped_parse_error += 1; continue
        elif not pd.isna(species_list_str): skipped_parse_error += 1; continue
        if not raw_species_set: skipped_no_species_initially += 1; continue

        # 2. Filter Species List using Reference Set
        filtered_species_set = raw_species_set.intersection(reference_species_set)

        # 3. Calculate Filtered Richness
        filtered_richness = len(filtered_species_set)

        # 4. Filter by Minimum *Filtered* Species Threshold (Calculation/Inclusion)
        if filtered_richness < args.min_species_threshold:
            skipped_min_species_count += 1; continue

        # If passed threshold, store result
        processed_count += 1
        richness_results.append({'grid_id': grid_id, 'filtered_richness': filtered_richness})

    logger.info(f"Richness calculation summary:")
    logger.info(f"  Total input cells: {total_cells_input}")
    logger.info(f"  Skipped due to parsing error: {skipped_parse_error}")
    logger.info(f"  Skipped due to no initial species: {skipped_no_species_initially}")
    logger.info(f"  Skipped by min filtered species threshold (< {args.min_species_threshold}): {skipped_min_species_count}")
    logger.info(f"  Cells included in output CSV (passed threshold): {processed_count}")

    # --- Process & Save Results ---
    if not richness_results:
         logger.warning("No cells met the minimum filtered richness threshold. Output CSV will be empty and map will not be generated.")
         richness_df = pd.DataFrame(columns=['grid_id', 'filtered_richness'])
         try: richness_df.to_csv(output_csv, index=False); logger.info(f"Saved empty richness data to {output_csv}")
         except Exception as e: logger.error(f"Failed to save empty richness data: {e}")
         return 0 # Exit cleanly
    else:
        richness_df = pd.DataFrame(richness_results)
        richness_df['grid_id'] = richness_df['grid_id'].astype(str) # Ensure consistent type
        logger.info(f"\nFiltered Species Richness Statistics (for {len(richness_df)} included cells):")
        logger.info(f"  Mean: {richness_df['filtered_richness'].mean():.2f}, Median: {richness_df['filtered_richness'].median():.2f}")
        logger.info(f"  Min: {richness_df['filtered_richness'].min():.2f}, Max: {richness_df['filtered_richness'].max():.2f}")
        cells_meeting_display_thresh = (richness_df['filtered_richness'] >= args.min_richness_display).sum()
        logger.info(f"  Cells meeting *display* threshold (>= {args.min_richness_display}): {cells_meeting_display_thresh} / {len(richness_df)}")
        try: richness_df.to_csv(output_csv, index=False); logger.info(f"Saved filtered richness data to {output_csv}")
        except Exception as e: logger.error(f"Failed to save richness data: {e}")

    # --- Mapping ---
    # Load Grid Geometry
    logger.info(f"Loading grid cell geometry from {args.grid_file}...")
    try:
        grid_gdf = gpd.read_file(args.grid_file, dtype={'grid_id': str}) # Ensure string grid_id
        if 'grid_id' not in grid_gdf.columns: logger.error(f"Grid file missing 'grid_id'."); return 1
    except FileNotFoundError: logger.error(f"Grid geometry file not found: {args.grid_file}"); return 1
    except Exception as e: logger.error(f"Failed to load grid file: {e}"); return 1

    # Merge Filtered Richness Data
    logger.info("Merging filtered richness data with grid geometry...")
    try:
        grid_richness = grid_gdf.merge(richness_df, on='grid_id', how='left') # Left merge keeps all geometries
        logger.info(f"Merge complete. Result shape: {grid_richness.shape}")
        merged_cells_with_data = grid_richness['filtered_richness'].notna().sum()
        logger.info(f"Successfully merged richness data for {merged_cells_with_data} grid cells.")
        if merged_cells_with_data != len(richness_df):
             logger.warning(f"Merge issue: Expected {len(richness_df)} cells with data, but merged {merged_cells_with_data}. Check 'grid_id' consistency.")
    except Exception as e: logger.error(f"Error merging: {e}"); return 1

    # Determine Map Scale (Vmin/Vmax) for Display
    # Use only the non-NaN richness values that also meet the display threshold
    data_for_display_scaling = grid_richness['filtered_richness'].dropna()
    data_for_display_scaling = data_for_display_scaling[data_for_display_scaling >= args.min_richness_display]

    if data_for_display_scaling.empty:
        logger.warning(f"No included cells meet the display threshold (>= {args.min_richness_display}). Map may be empty or show only background.")
        vmin_final, vmax_final = 1, 10 # Placeholder values
    else:
        vmin_auto = data_for_display_scaling.min()
        vmax_auto = data_for_display_scaling.max()
        vmin_final = args.vmin if args.vmin is not None else vmin_auto
        vmax_final = args.vmax if args.vmax is not None else vmax_auto
        # Ensure vmin < vmax
        if vmin_final >= vmax_final:
             if np.isclose(vmin_final, vmax_final): vmax_final = vmin_final + 1
             else: vmin_final, vmax_final = vmin_auto, vmax_auto
             if np.isclose(vmin_final, vmax_final): vmin_final, vmax_final = 1, max(2, vmax_final + 1) # Final fallback

    # Create and Save Map
    logger.info(f"Creating filtered richness map...")
    # Prepare the GeoDataFrame for mapping: Set values below display threshold to NaN
    # This ensures they are not colored by pcolormesh
    map_gdf = grid_richness.copy()
    map_gdf.loc[map_gdf['filtered_richness'] < args.min_richness_display, 'filtered_richness'] = np.nan

    create_generic_map(
        gdf=map_gdf, # Use the modified gdf with NaNs for below-display-threshold
        output_path=output_png,
        value_col_name='filtered_richness',
        map_title_nice="Filtered Richness",
        cmap_name=args.cmap,
        vmin=vmin_final,
        vmax=vmax_final,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        bg_color=args.bg_color,
        land_color=args.land_color,
        ocean_color=args.ocean_color,
        legend_size=args.legend_size,
        alpha=args.alpha,
        base_legend_title=args.legend_title,
        log_scale=args.log_scale
    )
    # Save PDF version
    create_generic_map(
        gdf=map_gdf, # Use the same modified gdf
        output_path=output_pdf,
        value_col_name='filtered_richness',
        map_title_nice="Filtered Richness",
        cmap_name=args.cmap,
        vmin=vmin_final,
        vmax=vmax_final,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        bg_color=args.bg_color,
        land_color=args.land_color,
        ocean_color=args.ocean_color,
        legend_size=args.legend_size,
        alpha=args.alpha,
        base_legend_title=args.legend_title,
        log_scale=args.log_scale
    )

    logger.info("Done!"); return 0

if __name__ == "__main__":
    sys.exit(main())