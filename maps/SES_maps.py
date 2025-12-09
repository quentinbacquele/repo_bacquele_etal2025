#!/usr/bin/env python3

# --- Core Analysis Imports ---
import numpy as np
import pandas as pd
import ast
import os
import time
import logging
import sys
import argparse
from tqdm.auto import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import pickle
import gc
from multiprocessing import Manager

# --- Standard Map/Grid Imports ---
import geopandas as gpd
from shapely.geometry import box
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns # For potentially nicer scatter plots
from matplotlib.colors import ListedColormap, LogNorm, Normalize, BoundaryNorm, CenteredNorm

# Configure logging (Standard Setup)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Constants and Defaults ---
PROPORTION_THRESHOLD = 0.05 # Threshold for species count metric
THRESH_LABEL = f"{PROPORTION_THRESHOLD*100:.0f}" # e.g., "5"
DEFAULT_VALID_CELLS_FILE = "./output/richness_map/richness_1deg.csv" # MANDATORY FILTER
DEFAULT_REALM_MAP_FILE = "./output/biogeographical_realms_map/global_biogeographical_realms_map_1deg_agg_filtered_aligned_data.csv" # Realm mapping file
DEFAULT_SPECIES_PROBS_FILE = "./data/model_traits_data.csv" # INPUT FILE WITH MEAN PROBS
GRID_ID_PRECISION = 6
OUTPUT_SUBDIR = "proportion_analysis_vs_null_realm" # Subdir for all outputs

# --- Custom Colormap Configuration (Standard, plus diverging) ---
CUSTOM_CMAP_NAME = "nipy_spectral_darkred"
CUSTOM_CMAP_BASE = "nipy_spectral"
CUSTOM_CMAP_TRANSITION_POINT = 0.95
DARK_RED_RGBA = np.array([0.5, 0.0, 0.0, 1.0])
DIVERGING_CMAP_NAME = 'coolwarm' # Standard diverging colormap for SES

# --- Set publication-quality defaults (Standard) ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 0.5,
    'savefig.dpi': 300, # Default DPI, can be overridden by arg
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1, # Adjusted slightly for scatter plot labels
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})
# Specific settings for scatter plots if desired
sns.set_style("whitegrid")

# --- Helper Functions (Standard Grid ID) ---
def format_coord_for_id(coord_val, precision=GRID_ID_PRECISION):
    fmt = f"{{:.{precision}f}}"
    return fmt.format(coord_val).replace('.', 'p').replace('-', 'n')

def generate_coordinate_grid_id(lon, lat, precision=GRID_ID_PRECISION):
    lon_str = format_coord_for_id(lon, precision)
    lat_str = format_coord_for_id(lat, precision)
    return f"lon{lon_str}_lat{lat_str}"

# --- Custom Colormap Registration Function (Standard) ---
def register_custom_nipy_spectral_darkred(n_colors=256):
    if CUSTOM_CMAP_NAME in mpl.colormaps: logger.debug(f"Custom cmap '{CUSTOM_CMAP_NAME}' already registered."); return True
    try: base_cmap = mpl.colormaps[CUSTOM_CMAP_BASE]
    except KeyError: logger.error(f"Base colormap '{CUSTOM_CMAP_BASE}' not found."); return False
    try:
        n_original = int(np.floor(n_colors * CUSTOM_CMAP_TRANSITION_POINT)); n_transition = n_colors - n_original
        if n_original <= 0 or n_transition <= 0: logger.error(f"Invalid color split: {n_original=}, {n_transition=}"); return False
        original_points = np.linspace(0, CUSTOM_CMAP_TRANSITION_POINT, n_original, endpoint=False); colors_original = base_cmap(original_points)
        color_at_transition = base_cmap(CUSTOM_CMAP_TRANSITION_POINT); transition_colors = np.zeros((n_transition, 4))
        for i in range(4): transition_colors[:, i] = np.linspace(color_at_transition[i], DARK_RED_RGBA[i], n_transition)
        all_colors = np.vstack((colors_original, transition_colors)); custom_cmap = ListedColormap(all_colors, name=CUSTOM_CMAP_NAME)
        mpl.colormaps.register(cmap=custom_cmap); logger.info(f"Registered custom cmap: '{CUSTOM_CMAP_NAME}'"); return True
    except Exception as e: logger.error(f"Failed custom cmap registration: {e}", exc_info=True); return False


# --- Data Loading and Management Class ---
class SpeciesDataManager:
    """Loads grid data, species probabilities, realm info, and applies initial filtering."""
    def __init__(self,
                 species_probs_file=DEFAULT_SPECIES_PROBS_FILE,
                 valid_cells_file=DEFAULT_VALID_CELLS_FILE,
                 realm_map_file=DEFAULT_REALM_MAP_FILE):
        self.species_probs_file = species_probs_file
        self.valid_cells_file = valid_cells_file
        self.realm_map_file = realm_map_file
        self.grid_df = None
        self.species_proba_dict = {}
        self.valid_species_set = set() # Species with probability data
        self.realm_species_pools = {} # Dict: realm_name -> list of valid species in that realm
        self.prob_cols = []
        self.n_components = 0

    def load_data(self):
        """Loads all necessary data and performs initial filtering."""
        start_time = time.time()

        # 1. Load Valid Grid Cell IDs
        logger.info(f"Loading valid grid cell IDs from {self.valid_cells_file}...")
        try:
            valid_cells_df = pd.read_csv(self.valid_cells_file, usecols=['grid_id'], dtype={'grid_id': str})
            self.valid_grid_ids = set(valid_cells_df['grid_id'].unique())
            if not self.valid_grid_ids:
                raise ValueError("No valid grid IDs found in the provided valid cells file.")
            logger.info(f"Loaded {len(self.valid_grid_ids)} unique valid grid IDs.")
        except FileNotFoundError:
            logger.error(f"Valid cells file not found: {self.valid_cells_file}")
            raise
        except Exception as e:
            logger.error(f"Error loading valid cells file: {e}")
            raise

        # 2. Load Grid Species Data and Filter
        logger.info("Loading grid species data...")
        try:
            grid_df_raw = pd.read_csv("data/grid_1.0deg_species_lists.csv", dtype={'grid_id': str})
            if 'grid_id' not in grid_df_raw.columns or 'sci_name_list' not in grid_df_raw.columns:
                 raise ValueError("Grid CSV missing 'grid_id' or 'sci_name_list' columns.")

            # --- Apply Mandatory Grid Filter ---
            initial_grid_count = len(grid_df_raw)
            self.grid_df = grid_df_raw[grid_df_raw['grid_id'].isin(self.valid_grid_ids)].copy()
            filtered_grid_count = len(self.grid_df)
            logger.info(f"Filtered grid cells based on valid IDs: {initial_grid_count} -> {filtered_grid_count}")
            if filtered_grid_count == 0:
                logger.error("No grid cells remained after filtering by valid IDs. Cannot proceed.")
                raise ValueError("Filtering resulted in zero valid grid cells.")

            # Parse species lists for the remaining valid grids
            def parse_species_list(sp_list_str):
                if pd.isna(sp_list_str): return []
                try:
                    sp_list = ast.literal_eval(sp_list_str)
                    # Standardize species names (replace space with underscore)
                    return [sp.replace(' ', '_') for sp in sp_list if isinstance(sp, str)]
                except (ValueError, SyntaxError): return []
            self.grid_df['species_list_parsed'] = self.grid_df['sci_name_list'].apply(parse_species_list)
            logger.info(f"Parsed species lists for {len(self.grid_df)} filtered grid cells.")

        except FileNotFoundError:
            logger.error("Grid species list file not found: data/grid_1.0deg_species_lists.csv")
            raise
        except Exception as e:
            logger.error(f"Error loading or parsing grid species list: {e}")
            raise

        # 3. Load Species Mean Probabilities
        logger.info(f"Loading species mean probabilities from {self.species_probs_file}...")
        try:
            traits_df = pd.read_csv(self.species_probs_file)
            self.prob_cols = sorted([col for col in traits_df.columns if col.startswith('gmm_prob_') and col.endswith('_mean')])
            if not self.prob_cols:
                 raise ValueError(f"No columns matching 'gmm_prob_*_mean' found in {self.species_probs_file}")
            self.n_components = len(self.prob_cols)
            logger.info(f"Found {self.n_components} probability columns: {self.prob_cols}")

            if 'species' not in traits_df.columns: raise ValueError("Missing 'species' column.")
            traits_df['species_key'] = traits_df['species'].str.replace(' ', '_') # Consistent formatting

            # Create species -> probability vector dictionary and identify valid species
            self.species_proba_dict = {}
            valid_species_temp = []
            for _, row in traits_df.iterrows():
                species_key = row['species_key']
                prob_vector = row[self.prob_cols].values.astype(float)
                # Optional: Normalize if sum is not close to 1
                if not np.isclose(np.sum(prob_vector), 1.0, atol=1e-4):
                    prob_sum = np.sum(prob_vector)
                    if prob_sum > 1e-6: # Avoid division by zero
                       prob_vector /= prob_sum
                    else: # If sum is zero, can't normalize, keep as is (maybe uniform?) - For now, keep zeros
                       pass # logger.debug(f"Probabilities sum to zero for {species_key}. Cannot normalize.")
                self.species_proba_dict[species_key] = prob_vector
                valid_species_temp.append(species_key)
            self.valid_species_set = set(valid_species_temp)
            logger.info(f"Loaded and processed mean probabilities for {len(self.valid_species_set)} unique species.")

        except FileNotFoundError:
            logger.error(f"Species probability file not found: {self.species_probs_file}")
            raise
        except Exception as e:
            logger.error(f"Error loading or processing species probabilities: {e}")
            raise

        # 4. Load Realm Data and Merge
        logger.info(f"Loading biogeographical realm data from {self.realm_map_file}...")
        try:
            realm_df = pd.read_csv(self.realm_map_file, usecols=['grid_id', 'realm_name'], dtype={'grid_id': str})
            # Rename for clarity before merge if needed, e.g., realm_df.rename(columns={'realm_name': 'biogeo_realm'}, inplace=True)
            realm_df.dropna(subset=['realm_name'], inplace=True) # Remove grids without assigned realm if necessary

            # Merge realm info into the filtered grid dataframe
            initial_grid_count = len(self.grid_df)
            self.grid_df = pd.merge(self.grid_df, realm_df, on='grid_id', how='left')
            merged_count = len(self.grid_df)
            if initial_grid_count != merged_count:
                logger.warning("Merge operation changed the number of rows. Check grid_ids.")

            # Handle cells potentially missing a realm after merge (assign 'Unknown' or filter)
            cells_no_realm = self.grid_df['realm_name'].isnull().sum()
            if cells_no_realm > 0:
                logger.warning(f"{cells_no_realm} grid cells lack a realm assignment after merge. Assigning 'Unknown'.")
                self.grid_df['realm_name'].fillna('Unknown', inplace=True)
            logger.info("Merged realm information with grid data.")

        except FileNotFoundError:
            logger.error(f"Realm map file not found: {self.realm_map_file}")
            raise # Or handle differently, e.g., skip realm-based null model
        except Exception as e:
            logger.error(f"Error loading or merging realm data: {e}")
            raise

        # 5. Create Realm-Specific Species Pools
        logger.info("Creating realm-specific species pools for null model...")
        self.realm_species_pools = {}
        # Group grid cells by realm
        grouped_by_realm = self.grid_df.groupby('realm_name')['species_list_parsed']

        for realm_name, species_lists in grouped_by_realm:
            if realm_name == 'Unknown': continue # Skip the unknown category for pooling
            # Flatten the list of lists and get unique species for the realm
            all_species_in_realm = set(sp for sublist in species_lists for sp in sublist)
            # Filter these species to keep only those with valid probability data
            valid_species_in_realm = list(all_species_in_realm.intersection(self.valid_species_set))
            if valid_species_in_realm:
                self.realm_species_pools[realm_name] = valid_species_in_realm
            else:
                logger.warning(f"No valid species (with probability data) found for realm: {realm_name}")

        # Add an 'Unknown' pool? Or maybe a global pool as fallback? For now, only known realms.
        num_realms_with_pools = len(self.realm_species_pools)
        logger.info(f"Created species pools for {num_realms_with_pools} realms.")
        if num_realms_with_pools == 0 and len(self.grid_df) > 0:
             logger.error("Failed to create any realm-specific species pools, though grid cells exist. Cannot run realm-based null model.")
             raise ValueError("Realm pool creation failed.")


        logger.info(f"Data loading and preparation finished in {time.time() - start_time:.2f} seconds.")
        return self.grid_df, self.species_proba_dict, self.valid_species_set, self.realm_species_pools, self.n_components

# --- Observed Metrics Calculation ---
def calculate_observed_metrics_for_cell(args_tuple):
    """Calculates observed metrics for a single grid cell."""
    grid_id, species_list_parsed, species_proba_dict, valid_species_set, n_components = args_tuple

    # Find species present in the cell that have probability data
    matched_species = [sp for sp in species_list_parsed if sp in valid_species_set]
    num_species_in_cell = len(matched_species)

    # Prepare default results
    results = {
        'grid_id': grid_id,
        'num_species': num_species_in_cell, # Store the *matched* richness
        'status': 'success' if num_species_in_cell > 0 else 'no_valid_species',
        **{f'mean_prob_{i}': np.nan for i in range(n_components)},
        **{f'count_gt{THRESH_LABEL}_{i}': 0 for i in range(n_components)} # Default count is 0
    }

    if num_species_in_cell > 0:
        try:
            # Retrieve probability vectors for matched species
            species_probs_list = [species_proba_dict[sp] for sp in matched_species]
            species_probs_array = np.array(species_probs_list)

            # 1. Mean proportion per strategy
            grid_mean_probs = np.mean(species_probs_array, axis=0)
            for i in range(n_components):
                results[f'mean_prob_{i}'] = grid_mean_probs[i]

            # 2. Count species with proportion > threshold per strategy
            for i in range(n_components):
                count = np.sum(species_probs_array[:, i] > PROPORTION_THRESHOLD)
                results[f'count_gt{THRESH_LABEL}_{i}'] = count

        except Exception as e:
            logger.error(f"Error calculating observed metrics for grid {grid_id}: {e}")
            results['status'] = 'error'
            results.update({f'mean_prob_{i}': np.nan for i in range(n_components)})
            results.update({f'count_gt{THRESH_LABEL}_{i}': 0 for i in range(n_components)})

    return results

def calculate_observed_metrics(grid_df, species_data_manager, workers=None):
    """Calculates mean proportions and counts for all filtered grid cells in parallel."""
    logger.info("Calculating observed metrics (mean proportion, count > threshold)...")
    start_time = time.time()

    # Prepare arguments for parallel processing using the filtered grid_df
    task_args = [
        (row.grid_id, row.species_list_parsed, species_data_manager.species_proba_dict,
         species_data_manager.valid_species_set, species_data_manager.n_components)
        for _, row in grid_df.iterrows() # grid_df is already filtered
    ]

    if not task_args:
        logger.warning("No tasks generated for observed metrics calculation (grid_df might be empty).")
        return pd.DataFrame()

    all_results = []
    effective_workers = workers if workers is not None else multiprocessing.cpu_count() - 1
    if effective_workers < 1 : effective_workers = 1

    if effective_workers == 1 or len(task_args) < 100: # Use single process for small tasks
         logger.info("Using single process for observed metrics calculation.")
         for args_tuple in tqdm(task_args, desc="Calculating Observed Metrics"):
              all_results.append(calculate_observed_metrics_for_cell(args_tuple))
    else:
         logger.info(f"Using {effective_workers} workers for observed metrics calculation.")
         # Consider using shared memory for species_proba_dict if it's very large, but Manager is simpler for now
         with ProcessPoolExecutor(max_workers=effective_workers) as executor:
              results_iterator = executor.map(calculate_observed_metrics_for_cell, task_args)
              all_results = list(tqdm(results_iterator, total=len(task_args), desc="Calculating Observed Metrics"))

    observed_df = pd.DataFrame(all_results)
    if observed_df.empty:
        logger.warning("Observed metrics calculation resulted in an empty DataFrame.")
        return observed_df

    num_success = (observed_df['status'] == 'success').sum()
    num_processed = len(observed_df)
    success_rate = num_success / num_processed if num_processed > 0 else 0
    logger.info(f"Observed metrics calculation finished in {time.time() - start_time:.2f} seconds.")
    logger.info(f"Processed {num_success}/{num_processed} ({success_rate*100:.1f}%) grid cells successfully.")

    return observed_df

# --- Null Model Calculation (Realm Specific) ---
def calculate_null_metrics_for_cell_realm(args_tuple):
    """Calculates null model metrics using realm-specific species pools."""
    grid_id, realm_name, num_species_in_cell, species_proba_dict, realm_species_pools, num_permutations, n_components = args_tuple

    # Determine the correct species pool for this cell's realm
    current_species_pool = realm_species_pools.get(realm_name)

    # Handle cases: no species in observed cell, unknown realm, or empty realm pool
    if num_species_in_cell == 0 or not current_species_pool:
        status = 'skipped_no_species' if num_species_in_cell == 0 else 'skipped_no_realm_pool'
        null_means = np.full((num_permutations, n_components), np.nan)
        null_counts = np.full((num_permutations, n_components), 0) # Use 0 for counts? Or Nan? Let's stick to 0.
        return grid_id, status, null_means, null_counts

    pool_size = len(current_species_pool)
    if pool_size < num_species_in_cell:
        # Decide how to handle this: Sample with replacement (current default), or skip?
        # Sampling with replacement allows simulation even if pool < N.
        # logger.warning(f"Realm pool size ({pool_size}) < cell richness ({num_species_in_cell}) for {grid_id} in {realm_name}. Sampling with replacement.")
        pass # Proceed with replacement

    # Store results for this cell across permutations
    null_means_cell = np.zeros((num_permutations, n_components))
    null_counts_cell = np.zeros((num_permutations, n_components))

    for perm in range(num_permutations):
        try:
            # Randomly draw N species from the *realm-specific* pool with replacement
            random_species_names = np.random.choice(current_species_pool, size=num_species_in_cell, replace=True)

            # Get probability vectors for the randomized community
            # This step might be slow if species_proba_dict is huge. Consider optimizations if needed.
            species_probs_list = [species_proba_dict[sp] for sp in random_species_names]
            species_probs_array = np.array(species_probs_list)

            # Calculate metrics for the randomized community
            null_mean_probs = np.mean(species_probs_array, axis=0)
            null_means_cell[perm, :] = null_mean_probs

            for i in range(n_components):
                count = np.sum(species_probs_array[:, i] > PROPORTION_THRESHOLD)
                null_counts_cell[perm, i] = count

        except Exception as e:
            # logger.error(f"Error during null model perm {perm} for grid {grid_id} ({realm_name}): {e}") # Can be verbose
            null_means_cell[perm, :] = np.nan
            null_counts_cell[perm, :] = 0 # Or NaN? Keep 0 for count.

    return grid_id, 'success', null_means_cell, null_counts_cell

def run_null_model_realm(observed_metrics_df, species_data_manager, num_permutations, workers=None):
    """Runs the realm-specific null model."""
    logger.info(f"Running realm-specific null model with {num_permutations} permutations...")
    start_time = time.time()

    # Ensure observed_metrics_df has 'realm_name' (it should from data_manager)
    if 'realm_name' not in observed_metrics_df.columns:
        logger.error("Column 'realm_name' is missing from observed_metrics_df. Cannot run realm null model.")
        # Attempt to merge it back in if grid_df is accessible? Risky. Better to stop.
        raise ValueError("Missing realm_name in observed data.")
    if 'num_species' not in observed_metrics_df.columns:
        raise KeyError("'num_species' column is missing from observed metrics DataFrame.")

    # Prepare arguments for parallel processing
    task_args = []
    # We process all rows from observed_metrics_df; the worker function handles skipping if needed
    for _, row in observed_metrics_df.iterrows():
         # Ensure num_species is integer, handle potential NaN from observed calc errors
         num_species = 0 if pd.isna(row['num_species']) else int(row['num_species'])
         task_args.append((
             row.grid_id,
             row.realm_name, # Pass the realm name
             num_species, # Pass the observed *matched* richness
             species_data_manager.species_proba_dict,
             species_data_manager.realm_species_pools, # Pass the realm pools dict
             num_permutations,
             species_data_manager.n_components
         ))

    if not task_args:
        logger.warning("No tasks generated for null model calculation.")
        return {}

    null_results_dict = {}
    effective_workers = workers if workers is not None else multiprocessing.cpu_count() - 1
    if effective_workers < 1 : effective_workers = 1

    if effective_workers == 1 or len(task_args) < 100:
        logger.info("Using single process for realm null model calculation.")
        for args_tuple in tqdm(task_args, desc="Running Realm Null Model"):
            grid_id, status, null_means, null_counts = calculate_null_metrics_for_cell_realm(args_tuple)
            if status == 'success':
                null_results_dict[grid_id] = {'null_means': null_means, 'null_counts': null_counts}
            # else: logger.debug(f"Null model skipped for {grid_id} ({status})")
    else:
        logger.info(f"Using {effective_workers} workers for realm null model calculation.")
        # Consider shared memory again for species_proba_dict and realm_species_pools if performance is an issue
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            results_iterator = executor.map(calculate_null_metrics_for_cell_realm, task_args)
            for grid_id, status, null_means, null_counts in tqdm(results_iterator, total=len(task_args), desc="Running Realm Null Model"):
                if status == 'success':
                    null_results_dict[grid_id] = {'null_means': null_means, 'null_counts': null_counts}
                # else: logger.debug(f"Null model skipped for {grid_id} ({status})")


    logger.info(f"Realm null model calculation finished in {time.time() - start_time:.2f} seconds.")
    logger.info(f"Generated null distributions for {len(null_results_dict)} grid cells.")
    return null_results_dict


# --- SES Calculation ---
def calculate_ses(observed_metrics_df, null_results_dict, n_components):
    """Calculates Standardized Effect Size (SES) comparing observed to null metrics."""
    logger.info("Calculating Standardized Effect Size (SES)...")
    # Start from the observed results (which includes grid_id, num_species, realm_name, obs metrics)
    ses_df = observed_metrics_df.copy()

    # Initialize SES columns with NaN
    for i in range(n_components):
        ses_df[f'ses_mean_prob_{i}'] = np.nan
        ses_df[f'ses_count_gt{THRESH_LABEL}_{i}'] = np.nan

    grids_missing_null = 0
    for index, obs_row in tqdm(ses_df.iterrows(), total=len(ses_df), desc="Calculating SES"):
        grid_id = obs_row['grid_id']
        # Check if null results exist for this grid_id
        if grid_id not in null_results_dict:
            grids_missing_null += 1
            continue # Skip SES calculation if null model failed/skipped for this cell

        null_data = null_results_dict[grid_id]
        null_means = null_data['null_means'] # Shape: (num_permutations, n_components)
        null_counts = null_data['null_counts'] # Shape: (num_permutations, n_components)

        for i in range(n_components):
            # --- SES for Mean Probability ---
            obs_mean = obs_row[f'mean_prob_{i}']
            if pd.isna(obs_mean): continue # Skip if observed value is NaN

            null_means_i = null_means[:, i]
            valid_null_means_i = null_means_i[~np.isnan(null_means_i)] # Filter NaNs from null if errors occurred

            if len(valid_null_means_i) < 2: # Need at least 2 points for std dev
                ses_df.loc[index, f'ses_mean_prob_{i}'] = np.nan
            else:
                mean_null = np.mean(valid_null_means_i)
                std_null = np.std(valid_null_means_i)
                if std_null > 1e-9: # Avoid division by zero or near-zero
                    ses = (obs_mean - mean_null) / std_null
                    ses_df.loc[index, f'ses_mean_prob_{i}'] = ses
                else: # Handle zero std dev in null distribution
                    ses_df.loc[index, f'ses_mean_prob_{i}'] = 0.0 if np.isclose(obs_mean, mean_null) else np.inf * np.sign(obs_mean - mean_null)

            # --- SES for Count > Threshold ---
            obs_count = obs_row[f'count_gt{THRESH_LABEL}_{i}']
            if pd.isna(obs_count): continue # Skip if observed value is NaN

            null_counts_i = null_counts[:, i]
            valid_null_counts_i = null_counts_i[~np.isnan(null_counts_i)] # Counts should generally not be NaN

            if len(valid_null_counts_i) < 2:
                ses_df.loc[index, f'ses_count_gt{THRESH_LABEL}_{i}'] = np.nan
            else:
                mean_null_c = np.mean(valid_null_counts_i)
                std_null_c = np.std(valid_null_counts_i)
                if std_null_c > 1e-9:
                    ses_c = (obs_count - mean_null_c) / std_null_c
                    ses_df.loc[index, f'ses_count_gt{THRESH_LABEL}_{i}'] = ses_c
                else: # Handle zero std dev
                    ses_df.loc[index, f'ses_count_gt{THRESH_LABEL}_{i}'] = 0.0 if np.isclose(obs_count, mean_null_c) else np.inf * np.sign(obs_count - mean_null_c)

    if grids_missing_null > 0:
        logger.warning(f"Could not calculate SES for {grids_missing_null} grids because corresponding null model results were missing.")
    logger.info("SES calculation finished.")
    # Replace infinite SES values with NaN or a large number? Let's use NaN for plotting.
    ses_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return ses_df


# --- Standard GeoDataFrame Creation ---
def create_geopandas_from_1deg_grid(df, grid_id_col='grid_id'):
    """ Creates GeoDataFrame from DataFrame with grid IDs, parsing coords if needed. """
    logger.info("Creating GeoDataFrame from 1-degree grid data (using exact bounds)...")
    if df.empty: logger.warning("Input DataFrame empty. Creating empty GeoDataFrame."); return gpd.GeoDataFrame([], columns=list(df.columns) + ['geometry'], crs="EPSG:4326")

    work_df = df.copy() # Work on a copy

    # Attempt parsing coordinates if 'grid_lon'/'grid_lat' are missing
    if 'grid_lon' not in work_df.columns or 'grid_lat' not in work_df.columns:
        logger.info("grid_lon/grid_lat not found, attempting to parse from grid_id...");
        try:
            # Regex to extract lon/lat strings like 'n10p5', '10p5', 'n10', '10'
            parsed_coords = work_df[grid_id_col].str.extract(r'lon([npm]?\d+[p]?\d*)_lat([npm]?\d+[p]?\d*)')
            if parsed_coords.isnull().any().any(): raise ValueError("Coordinate parsing resulted in NaNs.")
            def np_to_float(coord_str):
                 if pd.isna(coord_str): return np.nan
                 # Ensure 'p' exists before replacing if parsing integers
                 if 'p' in coord_str:
                     return float(coord_str.replace('n', '-').replace('p', '.'))
                 else:
                     return float(coord_str.replace('n', '-')) # Handle integer coords

            work_df['grid_lon'] = parsed_coords[0].apply(np_to_float); work_df['grid_lat'] = parsed_coords[1].apply(np_to_float)
            if work_df['grid_lon'].isnull().any() or work_df['grid_lat'].isnull().any(): raise ValueError("Coordinate parsing resulted in NaNs after conversion.")
            logger.info("Successfully parsed coordinates from grid_id.")
        except Exception as parse_err: logger.error(f"Failed to parse coordinates from grid_id: {parse_err}."); return gpd.GeoDataFrame([], columns=list(df.columns) + ['geometry'], crs="EPSG:4326")

    # Ensure coords are numeric after potential parsing
    work_df['grid_lon'] = pd.to_numeric(work_df['grid_lon'], errors='coerce')
    work_df['grid_lat'] = pd.to_numeric(work_df['grid_lat'], errors='coerce')
    work_df.dropna(subset=['grid_lon', 'grid_lat'], inplace=True) # Drop rows where parsing failed
    if work_df.empty: logger.error("No valid coordinates found after parsing/conversion."); return gpd.GeoDataFrame([], columns=list(df.columns) + ['geometry'], crs="EPSG:4326")

    # Calculate bounds (assuming 1-degree cells based on naming convention)
    work_df['lon_min'] = work_df['grid_lon']; work_df['lat_min'] = work_df['grid_lat']; work_df['lon_max'] = work_df['grid_lon'] + 1.0; work_df['lat_max'] = work_df['grid_lat'] + 1.0
    required_cols = ['lon_min', 'lat_min', 'lon_max', 'lat_max'];
    if not all(col in work_df.columns for col in required_cols): logger.error(f"DataFrame missing required bound columns: {required_cols}"); return gpd.GeoDataFrame([], columns=list(df.columns) + ['geometry'], crs="EPSG:4326")
    logger.info(f"Generating geometries for {len(work_df)} grid cells...")
    geometries = [box(row.lon_min, row.lat_min, row.lon_max, row.lat_max) for _, row in tqdm(work_df.iterrows(), total=len(work_df), desc="Creating Geometries", unit="cell")]
    gdf = gpd.GeoDataFrame(work_df, geometry=geometries, crs="EPSG:4326")
    logger.info(f"Created GeoDataFrame with {len(gdf)} cells.")
    return gdf

# --- Standard Mapping Function (Adapted for SES) ---
def create_aligned_map(gdf, output_path, value_col_name, map_title_nice="Map", cmap_name=CUSTOM_CMAP_NAME, vmin=None, vmax=None, is_ses_map=False, figure_width=10.0, figure_height=5.0, bg_color="#FFFFFF", land_color="#FFFFFF", ocean_color="#E6ECF5", legend_size=0.03, alpha=1.0, base_legend_title="Value", log_scale=False, coastline_resolution='110m'):
    """ Creates map visualizing specified column, styled consistently. Adapts for SES (diverging cmap, centering). """
    global args # Access global args for projection
    projection_dict = {
        "robinson": ccrs.Robinson(), "mercator": ccrs.Mercator(), "mollweide": ccrs.Mollweide(),
        "platecarree": ccrs.PlateCarree(), "sinusoidal": ccrs.Sinusoidal()
    }
    selected_projection = projection_dict.get(args.projection.lower(), ccrs.Robinson())

    logger.info(f"Generating map '{map_title_nice}' to {output_path}...")
    logger.debug(f"Value column: '{value_col_name}', Cmap: '{cmap_name}', SES map: {is_ses_map}")
    if value_col_name not in gdf.columns: logger.error(f"Value column '{value_col_name}' not found."); return
    if gdf.empty: logger.warning("Input GeoDataFrame is empty. Map will be blank."); return

    # Ensure data is numeric, create a copy for plotting
    gdf_plot = gdf.copy()
    gdf_plot[value_col_name] = pd.to_numeric(gdf_plot[value_col_name], errors='coerce')
    valid_data = gdf_plot[value_col_name].dropna()
    if valid_data.empty : logger.warning(f"No valid numeric data in '{value_col_name}' after dropping NaNs. Map might be blank.")

    try: # Colormap selection
        if cmap_name not in plt.colormaps():
            fallback_cmap = DIVERGING_CMAP_NAME if is_ses_map else CUSTOM_CMAP_BASE
            logger.warning(f"Colormap '{cmap_name}' not found. Using fallback '{fallback_cmap}'.")
            cmap_name = fallback_cmap if fallback_cmap in plt.colormaps() else "viridis"
        cmap = plt.get_cmap(cmap_name)
    except Exception as e: logger.error(f"Error getting colormap '{cmap_name}': {e}. Using 'viridis'."); cmap = plt.get_cmap("viridis")

    # Auto-determine vmin/vmax if not provided, based on *valid* data
    auto_vmin = valid_data.min() if not valid_data.empty else 0
    auto_vmax = valid_data.max() if not valid_data.empty else 1
    # Use provided vmin/vmax unless they are None
    plot_vmin = vmin if vmin is not None else auto_vmin
    plot_vmax = vmax if vmax is not None else auto_vmax

    # Validate and adjust vmin/vmax
    if not (np.isfinite(plot_vmin) and np.isfinite(plot_vmax)):
        logger.warning(f"Non-finite vmin/vmax detected ({plot_vmin}, {plot_vmax}). Using auto range: [{auto_vmin:.3g}, {auto_vmax:.3g}]"); plot_vmin=auto_vmin; plot_vmax=auto_vmax
    elif plot_vmin > plot_vmax:
        logger.warning(f"vmin ({plot_vmin:.3g}) > vmax ({plot_vmax:.3g}). Using auto range: [{auto_vmin:.3g}, {auto_vmax:.3g}]"); plot_vmin = auto_vmin; plot_vmax = auto_vmax
    elif np.isclose(plot_vmin, plot_vmax):
        logger.warning(f"vmin close to vmax ({plot_vmin:.3g}). Expanding range slightly.");
        expand = abs(plot_vmin * 0.1) + 0.1 # Expand slightly
        plot_vmax = plot_vmin + expand; plot_vmin = plot_vmin - expand
        if np.isclose(plot_vmin, plot_vmax): plot_vmin=0; plot_vmax=max(1.0, plot_vmax) # Final fallback

    # Normalization setup
    norm = None
    if is_ses_map:
        # Center the colormap around 0 using the potentially adjusted plot_vmin/vmax
        extreme_abs = max(abs(plot_vmin), abs(plot_vmax)) # Find the largest deviation from 0
        norm = CenteredNorm(vcenter=0, halfrange=extreme_abs if extreme_abs > 1e-9 else 1.0) # Avoid halfrange=0
        # Update plot limits for the colorbar range to be symmetric
        plot_vmin = -extreme_abs if extreme_abs > 1e-9 else -1.0
        plot_vmax = extreme_abs if extreme_abs > 1e-9 else 1.0
        logger.debug(f"Using SES CenteredNorm. Final range: [{plot_vmin:.3g}, {plot_vmax:.3g}]")
    elif log_scale: # Log scale (less likely needed now)
        # Adjust vmin if non-positive for log scale
        if plot_vmin <= 0:
            min_positive = valid_data[valid_data > 0].min() if not valid_data[valid_data > 0].empty else 1e-6
            adjusted_vmin = max(min_positive, 1e-6); logger.warning(f"Adjusting vmin from {plot_vmin:.3g} to {adjusted_vmin:.3g} for log scale."); plot_vmin = adjusted_vmin
        if plot_vmin >= plot_vmax: plot_vmax = plot_vmin * 10 # Ensure vmax > vmin
        try: norm = LogNorm(vmin=plot_vmin, vmax=plot_vmax); logger.debug(f"Using Log scale normalization.")
        except ValueError as e: logger.error(f"Error creating LogNorm (vmin={plot_vmin}, vmax={plot_vmax}): {e}. Falling back linear."); norm = Normalize(vmin=plot_vmin, vmax=plot_vmax); log_scale = False
    else: # Linear scale (default)
         norm = Normalize(vmin=plot_vmin, vmax=plot_vmax); logger.debug(f"Using Linear scale normalization.")

    logger.debug(f"Final plot limits for colormap: [{plot_vmin:.3g}, {plot_vmax:.3g}]")

    # --- Create Figure ---
    fig = plt.figure(figsize=(figure_width, figure_height)); ax = fig.add_subplot(1, 1, 1, projection=selected_projection)
    fig.patch.set_facecolor(bg_color); ax.set_facecolor(bg_color); ax.set_global(); ax.axis('off')
    land = cfeature.LAND.with_scale(coastline_resolution); ocean = cfeature.OCEAN.with_scale(coastline_resolution)
    ax.add_feature(land, edgecolor='none', facecolor=land_color, zorder=0); ax.add_feature(ocean, edgecolor='none', facecolor=ocean_color, zorder=0)
    ax.coastlines(resolution=coastline_resolution, color='black', linewidth=0.2, zorder=2)

    # --- Plot Data ---
    mesh = None
    # Use the GDF with NaNs already handled for plotting check
    gdf_plot_valid = gdf_plot.dropna(subset=[value_col_name])
    if not gdf_plot_valid.empty:
        logger.debug(f"Plotting {len(gdf_plot_valid)} valid grid cells using pcolormesh.")
        try:
            # Define the grid edges (center points are in gdf_plot_valid['grid_lon'/'grid_lat'])
            x_edges = np.arange(-180, 181, 1.0); y_edges = np.arange(-90, 91, 1.0)
            # Create an empty grid to store values
            value_grid = np.full((len(y_edges)-1, len(x_edges)-1), np.nan)

            # Check if coordinate columns exist before indexing
            if 'grid_lon' not in gdf_plot_valid.columns or 'grid_lat' not in gdf_plot_valid.columns:
                raise KeyError("Missing 'grid_lon'/'grid_lat' columns required for pcolormesh indexing.")

            # Calculate indices efficiently
            lon_indices = (gdf_plot_valid['grid_lon'] + 180).astype(int)
            lat_indices = (gdf_plot_valid['grid_lat'] + 90).astype(int)

            # Ensure indices are within bounds
            valid_indices_mask = (lon_indices >= 0) & (lon_indices < value_grid.shape[1]) & \
                                 (lat_indices >= 0) & (lat_indices < value_grid.shape[0])

            # Use numpy indexing for speed
            value_grid[lat_indices[valid_indices_mask], lon_indices[valid_indices_mask]] = gdf_plot_valid.loc[valid_indices_mask, value_col_name]

            mesh = ax.pcolormesh(x_edges, y_edges, value_grid,
                                 cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
                                 alpha=alpha, shading='flat', rasterized=True, zorder=1)
            logger.debug("pcolormesh grid plotted successfully.")
        except Exception as pcm_err: logger.error(f"Error during pcolormesh plotting: {pcm_err}", exc_info=True); mesh = None
    else: logger.info("No valid data to plot for this variable.")

    # --- Add Colorbar ---
    if mesh:
        legend_width = 0.5; legend_left = (1 - legend_width) / 2; legend_bottom = 0.08
        cax = fig.add_axes([legend_left, legend_bottom, legend_width, legend_size])
        cbar = fig.colorbar(mesh, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(size=0, labelsize=8, color='#555555'); cbar.outline.set_visible(False); cbar.set_label(base_legend_title, size=9, color='#333333')
    else: logger.info("No mesh plotted, skipping colorbar.")

    plt.suptitle(map_title_nice, fontsize=12, y=0.92)

    # --- Save Figure ---
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=plt.rcParams['savefig.dpi'], bbox_inches='tight', pad_inches=plt.rcParams['savefig.pad_inches'], facecolor=fig.get_facecolor(), edgecolor='none'); logger.info(f"  Saved map (PNG): {os.path.relpath(output_path)}")
        pdf_output_path = os.path.splitext(output_path)[0] + ".pdf"; plt.savefig(pdf_output_path, dpi=plt.rcParams['savefig.dpi'], bbox_inches='tight', pad_inches=plt.rcParams['savefig.pad_inches'], facecolor=fig.get_facecolor(), edgecolor='none'); logger.info(f"  Saved map (PDF): {os.path.relpath(pdf_output_path)}")
    except Exception as e: logger.error(f"  Failed to save map to {output_path} (and/or PDF): {e}")
    finally: plt.close(fig); gc.collect()


# --- Scatter Plot Function ---
def plot_ses_vs_richness(df, richness_col, ses_col, strategy_label, metric_label, output_path):
    """Creates a scatter plot of SES vs. species richness."""
    logger.info(f"Generating scatter plot: SES ({metric_label}) vs Richness - {strategy_label}")

    plot_df = df[[richness_col, ses_col]].copy()
    plot_df.dropna(inplace=True) # Remove points where either value is NaN

    if plot_df.empty:
        logger.warning(f"No valid data points for scatter plot: {strategy_label} - {metric_label}. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(6, 5)) # Adjust size as needed
    sns.scatterplot(data=plot_df, x=richness_col, y=ses_col, ax=ax, alpha=0.5, s=10, edgecolor=None)

    ax.set_xlabel("Species Richness (Matched)")
    ax.set_ylabel(f"SES ({metric_label})")
    ax.set_title(f"SES ({metric_label}) vs. Richness\n{strategy_label}")
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8) # Add line at SES = 0
    # Optional: Add lines at +/- 1.96 for significance
    ax.axhline(1.96, color='red', linestyle=':', linewidth=0.6)
    ax.axhline(-1.96, color='red', linestyle=':', linewidth=0.6)

    plt.tight_layout()
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=plt.rcParams['savefig.dpi'])
        logger.info(f"  Saved scatter plot: {os.path.relpath(output_path)}")
        pdf_output_path = os.path.splitext(output_path)[0] + ".pdf"
        plt.savefig(pdf_output_path, dpi=plt.rcParams['savefig.dpi'])
        logger.info(f"  Saved scatter plot (PDF): {os.path.relpath(pdf_output_path)}")
    except Exception as e:
        logger.error(f"  Failed to save scatter plot to {output_path}: {e}")
    finally:
        plt.close(fig); gc.collect()


# --- Main Function ---
def main():
    """Main entry point: Load data, calculate observed metrics, run null model, calculate SES, create maps and plots."""

    # --- Register Custom Colormap ---
    if not register_custom_nipy_spectral_darkred(): logger.warning("Custom colormap registration failed."); default_cmap = "nipy_spectral"
    else: default_cmap = CUSTOM_CMAP_NAME

    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description="Compare observed species proportion patterns to a realm-specific null model based on richness.")
    # Core Options
    parser.add_argument("--species-probs-file", type=str, default=DEFAULT_SPECIES_PROBS_FILE, help=f"Path to CSV with species mean probabilities (default: {DEFAULT_SPECIES_PROBS_FILE})")
    parser.add_argument("--valid-cells-richness-file", type=str, default=DEFAULT_VALID_CELLS_FILE, help=f"Path to CSV with 'grid_id's of cells to process (mandatory filter). Default: {DEFAULT_VALID_CELLS_FILE}")
    parser.add_argument("--realm-map-file", type=str, default=DEFAULT_REALM_MAP_FILE, help=f"Path to CSV mapping grid_id to realm_name. Default: {DEFAULT_REALM_MAP_FILE}")
    parser.add_argument("--output-dir", type=str, default="output", help="Base directory for ALL outputs")
    parser.add_argument("--workers", type=int, help="Number of parallel processes (default: auto)")
    parser.add_argument("--num-permutations", type=int, default=999, help="Number of permutations for the null model (default: 99)")
    # Output Control
    parser.add_argument("--save-results-csv", action="store_true", help="Save combined observed metrics and SES results as CSV.")
    parser.add_argument("--results-csv-name", type=str, default="observed_vs_null_realm_results", help="Base name for the combined results CSV output file.")
    parser.add_argument("--map-output-name-base", type=str, default="prop_vs_null_realm_map", help="Base name for map output files")
    parser.add_argument("--plot-output-name-base", type=str, default="ses_vs_richness_scatter", help="Base name for scatter plot output files")
    parser.add_argument("--strategy-labels", nargs='+', default=None, help="Optional: List of labels for strategies (used in map/plot titles/legends)")
    # Map Styling Args
    parser.add_argument("--cmap-observed", type=str, default=default_cmap, help=f"Colormap for observed mean proportion/count maps. Default: {default_cmap}")
    parser.add_argument("--cmap-ses", type=str, default=DIVERGING_CMAP_NAME, help=f"Diverging colormap for SES maps. Default: {DIVERGING_CMAP_NAME}")
    parser.add_argument("--figure-width", type=float, default=10.0, help="Map figure width")
    parser.add_argument("--figure-height", type=float, default=5.0, help="Map figure height")
    parser.add_argument("--bg-color", type=str, default="#FFFFFF", help="Map background color")
    parser.add_argument("--land-color", type=str, default="#FFFFFF", help="Map land color")
    parser.add_argument("--ocean-color", type=str, default="#E6ECF5", help="Map ocean color")
    parser.add_argument("--legend-size", type=float, default=0.03, help="Relative height of the colorbar")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha transparency for map data")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution for saved map/plot files")
    parser.add_argument("--projection", type=str, default="robinson", choices=["robinson", "mercator", "mollweide", "platecarree", "sinusoidal"], help="Map projection.")
    parser.add_argument("--coastline-resolution", type=str, default="110m", choices=["10m", "50m", "110m"], help="Map feature resolution.")

    global args # Make args global for map function
    args = parser.parse_args()

    # --- Setup ---
    output_basedir = os.path.join(args.output_dir, OUTPUT_SUBDIR)
    os.makedirs(output_basedir, exist_ok=True)
    # Define specific output subdirectories
    map_output_dir_observed = os.path.join(output_basedir, "maps_observed")
    map_output_dir_ses = os.path.join(output_basedir, "maps_ses")
    plot_output_dir_scatter = os.path.join(output_basedir, "plots_scatter_ses_richness")
    os.makedirs(map_output_dir_observed, exist_ok=True)
    os.makedirs(map_output_dir_ses, exist_ok=True)
    os.makedirs(plot_output_dir_scatter, exist_ok=True)

    plt.rcParams['savefig.dpi'] = args.dpi
    num_workers = args.workers # Will be handled by functions if None

    # --- 1. Load Data and Apply Initial Filter ---
    try:
        data_manager = SpeciesDataManager(
            species_probs_file=args.species_probs_file,
            valid_cells_file=args.valid_cells_richness_file,
            realm_map_file=args.realm_map_file
        )
        grid_df_filtered, species_proba_dict, valid_species_set, realm_species_pools, n_components = data_manager.load_data()
        if grid_df_filtered.empty:
             logger.error("Initial data loading and filtering resulted in an empty grid DataFrame. Exiting.")
             sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load and prepare initial data: {e}. Exiting.", exc_info=True)
        sys.exit(1)

    # --- Validation (Post Data Load) ---
    if args.strategy_labels and len(args.strategy_labels) != n_components:
         logger.error(f"Number of --strategy-labels ({len(args.strategy_labels)}) must match the number of detected components ({n_components}). Exiting.")
         sys.exit(1)

    # --- 2. Calculate Observed Metrics ---
    try:
        # Pass the already filtered grid_df
        observed_metrics_df = calculate_observed_metrics(grid_df_filtered, data_manager, workers=num_workers)
        if observed_metrics_df.empty or (observed_metrics_df['status'] == 'success').sum() == 0:
             logger.error("No grid cells processed successfully for observed metrics. Cannot proceed.")
             sys.exit(1)
        # Merge realm_name back into observed_metrics_df for the null model runner
        observed_metrics_df = pd.merge(observed_metrics_df, grid_df_filtered[['grid_id', 'realm_name']], on='grid_id', how='left')

    except Exception as e:
        logger.error(f"Failed during observed metrics calculation: {e}", exc_info=True)
        sys.exit(1)

    # --- 3. Run Null Model (Realm Specific) ---
    try:
        null_results_dict = run_null_model_realm(observed_metrics_df, data_manager, args.num_permutations, workers=num_workers)
        if not null_results_dict:
            logger.warning("Realm null model did not produce results for any cells. SES calculation will be skipped/empty.")
            # Allow script to continue to potentially save observed results, but SES maps/plots won't work.
    except Exception as e:
        logger.error(f"Failed during realm null model execution: {e}", exc_info=True)
        sys.exit(1) # Stop if null model crashes

    # --- 4. Calculate SES ---
    final_results_df = pd.DataFrame() # Initialize
    if null_results_dict: # Only calculate SES if null model ran successfully for at least some cells
        try:
            final_results_df = calculate_ses(observed_metrics_df, null_results_dict, n_components)
        except Exception as e:
            logger.error(f"Failed during SES calculation: {e}", exc_info=True)
            # Use observed_metrics_df if SES fails, so observed maps/CSV can still be generated
            final_results_df = observed_metrics_df.copy()
            # Add empty SES columns if they don't exist, to avoid KeyErrors later
            for i in range(n_components):
                 if f'ses_mean_prob_{i}' not in final_results_df.columns: final_results_df[f'ses_mean_prob_{i}'] = np.nan
                 if f'ses_count_gt{THRESH_LABEL}_{i}' not in final_results_df.columns: final_results_df[f'ses_count_gt{THRESH_LABEL}_{i}'] = np.nan
    else:
        logger.warning("Skipping SES calculation as null model results were empty.")
        final_results_df = observed_metrics_df.copy()
        # Add empty SES columns
        for i in range(n_components):
             final_results_df[f'ses_mean_prob_{i}'] = np.nan
             final_results_df[f'ses_count_gt{THRESH_LABEL}_{i}'] = np.nan


    # --- 5. Save Combined Results CSV (Optional) ---
    if args.save_results_csv:
        results_csv_path = os.path.join(output_basedir, f"{args.results_csv_name}.csv")
        logger.info(f"Saving combined results to {os.path.relpath(results_csv_path)}")
        try:
            # Define columns to save
            cols_to_save = ['grid_id', 'realm_name', 'num_species', 'status'] + \
                           [f'mean_prob_{i}' for i in range(n_components)] + \
                           [f'count_gt{THRESH_LABEL}_{i}' for i in range(n_components)] + \
                           [f'ses_mean_prob_{i}' for i in range(n_components)] + \
                           [f'ses_count_gt{THRESH_LABEL}_{i}' for i in range(n_components)]
            # Filter for columns that actually exist in the final dataframe
            cols_exist = [col for col in cols_to_save if col in final_results_df.columns]
            final_results_df[cols_exist].to_csv(results_csv_path, index=False, float_format='%.6f')
        except Exception as e:
            logger.error(f"Failed to save results CSV: {e}", exc_info=True)

    # --- 6. Create GeoDataFrame for Mapping ---
    logger.info("Creating GeoDataFrame for mapping...")
    try:
        # Use the final results dataframe which includes observed and potentially SES metrics
        merged_gdf = create_geopandas_from_1deg_grid(final_results_df.copy(), grid_id_col='grid_id')
        if merged_gdf.empty:
            logger.error("Creation of GeoDataFrame resulted in empty object. Cannot create maps/plots.")
            sys.exit(1) # Stop if geodataframe creation fails
    except Exception as e:
        logger.error(f"Failed to create GeoDataFrame: {e}", exc_info=True)
        sys.exit(1)

    # --- 7. Create Maps and Plots ---
    logger.info("Generating maps and scatter plots...")
    current_strategy_labels = args.strategy_labels if args.strategy_labels else [f"Strategy {i}" for i in range(n_components)]

    for i in range(n_components):
        strategy_label_i = current_strategy_labels[i]
        gc.collect() # Collect garbage between strategy loops

        # --- Observed Maps ---
        map_output_base_obs = os.path.join(map_output_dir_observed, args.map_output_name_base)
        mean_prob_col = f'mean_prob_{i}'
        count_col = f'count_gt{THRESH_LABEL}_{i}'

        create_aligned_map( gdf=merged_gdf, output_path=f"{map_output_base_obs}_obs_mean_prob_{i}.png",
            value_col_name=mean_prob_col, map_title_nice=f"Observed Mean Proportion - {strategy_label_i}",
            cmap_name=args.cmap_observed, vmin=0.0, vmax=1.0, base_legend_title=f"Mean Proportion", is_ses_map=False,
            figure_width=args.figure_width, figure_height=args.figure_height, bg_color=args.bg_color, land_color=args.land_color, ocean_color=args.ocean_color, legend_size=args.legend_size, alpha=args.alpha, coastline_resolution=args.coastline_resolution )

        create_aligned_map( gdf=merged_gdf, output_path=f"{map_output_base_obs}_obs_count_gt{THRESH_LABEL}_{i}.png",
            value_col_name=count_col, map_title_nice=f"Observed Count (Prob > {THRESH_LABEL}%) - {strategy_label_i}",
            cmap_name=args.cmap_observed, vmin=0, vmax=None, base_legend_title=f"# Species (Prob > {THRESH_LABEL}%)", is_ses_map=False, log_scale=False,
            figure_width=args.figure_width, figure_height=args.figure_height, bg_color=args.bg_color, land_color=args.land_color, ocean_color=args.ocean_color, legend_size=args.legend_size, alpha=args.alpha, coastline_resolution=args.coastline_resolution )

        # --- SES Maps ---
        map_output_base_ses = os.path.join(map_output_dir_ses, args.map_output_name_base)
        ses_mean_col = f'ses_mean_prob_{i}'
        ses_count_col = f'ses_count_gt{THRESH_LABEL}_{i}'

        if ses_mean_col in merged_gdf.columns: # Check if SES columns exist
             create_aligned_map( gdf=merged_gdf, output_path=f"{map_output_base_ses}_ses_mean_prob_{i}.png",
                 value_col_name=ses_mean_col, map_title_nice=f"SES Mean Proportion - {strategy_label_i}",
                 cmap_name=args.cmap_ses, vmin=None, vmax=None, base_legend_title=f"SES (Mean Proportion)", is_ses_map=True,
                 figure_width=args.figure_width, figure_height=args.figure_height, bg_color=args.bg_color, land_color=args.land_color, ocean_color=args.ocean_color, legend_size=args.legend_size, alpha=args.alpha, coastline_resolution=args.coastline_resolution )
        else: logger.warning(f"SES column {ses_mean_col} not found, skipping map.")

        if ses_count_col in merged_gdf.columns:
             create_aligned_map( gdf=merged_gdf, output_path=f"{map_output_base_ses}_ses_count_gt{THRESH_LABEL}_{i}.png",
                 value_col_name=ses_count_col, map_title_nice=f"SES Count (Prob > {THRESH_LABEL}%) - {strategy_label_i}",
                 cmap_name=args.cmap_ses, vmin=None, vmax=None, base_legend_title=f"SES (# Species > {THRESH_LABEL}%)", is_ses_map=True,
                 figure_width=args.figure_width, figure_height=args.figure_height, bg_color=args.bg_color, land_color=args.land_color, ocean_color=args.ocean_color, legend_size=args.legend_size, alpha=args.alpha, coastline_resolution=args.coastline_resolution )
        else: logger.warning(f"SES column {ses_count_col} not found, skipping map.")


        # --- Scatter Plots (SES vs Richness) ---
        scatter_output_base = os.path.join(plot_output_dir_scatter, args.plot_output_name_base)
        richness_col = 'num_species' # This column stores the *matched* richness

        if ses_mean_col in merged_gdf.columns and richness_col in merged_gdf.columns:
            plot_ses_vs_richness( df=merged_gdf, richness_col=richness_col, ses_col=ses_mean_col,
                                  strategy_label=strategy_label_i, metric_label="Mean Proportion",
                                  output_path=f"{scatter_output_base}_mean_prob_{i}.png")
        else: logger.warning(f"Columns for scatter plot {ses_mean_col} vs {richness_col} missing. Skipping.")

        if ses_count_col in merged_gdf.columns and richness_col in merged_gdf.columns:
             plot_ses_vs_richness( df=merged_gdf, richness_col=richness_col, ses_col=ses_count_col,
                                   strategy_label=strategy_label_i, metric_label=f"Count > {THRESH_LABEL}%",
                                   output_path=f"{scatter_output_base}_count_gt{THRESH_LABEL}_{i}.png")
        else: logger.warning(f"Columns for scatter plot {ses_count_col} vs {richness_col} missing. Skipping.")


    logger.info("\nScript finished.")


if __name__ == "__main__":
    script_start_time = time.time()
    main()
    script_end_time = time.time()
    logger.info(f"Total script execution time: {script_end_time - script_start_time:.2f} seconds")