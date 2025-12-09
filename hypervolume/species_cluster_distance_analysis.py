#!/usr/bin/env python3
"""
Species Cluster Distance Analysis

This script analyzes species clusters across four different dimensional spaces:
1. Raw data space (using original MPS features)
2. PC space (using PC1-PC37)
3. UMAP 3D space (using UMAP3D 1, UMAP3D 2, and UMAP3D 3)
4. UMAP 2D space (using UMAP 1 and UMAP 2)

For each space, it computes:
- Species centroids
- Pairwise distances between species centroids
- Co-ranking matrices comparing the rank preservation across spaces
- Area Under the Curve (AUC) metrics to quantify the quality of dimensionality reduction
- Mean Absolute Deviation (MAD) to detect extreme distortions

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import argparse
from tqdm import tqdm
import os
from itertools import combinations, product
from scipy.spatial.distance import pdist, squareform, euclidean
import warnings
warnings.filterwarnings('ignore')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Analyze species cluster distances in different dimensional spaces')
parser.add_argument('--traits_file', default='data/traits_data.csv', 
                    help='CSV file with traits data')
parser.add_argument('--output_dir', default='output/species_distances', 
                    help='Directory to save the output files')
parser.add_argument('--min_samples', type=int, default=5,
                    help='Minimum number of samples per species to include in analysis')
parser.add_argument('--random_state', type=int, default=42,
                    help='Random state for reproducibility')
parser.add_argument('--mps_data', default='data/X_500ms.npy',
                    help='NumPy file containing the original MPS data')
parser.add_argument('--metadata', default='data/metadata_500ms.npz',
                    help='NumPy file containing metadata for MPS data')
parser.add_argument('--expected_length', type=int, default=9898,
                    help='Expected length of MPS samples to keep')
parser.add_argument('--sample_dims', type=int, default=500,
                    help='Number of random dimensions to sample from raw data (if too large)')
parser.add_argument('--debug', action='store_true',
                    help='Print debug information')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Set plot style
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.title_fontsize': 16
})

def calculate_centroids(data, feature_cols, species_col='species'):
    """
    Calculate species centroids in the given feature space
    """
    # Group by species and calculate centroids
    species_centroids = data.groupby(species_col)[feature_cols].mean().reset_index()
    return species_centroids

def create_distance_matrix(centroids, feature_cols):
    """
    Create a distance matrix between all species centroids
    """
    # Extract centroid coordinates
    centroid_coordinates = centroids[feature_cols].values
    
    # Calculate pairwise distances
    distances = pdist(centroid_coordinates, metric='euclidean')
    
    # Convert to square matrix
    distance_matrix = squareform(distances)
    
    return distance_matrix

def calculate_ranks(distance_matrix):
    """
    Calculate the ranks of distances for each row in the distance matrix
    """
    n = distance_matrix.shape[0]
    rank_matrix = np.zeros_like(distance_matrix)
    
    # For each species, rank its distances to all other species
    for i in range(n):
        # Get the row, ignoring the diagonal (distance to self)
        row = distance_matrix[i, :].copy()
        row[i] = np.nan  # Ignore self-distance
        
        # Skip if all values are NaN or only one non-NaN value
        if np.isnan(row).all() or (~np.isnan(row)).sum() <= 1:
            continue
        
        # Compute ranks (add 1 to get 1-indexed ranks)
        ranks = np.zeros_like(row)
        valid_indices = ~np.isnan(row)
        ranks[valid_indices] = np.argsort(np.argsort(row[valid_indices])) + 1
        rank_matrix[i, :] = ranks
    
    return rank_matrix

def create_coranking_matrix(rank_matrix_high, rank_matrix_low):
    """
    Create a co-ranking matrix Q comparing high-dimensional and low-dimensional ranks
    """
    n = rank_matrix_high.shape[0]
    max_rank = n - 1  # Maximum possible rank
    
    # Initialize co-ranking matrix
    Q = np.zeros((max_rank, max_rank), dtype=int)
    
    # Fill co-ranking matrix by counting rank pairs
    for i in range(n):
        for j in range(n):
            if i != j:  # Exclude self-comparisons
                high_rank = int(rank_matrix_high[i, j])
                low_rank = int(rank_matrix_low[i, j])
                
                # Only consider valid ranks (greater than 0)
                if high_rank > 0 and low_rank > 0:
                    k = high_rank - 1  # Convert to 0-indexed
                    l = low_rank - 1   # Convert to 0-indexed
                    
                    # Ensure indices are within bounds
                    if k < max_rank and l < max_rank:
                        Q[k, l] += 1
    
    return Q

def calculate_auc(Q):
    """
    Calculate the Area Under the Curve (AUC) metric from the co-ranking matrix
    
    AUC is based on Somer's D statistic as an asymmetric rank measure.
    - AUC = 1: Perfect preservation of rankings
    - AUC > 0.8: Excellent dimensionality reduction
    - AUC > 0.7: Good/acceptable dimensionality reduction
    - AUC < 0.5: Poor representation
    - AUC = 0: Random/no preservation
    """
    n = Q.shape[0]
    total_pairs = np.sum(Q)
    
    # Return NaN if the matrix is empty
    if total_pairs == 0:
        print("Warning: Co-ranking matrix is empty. Returning NaN for AUC.")
        return np.nan
    
    # Calculate concordant, discordant, and tied pairs
    concordant = 0
    discordant = 0
    
    for i in range(n):
        for j in range(n):
            if i < j:  # Upper triangle (concordant)
                concordant += Q[i, j]
            elif i > j:  # Lower triangle (discordant)
                discordant += Q[i, j]
            # Diagonal elements are tied
    
    # Avoid division by zero
    denominator = concordant + discordant
    if denominator == 0:
        print("Warning: No concordant or discordant pairs found. Returning NaN for AUC.")
        return np.nan
    
    # Calculate AUC
    auc = concordant / denominator
    
    return auc

def calculate_mad(dist_matrix_high, dist_matrix_low):
    """
    Calculate Mean Absolute Deviation (MAD) between distance matrices
    
    This captures extreme distortions that might not be reflected in rank-based measures.
    """
    # Check if matrices have valid values
    if np.all(np.isnan(dist_matrix_high)) or np.all(np.isnan(dist_matrix_low)):
        print("Warning: Distance matrix contains all NaN values. Returning NaN for MAD.")
        return np.nan
    
    # Normalize both matrices to [0, 1] range for fair comparison
    high_min = np.nanmin(dist_matrix_high)
    high_max = np.nanmax(dist_matrix_high)
    low_min = np.nanmin(dist_matrix_low)
    low_max = np.nanmax(dist_matrix_low)
    
    # Check for zero range
    if high_max == high_min or low_max == low_min:
        print("Warning: Distance matrix has zero range. Returning NaN for MAD.")
        return np.nan
    
    dist_matrix_high_norm = (dist_matrix_high - high_min) / (high_max - high_min)
    dist_matrix_low_norm = (dist_matrix_low - low_min) / (low_max - low_min)
    
    # Calculate absolute differences
    abs_diff = np.abs(dist_matrix_high_norm - dist_matrix_low_norm)
    
    # Return mean of absolute differences (excluding diagonal)
    n = dist_matrix_high.shape[0]
    mask = ~np.eye(n, dtype=bool)  # Mask to exclude diagonal
    mad = np.nanmean(abs_diff[mask])
    
    return mad

def plot_coranking_matrix(Q, high_dim_name, low_dim_name, output_dir):
    """
    Plot the co-ranking matrix as a heatmap
    """
    # Check if the matrix is empty
    if np.sum(Q) == 0:
        print(f"Warning: Co-ranking matrix for {high_dim_name} vs {low_dim_name} is empty. Skipping plot.")
        return
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.log1p(Q), cmap='viridis', xticklabels=100, yticklabels=100)
    plt.title(f'Co-ranking Matrix: {high_dim_name} vs {low_dim_name}')
    plt.xlabel(f'{low_dim_name} Ranks')
    plt.ylabel(f'{high_dim_name} Ranks')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'coranking_{high_dim_name.replace(" ", "_")}_{low_dim_name.replace(" ", "_")}.png'), dpi=300)
    plt.close()

def plot_all_results(results_df, output_dir):
    """
    Create visualizations for AUC and MAD results
    """
    # Remove rows with NaN values
    results_df_clean = results_df.dropna(subset=['AUC', 'MAD']).copy()
    
    # Check if there is any data left
    if len(results_df_clean) == 0:
        print("Warning: No valid results to plot. Skipping visualizations.")
        return
    
    # Create a figure for AUC comparison
    plt.figure(figsize=(12, 6))
    
    # Plot AUC values
    ax = sns.barplot(x='Comparison', y='AUC', data=results_df_clean)
    
    # Add reference lines for AUC interpretation
    plt.axhline(y=0.8, color='green', linestyle='--', label='Excellent (0.8)')
    plt.axhline(y=0.7, color='blue', linestyle='--', label='Good (0.7)')
    plt.axhline(y=0.5, color='red', linestyle='--', label='Poor (0.5)')
    
    # Customize plot
    plt.title('Quality of Dimensionality Reduction (AUC)')
    plt.ylabel('AUC Score')
    plt.ylim(0, 1.05)
    plt.legend()
    
    # Add value labels
    for i, v in enumerate(results_df_clean['AUC']):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auc_comparison.png'), dpi=300)
    plt.close()
    
    # Create a figure for MAD comparison
    plt.figure(figsize=(12, 6))
    
    # Plot MAD values
    ax = sns.barplot(x='Comparison', y='MAD', data=results_df_clean)
    
    # Customize plot
    plt.title('Distance Distortion (Mean Absolute Deviation)')
    plt.ylabel('MAD Score')
    plt.ylim(0, max(results_df_clean['MAD']) * 1.2)
    
    # Add value labels
    for i, v in enumerate(results_df_clean['MAD']):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mad_comparison.png'), dpi=300)
    plt.close()

def main():
    # Load traits data
    print(f"Loading traits data from {args.traits_file}...")
    df = pd.read_csv(args.traits_file)
    
    # DEBUG info for traits data
    if args.debug or True:
        print("\n===== DEBUG: Species from traits data =====")
        print("First 5 species:", df['species'].iloc[:5].tolist())
        print("Random 5 species:", df['species'].sample(5, random_state=args.random_state).tolist())
        print("Total unique species in traits data:", df['species'].nunique())
        
        # Print the format of the species column
        print("\nFormat of species in traits data:")
        sample_species = df['species'].iloc[0]
        print(f"Example: '{sample_species}' - Type: {type(sample_species)}")
        
        # Check for whitespace or special characters
        if ' ' in sample_species:
            print("Warning: Species contains spaces")
        if '\t' in sample_species:
            print("Warning: Species contains tabs")
        if '\n' in sample_species:
            print("Warning: Species contains newlines")
    
    # Try to load raw MPS data and metadata
    has_raw_data = False
    matching_species = []
    
    try:
        print(f"Loading raw MPS data from {args.mps_data} and metadata from {args.metadata}...")
        MPS = np.load(args.mps_data, allow_pickle=True)
        metadata = np.load(args.metadata, allow_pickle=True)
        
        # Check if 'species' key exists in metadata
        if 'species' not in metadata:
            raise KeyError("Metadata does not contain 'species' information")
        
        # DEBUG info for raw data
        if args.debug or True:
            print("\n===== DEBUG: Species from raw data =====")
            species_array = metadata['species']
            print("First 5 species:", species_array[:5].tolist() if len(species_array) >= 5 else species_array.tolist())
            
            # Get random indices
            if len(species_array) >= 5:
                np.random.seed(args.random_state)
                random_indices = np.random.choice(len(species_array), 5, replace=False)
                print("Random 5 species:", species_array[random_indices].tolist())
            
            print("Total unique species in raw data:", len(np.unique(species_array)))
            
            # Print the format of the species array
            print("\nFormat of species in raw data:")
            sample_raw_species = species_array[0]
            print(f"Example: '{sample_raw_species}' - Type: {type(sample_raw_species)}")
            
            # Check for whitespace or special characters
            if isinstance(sample_raw_species, str):
                if ' ' in sample_raw_species:
                    print("Warning: Species contains spaces")
                if '\t' in sample_raw_species:
                    print("Warning: Species contains tabs")
                if '\n' in sample_raw_species:
                    print("Warning: Species contains newlines")
            
            # Check for potential transformations that might help match
            print("\nPotential transformations:")
            
            # Check traits data format
            traits_first_species = df['species'].iloc[0]
            if '_' in traits_first_species:
                print(f"Traits data uses underscores, example: {traits_first_species}")
                print(f"Without underscores: {traits_first_species.replace('_', ' ')}")
            
            # Check raw data format if it's a string
            if isinstance(sample_raw_species, str):
                if ' ' in sample_raw_species:
                    print(f"Raw data uses spaces, example: {sample_raw_species}")
                    print(f"With underscores: {sample_raw_species.replace(' ', '_')}")
        
        # Filter MPS data by expected length
        valid_indices = []
        filtered_mps = []
        
        print("Filtering MPS data by expected length...")
        for i, mps in enumerate(tqdm(MPS)):
            if len(mps) == args.expected_length:
                filtered_mps.append(mps)
                valid_indices.append(i)
        
        print(f"Removed {len(MPS) - len(filtered_mps)} outlier(s) out of {len(MPS)}")
        
        # If raw data is very high-dimensional, randomly sample dimensions
        filtered_mps = np.array(filtered_mps)
        raw_feature_cols = []
        
        if filtered_mps.shape[1] > args.sample_dims:
            print(f"Raw data has {filtered_mps.shape[1]} dimensions. Randomly sampling {args.sample_dims} dimensions...")
            np.random.seed(args.random_state)
            selected_dims = np.random.choice(filtered_mps.shape[1], size=args.sample_dims, replace=False)
            selected_dims.sort()  # Sort for easier interpretation
            filtered_mps = filtered_mps[:, selected_dims]
            raw_feature_cols = [f'raw_{i}' for i in selected_dims]
            print(f"Sampled raw data shape: {filtered_mps.shape}")
        else:
            raw_feature_cols = [f'raw_{i}' for i in range(filtered_mps.shape[1])]
        
        # Get species array
        species_array = metadata['species'][valid_indices]
        
        # Convert raw species to match traits data format (spaces to underscores)
        transformed_raw_species = [s.replace(' ', '_') if isinstance(s, str) else s for s in species_array]
        
        # Create raw dataframe
        raw_df = pd.DataFrame(filtered_mps, columns=raw_feature_cols)
        raw_df['species'] = transformed_raw_species  # Use the transformed species names directly
        
        # Find common species between transformed raw data and traits data
        raw_species_set = set(np.unique(transformed_raw_species))
        all_species_set = set(df['species'].unique())
        matching_species = list(raw_species_set.intersection(all_species_set))
        
        print(f"\nFound {len(matching_species)} matching species between raw data and traits data after transformation")
        
        if len(matching_species) > 0:
            print("Example matching species:", matching_species[:5])
            
            # Filter traits data to include only matching species
            df_filtered = df[df['species'].isin(matching_species)].copy()
            
            # Apply min_samples filter to filtered traits data
            species_counts = df_filtered['species'].value_counts()
            valid_species = species_counts[species_counts >= args.min_samples].index
            df_filtered = df_filtered[df_filtered['species'].isin(valid_species)].copy()
            
            print(f"After filtering for minimum samples, traits data includes {len(valid_species)} species")
            
            # Filter raw data to include only species with sufficient samples
            raw_df_filtered = raw_df[raw_df['species'].isin(valid_species)].copy()
            
            print(f"Raw data analysis will include {len(raw_df_filtered)} samples across {len(raw_feature_cols)} dimensions")
            print(f"Number of species in filtered raw data: {len(raw_df_filtered['species'].unique())}")
            
            # Check if we have enough data
            if len(raw_df_filtered) > 0 and len(raw_df_filtered['species'].unique()) > 1:
                has_raw_data = True
                matching_species = list(valid_species)
            else:
                print("Not enough raw data samples or species for analysis. Skipping raw data comparison.")
        else:
            print("No matching species after transformation. Skipping raw data comparison.")
    except Exception as e:
        print(f"Error processing raw data: {e}")
        print("Continuing analysis without raw data comparison")
    
    # Filter traits data to include only matching species if we have raw data
    if has_raw_data and len(matching_species) > 0:
        df_filtered = df[df['species'].isin(matching_species)].copy()
        print(f"Filtered traits data to include only {len(matching_species)} species that match with raw data")
    else:
        # Filter out species with too few samples (default behavior)
        species_counts = df['species'].value_counts()
        valid_species = species_counts[species_counts >= args.min_samples].index
        df_filtered = df[df['species'].isin(valid_species)].copy()
        print(f"Analysis will include {len(valid_species)} species with at least {args.min_samples} samples each")
    
    # Define feature columns for each space
    pc_cols = [col for col in df.columns if col.startswith('PC')]
    umap_2d_cols = ['UMAP 1', 'UMAP 2']
    umap_3d_cols = ['UMAP3D 1', 'UMAP3D 2', 'UMAP3D 3']
    
    # Initialize containers for results
    space_names = ['PC Space', 'UMAP 3D', 'UMAP 2D']
    centroids_dict = {}
    distance_matrices = {}
    rank_matrices = {}
    
    # Add raw data to spaces if available
    if has_raw_data:
        space_names = ['Raw Data'] + space_names
        # Calculate raw data centroids
        raw_centroids = calculate_centroids(raw_df_filtered, raw_feature_cols)
        centroids_dict['Raw Data'] = raw_centroids
    
    # Calculate centroids for each space from traits data
    print("\nCalculating species centroids for each space...")
    
    pc_centroids = calculate_centroids(df_filtered, pc_cols)
    centroids_dict['PC Space'] = pc_centroids
    
    umap_3d_centroids = calculate_centroids(df_filtered, umap_3d_cols)
    centroids_dict['UMAP 3D'] = umap_3d_centroids
    
    umap_2d_centroids = calculate_centroids(df_filtered, umap_2d_cols)
    centroids_dict['UMAP 2D'] = umap_2d_centroids
    
    # Verify all centroids have the same species
    if args.debug or True:
        print("\n===== DEBUG: Species in centroids =====")
        for space in space_names:
            species_count = len(centroids_dict[space])
            print(f"{space}: {species_count} species")
    
    # Calculate distance matrices for each space
    print("Calculating distance matrices...")
    
    for space in space_names:
        try:
            if space == 'Raw Data' and has_raw_data:
                distance_matrices[space] = create_distance_matrix(centroids_dict[space], raw_feature_cols)
            elif space == 'PC Space':
                distance_matrices[space] = create_distance_matrix(centroids_dict[space], pc_cols)
            elif space == 'UMAP 3D':
                distance_matrices[space] = create_distance_matrix(centroids_dict[space], umap_3d_cols)
            elif space == 'UMAP 2D':
                distance_matrices[space] = create_distance_matrix(centroids_dict[space], umap_2d_cols)
        except Exception as e:
            print(f"Error calculating distance matrix for {space}: {e}")
            print(f"Skipping {space} in further analysis")
            if space in space_names:
                space_names.remove(space)
    
    # Calculate ranks for each distance matrix
    print("Calculating rank matrices...")
    
    for space in space_names:
        try:
            if space in distance_matrices:
                rank_matrices[space] = calculate_ranks(distance_matrices[space])
            else:
                print(f"No distance matrix for {space}. Skipping rank calculation.")
        except Exception as e:
            print(f"Error calculating rank matrix for {space}: {e}")
    
    # Create co-ranking matrices and calculate AUC & MAD
    print("Creating co-ranking matrices and calculating metrics...")
    
    results = []
    
    # Use the highest-dimensional space as reference
    reference_space = 'Raw Data' if has_raw_data and 'Raw Data' in space_names else 'PC Space'
    target_spaces = [s for s in space_names if s != reference_space]
    
    if not reference_space in rank_matrices:
        print(f"No rank matrix for reference space {reference_space}. Cannot calculate metrics.")
    else:
        for target_space in target_spaces:
            if not target_space in rank_matrices:
                print(f"No rank matrix for target space {target_space}. Skipping.")
                continue
                
            try:
                # Create co-ranking matrix
                Q = create_coranking_matrix(rank_matrices[reference_space], rank_matrices[target_space])
                
                # Calculate AUC
                auc = calculate_auc(Q)
                
                # Calculate MAD
                mad = calculate_mad(distance_matrices[reference_space], distance_matrices[target_space])
                
                # Determine quality rating
                if np.isnan(auc):
                    quality = 'N/A'
                else:
                    quality = 'Excellent' if auc >= 0.8 else ('Good' if auc >= 0.7 else ('Poor' if auc < 0.5 else 'Acceptable'))
                
                # Add results
                results.append({
                    'Comparison': f'{reference_space} → {target_space}',
                    'Reference': reference_space,
                    'Target': target_space,
                    'AUC': auc,
                    'MAD': mad,
                    'Quality': quality
                })
                
                # Plot co-ranking matrix
                plot_coranking_matrix(Q, reference_space, target_space, args.output_dir)
            except Exception as e:
                print(f"Error processing {reference_space} → {target_space}: {e}")
        
        # Also compare UMAP 3D to UMAP 2D if both are available
        if 'UMAP 3D' in rank_matrices and 'UMAP 2D' in rank_matrices:
            try:
                Q_umap = create_coranking_matrix(rank_matrices['UMAP 3D'], rank_matrices['UMAP 2D'])
                auc_umap = calculate_auc(Q_umap)
                mad_umap = calculate_mad(distance_matrices['UMAP 3D'], distance_matrices['UMAP 2D'])
                
                quality_umap = 'Excellent' if auc_umap >= 0.8 else ('Good' if auc_umap >= 0.7 else ('Poor' if auc_umap < 0.5 else 'Acceptable'))
                if np.isnan(auc_umap):
                    quality_umap = 'N/A'
                
                results.append({
                    'Comparison': 'UMAP 3D → UMAP 2D',
                    'Reference': 'UMAP 3D',
                    'Target': 'UMAP 2D',
                    'AUC': auc_umap,
                    'MAD': mad_umap,
                    'Quality': quality_umap
                })
                
                plot_coranking_matrix(Q_umap, 'UMAP 3D', 'UMAP 2D', args.output_dir)
            except Exception as e:
                print(f"Error comparing UMAP 3D to UMAP 2D: {e}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save results
    if len(results_df) > 0:
        results_df.to_csv(os.path.join(args.output_dir, 'dimensionality_reduction_quality.csv'), index=False)
        print(f"Results saved to {os.path.join(args.output_dir, 'dimensionality_reduction_quality.csv')}")
        
        # Create visualizations
        print("Creating visualizations...")
        plot_all_results(results_df, args.output_dir)
        
        # Print summary
        print("\nSummary of Results:")
        print(results_df.to_string(index=False))
    else:
        print("No results were generated. Check for errors above.")
    
    # Add interpretation guidance
    print("\nInterpretation Guide:")
    print("- AUC = 1.0: Perfect preservation of species relationships")
    print("- AUC ≥ 0.8: Excellent dimensionality reduction")
    print("- AUC ≥ 0.7: Good/acceptable dimensionality reduction")
    print("- AUC < 0.5: Poor representation (below 0.5 is as good as random)")
    print("- Lower MAD values indicate less distortion in distances")

if __name__ == "__main__":
    main() 