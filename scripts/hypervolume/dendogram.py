#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from scipy.spatial.distance import pdist, squareform, cdist
import matplotlib.lines as mlines
import argparse
import os
import matplotlib.collections as mcoll
import seaborn as sns
import time
from tqdm import tqdm
import re
from matplotlib import rcParams

# Set up matplotlib for high quality output
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['svg.fonttype'] = 'none'

# Define cluster mapping
cluster_mapping = {
    0: ("FW", "Flat Whistles"),
    5: ("SMW", "Slow Modulated Whistles"),
    1: ("ST", "Slow Trills"),
    6: ("FMW", "Fast Modulated Whistles"),
    3: ("NS", "Noisy Songs"),
    2: ("FT", "Fast Trills"),
    4: ("UT", "Ultrafast Trills"),
    7: ("HS", "Harmonic Stacks")
}

# Define custom colors to match the actual leaf labels with a more harmonious palette
custom_colors = {
    "FW": "#1f77b4",    # Flat Whistles - blue
    "SMW": "#ff7f0e",   # Slow Modulated Whistles - orange
    "ST": "#2ca02c",    # Slow Trills - green
    "FMW": "#d62728",   # Fast Modulated Whistles - red
    "NS": "#9467bd",    # Noisy Songs - purple
    "FT": "#8c564b",    # Fast Trills - brown
    "UT": "#e377c2",    # Ultrafast Trills - pink
    "HS": "#7f7f7f"     # Harmonic Stacks - gray
}

def compute_intercluster_distances(df, pc_cols, clusters):
    """
    Compute distance matrix between clusters using all data points
    rather than just centroids - optimized version.
    """
    print("Computing intercluster distances...")
    start_time = time.time()
    
    n_clusters = len(clusters)
    distance_matrix = np.zeros((n_clusters, n_clusters))
    
    # Pre-extract all cluster points to avoid repeated filtering
    cluster_points = {}
    for cluster in clusters:
        cluster_points[cluster] = df[df['gmm_cluster'] == cluster][pc_cols].values
        print(f"  Cluster {cluster}: {len(cluster_points[cluster])} points")
    
    # Use vectorized operations with cdist instead of nested loops
    for i, cluster1 in enumerate(tqdm(clusters, desc="Processing clusters")):
        points1 = cluster_points[cluster1]
        
        for j, cluster2 in enumerate(clusters):
            if i == j:
                # Set diagonal to 0 (distance from cluster to itself)
                distance_matrix[i, j] = 0
                continue
                
            points2 = cluster_points[cluster2]
            
            # Use cdist for fast pairwise distance computation
            if len(points1) > 0 and len(points2) > 0:
                pairwise_distances = cdist(points1, points2)
                # Store the distance in both (i,j) and (j,i) positions to ensure symmetry
                avg_distance = np.mean(pairwise_distances)
                distance_matrix[i, j] = avg_distance
                distance_matrix[j, i] = avg_distance  # Ensure symmetry
    
    # Verify symmetry
    is_symmetric = np.allclose(distance_matrix, distance_matrix.T)
    if not is_symmetric:
        print("WARNING: Distance matrix is not symmetric, forcing symmetry...")
        # Force symmetry by averaging
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
    else:
        print("Distance matrix is symmetric ✓")
    
    elapsed = time.time() - start_time
    print(f"Distance computation completed in {elapsed:.2f} seconds")
    return distance_matrix

def set_link_colors(linkage_matrix, labels, color_map):
    """Sets the colors for the dendrogram links."""
    print("Setting link colors...")
    link_colors = {}
    n_leaves = len(labels)
    for i, row in enumerate(linkage_matrix):
        idx1, idx2 = int(row[0]), int(row[1])
        color1 = color_map.get(labels[idx1]) if idx1 < n_leaves else link_colors.get(idx1 - n_leaves)
        color2 = color_map.get(labels[idx2]) if idx2 < n_leaves else link_colors.get(idx2 - n_leaves)
        link_colors[i] = color1 if color1 == color2 else "#000000"
    return link_colors

def plot_distance_heatmap(distance_matrix, labels, output_prefix):
    """Plot a heatmap of the distance matrix between clusters"""
    print(f"Plotting distance heatmap...")
    plt.figure(figsize=(12, 10))
    mask = np.zeros_like(distance_matrix, dtype=bool)
    np.fill_diagonal(mask, True)  # Mask the diagonal for better visualization
    
    # Use a better colormap and adjust aesthetics
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    ax = sns.heatmap(
        distance_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap=cmap,
        xticklabels=[f"{label} ({cluster_mapping[i][1]})" for i, label in enumerate(labels)],
        yticklabels=[f"{label} ({cluster_mapping[i][1]})" for i, label in enumerate(labels)],
        mask=mask,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Distance"},
        annot_kws={"size": 10}
    )
    
    # Improve heatmap appearance
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    
    plt.title("Inter-cluster Distances", fontsize=18, pad=20)
    plt.tight_layout()
    
    # Save as PDF for better quality
    pdf_path = f"{output_prefix}_distance_heatmap.pdf"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    
    # Also save as PNG
    png_path = f"{output_prefix}_distance_heatmap.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    
    print(f"Heatmap saved as PDF: {pdf_path}")
    print(f"Heatmap saved as PNG: {png_path}")

def sort_pc_columns(columns):
    """Sort PC columns numerically (PC1, PC2, ..., PC10) instead of alphabetically"""
    pc_columns = [col for col in columns if col.startswith('PC')]
    
    def pc_key(col):
        match = re.match(r'PC(\d+)', col)
        if match:
            return int(match.group(1))
        return 0
    
    return sorted(pc_columns, key=pc_key)

def create_dendrogram(Z, leaf_labels, link_colors, title, output_file, cluster_mapping):
    """Create and save a dendrogram with improved aesthetics"""
    plt.figure(figsize=(10, 12), dpi=300)
    
    # Use white background for better readability
    plt.style.use('default')
    
    # Create the dendrogram with improved layout
    d = dendrogram(
        Z,
        labels=[f"{label} ({cluster_mapping[i][1]})" for i, label in enumerate(leaf_labels)],
        orientation='right',  # Change to right orientation for better label display
        leaf_rotation=0,
        leaf_font_size=12,
        link_color_func=lambda k: link_colors.get(k, "k"),
        above_threshold_color='k',
        distance_sort='ascending'  # Sort branches by distance for better appearance
    )

    # Fix color palette
    color_palette = []
    for i in range(len(d['icoord'])):
        x_coords = d['icoord'][i]
        for x in [x_coords[0], x_coords[-1]]:
            index = int(x / 5)
            if 0 <= index - 1 < len(Z):
                color = link_colors.get(index - 1, 'k')
                break
            elif 0 <= index < len(leaf_labels):
                color = custom_colors.get(leaf_labels[index], 'k')
                break
        else:
            color = 'k'
        color_palette.append(color)
    set_link_color_palette(color_palette)

    # Increase dendrogram line thickness
    for line in plt.gca().collections:
        if isinstance(line, mcoll.LineCollection):
            line.set_linewidth(2.5)

    # Set colors for leaf labels
    ax = plt.gca()
    for label in ax.get_yticklabels():  # Using yticklabels with right orientation
        txt = label.get_text().split()[0]  # Extract the short label
        if txt in custom_colors:
            label.set_color(custom_colors[txt])
            label.set_fontweight("bold")
    
    # Set title and labels
    plt.title(title, fontsize=18, pad=20)
    plt.xlabel("Distance", fontsize=14, labelpad=10)
    plt.ylabel("Acoustic Strategies", fontsize=14, labelpad=10)
    
    # Adjust layout and appearance
    plt.grid(False)
    plt.tight_layout()
    
    # Save as PDF for high quality
    pdf_file = os.path.splitext(output_file)[0] + ".pdf"
    plt.savefig(pdf_file, format="pdf", bbox_inches="tight")
    print(f"Dendrogram saved as PDF: {pdf_file}")
    
    # Also save as PNG
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Dendrogram saved as PNG: {output_file}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate a dendrogram.")
    parser.add_argument("--traits_file", type=str, default="data/traits_data_pc_gmm_8components_proba.csv", help="CSV file")
    parser.add_argument("--n_components", type=int, default=37, help="Number of principal components (use -1 for all)")
    parser.add_argument("--linkage_method", type=str, default="ward", help="Linkage method (ward, complete, average, single)")
    parser.add_argument("--output_dir", type=str, default="./output/dendogram", help="Output directory")
    parser.add_argument("--output", type=str, help="Output filename (default: dendogram_<method>.png)")
    args = parser.parse_args()

    start_time = time.time()
    print(f"Starting dendogram analysis with linkage method: {args.linkage_method}")

    # Ensure output directory exists
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")
    
    # Set output filename
    if args.output:
        output_file = os.path.join(output_dir, args.output)
    else:
        output_file = os.path.join(output_dir, f"dendogram_{args.linkage_method}.png")
    
    output_prefix = os.path.splitext(output_file)[0]

    # Data Processing
    print(f"Loading data from {args.traits_file}...")
    df = pd.read_csv(args.traits_file)
    
    # Get PC columns sorted numerically
    all_pc_cols = sort_pc_columns(df.columns)
    print(f"Found {len(all_pc_cols)} PC columns: {', '.join(all_pc_cols)}")
    
    # Select PC columns based on n_components
    if args.n_components == -1 or args.n_components >= len(all_pc_cols):
        pc_cols = all_pc_cols
        print(f"Using all {len(pc_cols)} PC columns")
    else:
        pc_cols = all_pc_cols[:args.n_components]
        print(f"Using first {len(pc_cols)} PC columns: {', '.join(pc_cols)}")
    
    if not pc_cols:
        raise ValueError("No PC columns found.")
    if 'gmm_cluster' not in df.columns:
        raise ValueError("'gmm_cluster' column not found.")
    
    # Only use clusters in our cluster_mapping
    df = df[df['gmm_cluster'].isin(cluster_mapping.keys())]
    print(f"Dataset filtered to {len(df)} rows with valid cluster assignments")
    
    # Count samples per cluster
    for cluster_id, (short_name, long_name) in cluster_mapping.items():
        count = len(df[df['gmm_cluster'] == cluster_id])
        print(f"Cluster {cluster_id} ({short_name} - {long_name}): {count} samples")
    
    # Get clusters in the order we want to analyze them
    cluster_order = sorted(list(cluster_mapping.keys()))
    leaf_labels = [cluster_mapping[c][0] for c in cluster_order]
    
    # Calculate inter-cluster distances using all data points
    distance_matrix = compute_intercluster_distances(df, pc_cols, cluster_order)
    
    # Plot distance heatmap
    plot_distance_heatmap(distance_matrix, leaf_labels, output_prefix)
    
    # Convert distance matrix to condensed form for linkage
    print("Converting distance matrix for linkage...")
    condensed_distances = squareform(distance_matrix)
    
    # Perform hierarchical clustering
    print(f"Performing hierarchical clustering with {args.linkage_method} method...")
    Z = linkage(condensed_distances, method=args.linkage_method)
    
    # Set link colors based on the leaf labels
    link_colors = set_link_colors(Z, leaf_labels, custom_colors)

    # Create and save the dendrogram
    title = f"Hierarchical Clustering of Acoustic Strategies ({args.linkage_method})"
    create_dendrogram(Z, leaf_labels, link_colors, title, output_file, cluster_mapping)
    
    # Try different linkage methods and save the results
    if args.linkage_method == 'ward':  # Only do this for the default case
        for method in ['complete', 'average', 'single']:
            if method == args.linkage_method:
                continue
            
            print(f"\nGenerating alternative dendogram with {method} linkage method...")
            alt_output = f"{output_prefix}_{method}.png"
            
            Z_alt = linkage(condensed_distances, method=method)
            link_colors_alt = set_link_colors(Z_alt, leaf_labels, custom_colors)
            
            alt_title = f"Hierarchical Clustering of Acoustic Strategies ({method})"
            create_dendrogram(Z_alt, leaf_labels, link_colors_alt, alt_title, alt_output, cluster_mapping)
    
    total_time = time.time() - start_time
    print(f"\nDendogram analysis completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    main()