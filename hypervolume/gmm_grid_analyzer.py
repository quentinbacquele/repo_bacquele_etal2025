#!/usr/bin/env python3

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
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants for processing
CHUNK_SIZE = 100 

class GMMGridAnalyzer:
    """Class for analyzing GMM probability vectors by grid cell."""
    
    def __init__(self, output_dir="output", num_workers=None, chunk_size=CHUNK_SIZE, cache_dir=None, 
                 feature_type="pc", n_components=8):
        """Initialize the analyzer with options."""
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.grid_df = None
        self.traits_df = None
        self.species_proba_dict = None
        self.species_proba_extreme_dict = None  
        self.valid_species = None
        self.valid_species_extreme = None  
        self.gmm_prob_cols = None
        self.cache_dir = cache_dir or os.path.join(output_dir, "cache")
        self.feature_type = feature_type  # 'pc' or 'umap3d'
        self.n_components = n_components  # Number of GMM components
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def load_data(self):
        """Load datasets with optimized indexing for fast lookups."""
        start_time = time.time()
        
        # Load grid data
        if self.grid_df is None:
            logger.info("Loading grid species data...")
            self.grid_df = pd.read_csv("data/grid_species_lists.csv")
            logger.info(f"Loaded {len(self.grid_df)} grid cells")
        
        # Define feature set based on feature_type
        if self.feature_type == "pc":
            feature_label = f"pc_{self.n_components}"
        else:  # umap3d
            feature_label = f"umap3d_{self.n_components}"
        
        # Define cache path for species probability vectors
        cache_file = os.path.join(self.cache_dir, f"species_proba_cache_{feature_label}.pkl")
        cache_extreme_file = os.path.join(self.cache_dir, f"species_proba_extreme_cache_{feature_label}.pkl")
        
        # Try to load from cache first
        cache_loaded = False
        if os.path.exists(cache_file) and os.path.exists(cache_extreme_file):
            logger.info(f"Loading pre-computed species probability vectors from cache...")
            try:
                # Load regular cache
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.species_proba_dict = cache_data['species_proba_dict']
                    self.valid_species = cache_data['valid_species']
                    self.gmm_prob_cols = cache_data['gmm_prob_cols']
                
                # Load extreme cache
                with open(cache_extreme_file, 'rb') as f:
                    cache_extreme_data = pickle.load(f)
                    self.species_proba_extreme_dict = cache_extreme_data['species_proba_extreme_dict']
                    self.valid_species_extreme = cache_extreme_data['valid_species_extreme']
                
                logger.info(f"Loaded cached probability vectors for {len(self.species_proba_dict)} species")
                logger.info(f"Loaded cached extreme probability vectors for {len(self.species_proba_extreme_dict)} species")
                cache_loaded = True
                return self.grid_df, None 
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Computing from scratch.")
        
        # If cache didn't work, compute from scratch
        if not cache_loaded:
            logger.info("Computing species probability vectors from raw data...")
            
            # Load traits data
            traits_path = "data/traits_data.csv"
            logger.info(f"Loading traits data from {traits_path}...")
            
            # Read all data to compute GMM
            logger.info("Reading and processing traits data...")
            
            # Load the full dataset for GMM computation
            traits_df = pd.read_csv(traits_path)
            
            # Determine feature columns based on feature_type
            if self.feature_type == "pc":
                feature_cols = [col for col in traits_df.columns if col.startswith("PC")]
                logger.info(f"Using {len(feature_cols)} Principal Component columns for clustering")
            else:  # umap3d
                feature_cols = ['UMAP3D 1', 'UMAP3D 2', 'UMAP3D 3']
                logger.info(f"Using UMAP3D coordinates for clustering")
            
            # Extract feature data
            feature_data = traits_df[feature_cols].values
            
            # Scale the data
            logger.info("Scaling the data...")
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(feature_data)
            
            # Apply GMM with specified number of components
            logger.info(f"Applying GMM with {self.n_components} components (covariance_type='full')...")
            gmm = GaussianMixture(n_components=self.n_components, covariance_type='full', random_state=42)
            
            # Fit the model and predict cluster assignments
            traits_df['gmm_cluster'] = gmm.fit_predict(scaled_data)
            
            # Extract probability columns
            logger.info("Extracting cluster membership probabilities...")
            cluster_probs = gmm.predict_proba(scaled_data)
            
            # Define probability column names
            self.gmm_prob_cols = [f'gmm_prob_{i}' for i in range(self.n_components)]
            
            # Add probability columns to the dataframe
            for i in range(self.n_components):
                traits_df[f'gmm_prob_{i}'] = cluster_probs[:, i]
            
            # Count occurrences of probability=1 for each strategy before any averaging
            logger.info("\nCounting raw occurrences of probability=1 for each strategy:")
            print("\n1. Raw counts of extreme cases (before any averaging):")
            print("Strategy | Count (prob=1) | Total Samples | Percentage")
            print("-" * 55)
            total_samples = len(traits_df)
            for i in range(self.n_components):
                prob_col = f'gmm_prob_{i}'
                # Count samples where this probability is ~1 and others are ~0
                extreme_mask = (traits_df[prob_col] == 1)
                count = sum(extreme_mask)
                percentage = (count / total_samples) * 100
                print(f"{i:8d} | {count:13d} | {total_samples:13d} | {percentage:8.2f}%")
            
            # Also count by species
            print("\nSpecies with at least one sample having probability=1 for each strategy:")
            print("Strategy | Unique Species | Total Species | Percentage")
            print("-" * 55)
            total_species = len(traits_df['species'].unique())
            for i in range(self.n_components):
                prob_col = f'gmm_prob_{i}'
                species_with_extreme = len(traits_df[traits_df[prob_col] == 1]['species'].unique())
                percentage = (species_with_extreme / total_species) * 100
                print(f"{i:8d} | {species_with_extreme:14d} | {total_species:13d} | {percentage:8.2f}%")
            
            # Calculate BIC score for diagnostic purposes
            bic_score = gmm.bic(scaled_data)
            logger.info(f"\nGMM BIC score: {bic_score:.2f}")
            
            # Save the GMM probabilities to CSV for future reference
            output_proba_file = f"data/traits_data_{self.feature_type}_gmm_{self.n_components}components_proba.csv"
            logger.info(f"Saving GMM probabilities to {output_proba_file}...")
            traits_df.to_csv(output_proba_file, index=False)
            
            # New approach: Select top 2000 samples for each strategy as extreme cases
            EXTREME_SAMPLE_COUNT = 2000
            print(f"\nUsing new approach: Top {EXTREME_SAMPLE_COUNT} samples per strategy as extreme cases")
            
            # Dictionary to store species probability vectors
            self.species_proba_dict = {}
            
            # Dictionary to store species probability vectors for extreme cases (top 2000)
            self.species_proba_extreme_dict = {}
            
            # Dictionary to store extreme sample indices for each strategy
            extreme_samples_by_strategy = {}
            
            # Identify top N samples for each strategy
            for i in range(self.n_components):
                prob_col = f'gmm_prob_{i}'
                # Sort by probability (descending) and take top N
                top_samples = traits_df.sort_values(by=prob_col, ascending=False).head(EXTREME_SAMPLE_COUNT)
                extreme_samples_by_strategy[i] = set(top_samples.index)
                
                # Print statistics about these top samples
                min_prob = top_samples[prob_col].min()
                max_prob = top_samples[prob_col].max()
                species_count = len(top_samples['species'].unique())
                print(f"Strategy {i}: Min prob={min_prob:.6f}, Max prob={max_prob:.6f}, Unique species={species_count}")
            
            # Collect all extreme samples across all strategies
            all_extreme_samples = set()
            for strategy_samples in extreme_samples_by_strategy.values():
                all_extreme_samples.update(strategy_samples)
            
            print(f"Total unique extreme samples across all strategies: {len(all_extreme_samples)}")
            
            # Create a mask for all extreme samples
            extreme_mask = pd.Series(False, index=traits_df.index)
            extreme_mask.loc[list(all_extreme_samples)] = True
            
            # Extract the extreme dataframe
            extreme_df = traits_df.loc[extreme_mask].copy()
            print(f"Extreme dataframe shape: {extreme_df.shape}")
            
            # Collect all species-level extreme probabilities for statistics
            species_extreme_probs = [[] for _ in range(self.n_components)]
            
            # Process regular species averages
            logger.info("\nCalculating average probability vectors by species...")
            for species, group in tqdm(traits_df.groupby('species'), desc="Processing species"):
                # Regular mean calculation for all samples
                self.species_proba_dict[species] = group[self.gmm_prob_cols].mean().values
            
            # Process extreme species averages separately
            for species, group in tqdm(extreme_df.groupby('species'), desc="Processing extreme species"):
                # Calculate mean for this species using only extreme samples
                extreme_probs = group[self.gmm_prob_cols].mean().values
                self.species_proba_extreme_dict[species] = extreme_probs
                
                # Collect for statistics
                for i in range(self.n_components):
                    species_extreme_probs[i].append(extreme_probs[i])
            
            # Create sets of valid species for fast membership testing
            self.valid_species = set(self.species_proba_dict.keys())
            self.valid_species_extreme = set(self.species_proba_extreme_dict.keys())
            
            # Print statistics after species-level aggregation
            print("\n2. Statistics after species-level aggregation (extreme cases only - top 2000 approach):")
            print("Strategy | Mean Prob | Min Prob | Max Prob | Num Species")
            print("-" * 60)
            for i in range(self.n_components):
                if species_extreme_probs[i]:
                    mean_prob = np.mean(species_extreme_probs[i])
                    min_prob = np.min(species_extreme_probs[i])
                    max_prob = np.max(species_extreme_probs[i])
                    num_species = len(species_extreme_probs[i])
                    print(f"{i:8d} | {mean_prob:.8f} | {min_prob:.8f} | {max_prob:.8f} | {num_species:11d}")
                else:
                    print(f"{i:8d} | {'No data':^10} | {'No data':^10} | {'No data':^10} | {0:11d}")
            
            logger.info(f"Computed probability vectors for {len(self.species_proba_dict)} species")
            logger.info(f"Computed extreme probability vectors for {len(self.species_proba_extreme_dict)} species")
            
            # Save to cache for future runs
            try:
                logger.info(f"Saving species probability vectors to cache...")
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'species_proba_dict': self.species_proba_dict,
                        'valid_species': self.valid_species,
                        'gmm_prob_cols': self.gmm_prob_cols
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                with open(cache_extreme_file, 'wb') as f:
                    pickle.dump({
                        'species_proba_extreme_dict': self.species_proba_extreme_dict,
                        'valid_species_extreme': self.valid_species_extreme
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                logger.info("Cache saved successfully")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
            
            # Clear memory
            del traits_df, feature_data, scaled_data, cluster_probs
            gc.collect()
        
        return self.grid_df, None
    
    def process_batch(self, grid_batch, use_extreme=False):
        """Process a batch of grid IDs for improved efficiency."""
        results = []
        
        # Ensure data is loaded
        if self.species_proba_dict is None:
            self.load_data()
        
        # Select which dictionary and valid species set to use
        if use_extreme:
            proba_dict = self.species_proba_extreme_dict
            valid_species_set = self.valid_species_extreme
        else:
            proba_dict = self.species_proba_dict
            valid_species_set = self.valid_species
        
        for grid_id in grid_batch:
            try:
                # Find the grid in the dataframe
                grid_row = self.grid_df[self.grid_df.grid_id == grid_id]
                
                if grid_row.empty:
                    results.append({
                        'grid_id': grid_id,
                        'status': 'error',
                        'message': f"Grid {grid_id} not found",
                        'max_prob': None,
                        'assigned_cluster': None
                    })
                    continue
                
                # Extract and format species list
                species_list = ast.literal_eval(grid_row.iloc[0].sci_name_list)
                species_list_formatted = [sp.replace(' ', '_') for sp in species_list]
                
                # Find matches using the pre-computed set (much faster)
                matched_species = [sp for sp in species_list_formatted if sp in valid_species_set]
                
                if not matched_species:
                    results.append({
                        'grid_id': grid_id,
                        'status': 'error',
                        'message': f"No matching species found for grid {grid_id}",
                        'max_prob': None,
                        'assigned_cluster': None
                    })
                    continue
                
                # Use the pre-computed probability vectors directly
                species_avg_probs = [proba_dict[sp] for sp in matched_species]
                
                # Calculate average probability vector across all species in the grid
                if species_avg_probs:
                    grid_avg_probs = np.mean(species_avg_probs, axis=0)
                    
                    # Find the maximum probability and corresponding cluster
                    max_prob = float(np.max(grid_avg_probs))
                    assigned_cluster = int(np.argmax(grid_avg_probs))
                    
                    results.append({
                        'grid_id': grid_id,
                        'status': 'success',
                        'max_prob': max_prob,
                        'assigned_cluster': assigned_cluster,
                        'num_species': len(matched_species),
                        'num_samples': len(matched_species),  
                        **{f'prob_{i}': float(prob) for i, prob in enumerate(grid_avg_probs)}
                    })
                else:
                    results.append({
                        'grid_id': grid_id,
                        'status': 'error',
                        'message': f"No valid probability data for grid {grid_id}",
                        'max_prob': None,
                        'assigned_cluster': None
                    })
            
            except Exception as e:
                logger.error(f"Error processing grid {grid_id}: {str(e)}")
                results.append({
                    'grid_id': grid_id,
                    'status': 'error',
                    'message': str(e),
                    'max_prob': None,
                    'assigned_cluster': None
                })
        
        return results
    
    def get_optimal_worker_count(self, total_grids):
        """Calculate optimal number of worker processes based on system resources."""
        if self.num_workers is not None:
            return self.num_workers
        
        # If not specified, calculate based on CPU cores and grid count
        cpu_count = multiprocessing.cpu_count()
        
        # Adjust based on total number of grids
        if total_grids < 100:
            return 1  # Single process for small grid counts
        elif total_grids < 500:
            return min(4, cpu_count)
        else:
            return min(cpu_count - 1, 8)  # Leave one core free for system processes
    
    def setup_worker(self, shared_dict):
        """Initialize worker with shared data."""
        self.species_proba_dict = shared_dict['species_proba_dict']
        self.valid_species = shared_dict['valid_species']
        self.gmm_prob_cols = shared_dict['gmm_prob_cols']
        
        # Add extreme data if available
        if 'species_proba_extreme_dict' in shared_dict:
            self.species_proba_extreme_dict = shared_dict['species_proba_extreme_dict']
            self.valid_species_extreme = shared_dict['valid_species_extreme']
    
    def process_all_grids(self, use_extreme=False):
        """Process all grids to calculate GMM probability metrics with optimized algorithms."""
        start_time = time.time()
        if use_extreme:
            logger.info("Starting GMM extreme probability analysis for all grid cells...")
        else:
            logger.info("Starting GMM probability analysis for all grid cells...")
        
        # Load data and precompute species probability vectors
        self.load_data()
        
        # Get all grid IDs
        all_grid_ids = sorted(self.grid_df['grid_id'].unique())
        total_grids = len(all_grid_ids)
        logger.info(f"Processing {total_grids} grid cells")
        
        # Calculate optimal worker count
        num_workers = self.get_optimal_worker_count(total_grids)
        logger.info(f"Using {num_workers} worker processes")
        
        # Create batches of grid IDs for better efficiency
        grid_batches = [all_grid_ids[i:i + self.chunk_size] 
                      for i in range(0, len(all_grid_ids), self.chunk_size)]
        
        # Set up progress tracking
        pbar = tqdm(total=len(grid_batches), desc="Processing batches", position=0)
        grid_pbar = tqdm(total=total_grids, desc="Processed grids", position=1)
        
        # Results container
        all_results = []
        successful_grids = 0
        
        # Single-process execution for smaller datasets
        if num_workers == 1 or total_grids < 100:
            logger.info("Using single-process execution")
            for batch in grid_batches:
                batch_results = self.process_batch(batch, use_extreme=use_extreme)
                all_results.extend(batch_results)
                
                # Update progress
                pbar.update(1)
                grid_pbar.update(len(batch))
                
                # Count successful grid calculations
                success_count = sum(1 for r in batch_results if r['status'] == 'success')
                successful_grids += success_count
        else:
            # Multi-process execution with shared data for larger datasets
            logger.info("Using multi-process execution with shared data")
            
            # Create a manager to share data between processes
            with Manager() as manager:
                # Create shared dictionary with the pre-computed data
                shared_dict = manager.dict({
                    'species_proba_dict': self.species_proba_dict,
                    'valid_species': self.valid_species,
                    'gmm_prob_cols': self.gmm_prob_cols
                })
                
                # Add extreme data if we're using it
                if use_extreme:
                    shared_dict['species_proba_extreme_dict'] = self.species_proba_extreme_dict
                    shared_dict['valid_species_extreme'] = self.valid_species_extreme
                
                # Create a process pool with initialization
                with ProcessPoolExecutor(max_workers=num_workers, 
                                       initializer=self.setup_worker,
                                       initargs=(shared_dict,)) as executor:
                    
                    # Submit all tasks
                    future_to_batch = {
                        executor.submit(self.process_batch, batch, use_extreme=use_extreme): batch 
                        for batch in grid_batches
                    }
                    
                    # Process results as they complete
                    for future in future_to_batch:
                        batch = future_to_batch[future]
                        
                        try:
                            batch_results = future.result()
                            all_results.extend(batch_results)
                            
                            # Update progress
                            pbar.update(1)
                            grid_pbar.update(len(batch))
                            
                            # Count successful grid calculations
                            success_count = sum(1 for r in batch_results if r['status'] == 'success')
                            successful_grids += success_count
                            
                        except Exception as e:
                            logger.error(f"Error processing batch: {str(e)}")
        
        # Close progress bars
        pbar.close()
        grid_pbar.close()
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(all_results) if all_results else pd.DataFrame(
            columns=['grid_id', 'status', 'max_prob', 'assigned_cluster'])
        
        # Calculate overall summary statistics
        total_time = time.time() - start_time
        num_processed = len(results_df)
        num_success = len(results_df[results_df['status'] == 'success']) if not results_df.empty else 0
        success_rate = num_success / num_processed if num_processed > 0 else 0
        
        # Print statistics after grid-level aggregation
        successful_results = results_df[results_df['status'] == 'success']
        if not successful_results.empty:
            print("\n3. Statistics after grid-level aggregation (extreme cases only - top 2000 approach):")
            print("Strategy | Mean Prob | Min Prob | Max Prob | Num Grids")
            print("-" * 60)
            for i in range(self.n_components):
                prob_col = f'prob_{i}'
                if prob_col in successful_results.columns:
                    mean_prob = successful_results[prob_col].mean()
                    min_prob = successful_results[prob_col].min()
                    max_prob = successful_results[prob_col].max()
                    num_grids = len(successful_results[successful_results[prob_col] > 0])
                    print(f"{i:8d} | {mean_prob:.8f} | {min_prob:.8f} | {max_prob:.8f} | {num_grids:9d}")
                else:
                    print(f"{i:8d} | {'No data':^10} | {'No data':^10} | {'No data':^10} | {0:9d}")
        
        logger.info(f"Processing complete in {total_time/60:.1f} minutes")
        logger.info(f"Processed {num_processed} grids with {num_success} successes "
                   f"({success_rate*100:.1f}% success rate)")
        
        # Save results to CSV
        if not results_df.empty:
            feature_label = f"{self.feature_type}_{self.n_components}"
            if use_extreme:
                results_path = os.path.join(self.output_dir, f"grid_gmm_probabilities_{feature_label}_extreme.csv")
            else:
                results_path = os.path.join(self.output_dir, f"grid_gmm_probabilities_{feature_label}.csv")
            results_df.to_csv(results_path, index=False)
            logger.info(f"Results saved to {results_path}")
        
        return results_df
    
    def process_single_grid(self, target_grid_id, use_extreme=False):
        """Process a single grid ID to calculate average GMM probability vector."""
        # Load data if needed
        self.load_data()
        
        # Process the batch with just one grid ID
        results = self.process_batch([target_grid_id], use_extreme=use_extreme)
        
        # Return the first (and only) result
        return results[0] if results else {
            'grid_id': target_grid_id,
            'status': 'error',
            'message': 'Failed to process grid',
            'max_prob': None,
            'assigned_cluster': None
        }

def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Process grid data and calculate GMM probability metrics.")
    parser.add_argument("--grid", type=str, help="Process a specific grid ID (default: process all grids)")
    parser.add_argument("--output-dir", type=str, default="output", 
                       help="Directory to save results (default: 'output')")
    parser.add_argument("--workers", type=int, help="Number of parallel processes (default: auto)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, 
                       help=f"Number of grids per batch (default: {CHUNK_SIZE})")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching of species probability vectors")
    parser.add_argument("--feature-type", type=str, choices=["pc", "umap3d"], default="pc",
                       help="Feature type to use for GMM clustering (default: pc)")
    parser.add_argument("--n-components", type=int, default=8,
                       help="Number of GMM components (default: 8)")
    parser.add_argument("--run-all", action="store_true", 
                       help="Run all 6 configurations (4, 8, 12 components with PC and UMAP3D)")
    parser.add_argument("--extreme", action="store_true",
                       help="Process only samples with extreme (probability=1) GMM assignments")
    args = parser.parse_args()
    
    if args.run_all:
        # Run all 6 configurations
        configurations = [
            {"feature_type": "pc", "n_components": 4},
            {"feature_type": "pc", "n_components": 8},
            {"feature_type": "pc", "n_components": 12},
            {"feature_type": "umap3d", "n_components": 4},
            {"feature_type": "umap3d", "n_components": 8},
            {"feature_type": "umap3d", "n_components": 12},
        ]
        
        for i, config in enumerate(configurations, 1):
            logger.info(f"Running configuration {i}/6: {config['feature_type']} with {config['n_components']} components")
            
            # Create analyzer with this configuration
            analyzer = GMMGridAnalyzer(
                output_dir=args.output_dir,
                num_workers=args.workers,
                chunk_size=args.chunk_size,
                cache_dir=None if args.no_cache else os.path.join(args.output_dir, "cache"),
                feature_type=config['feature_type'],
                n_components=config['n_components']
            )
            
            # Process all grids
            analyzer.process_all_grids()
            
            # Process extreme cases if requested
            if args.extreme:
                analyzer.process_all_grids(use_extreme=True)
            
            # Clear memory between runs
            gc.collect()
    else:
        # Create analyzer with specified configuration
        analyzer = GMMGridAnalyzer(
            output_dir=args.output_dir,
            num_workers=args.workers,
            chunk_size=args.chunk_size,
            cache_dir=None if args.no_cache else os.path.join(args.output_dir, "cache"),
            feature_type=args.feature_type,
            n_components=args.n_components
        )
        
        # Process single grid or all grids
        if args.grid:
            print(f"🎯 Processing single grid: {args.grid}")
            
            # Process the target grid
            result = analyzer.process_single_grid(args.grid, use_extreme=args.extreme)
            
            # Print the result
            if result['status'] == 'success':
                print(f"\n✅ Grid {args.grid} processed successfully:")
                print(f"Max Probability: {result['max_prob']:.4f}")
                print(f"Assigned Cluster: {result['assigned_cluster']}")
                print(f"Species: {result['num_species']}")
                print(f"Samples: {result['num_samples']}")
                
                # Print all probability values
                print("\n Probability Vector:")
                for i in range(analyzer.n_components):
                    print(f" Cluster {i}: {result[f'prob_{i}']:.4f}")
            else:
                print(f"\n❌ Error processing grid {args.grid}: {result['message']}")
        else:
            # Process all grids
            analyzer.process_all_grids()
            
            # Process extreme cases if requested
            if args.extreme:
                analyzer.process_all_grids(use_extreme=True)

if __name__ == "__main__":
    main() 