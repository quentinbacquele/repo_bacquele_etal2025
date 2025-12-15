# The Global Biogeography of Passerine Songs

Code and analysis scripts for the paper:

**Bacquelé Q.**, Barnagaud J.-Y., Violle C., Theunissen F., Mathevon N. (2025). *The global biogeography of passerine songs.*

## Abstract

Birdsongs are courtship and territorial defense signals that have long served as models of the evolution of vocal communication. However, the extraordinary diversity of songs has lacked a unifying framework accounting for song variation across species and environments. By analyzing the acoustic architecture of songs from over 3,000 passerine species, we show that songs are built from just eight elementary acoustic motifs. Species morphology, social organization, and mating system influence which motifs compose the songs of a bird. Notably, the robustness of an acoustic motif to information loss during propagation affects its distribution across the globe. Our findings demonstrate that the evolution of animal signaling is guided by the interplay of species biology and environmental physics, reflected by geographical patterns on a planetary scale.

## Interactive Visualization

Explore the global vocal repertoire of passerines: [acoustic-biogeography.vercel.app](https://acoustic-biogeography.vercel.app)

## Repository Structure

```
repo_bacquele_etal2025/
├── data/                           # Acoustic feature data and phylogenetic trees
│   ├── AllBirdsEricson1.tre        # Source phylogenetic trees (1000 trees)
│   ├── consensus_sumtrees.tre      # Majority-rule consensus tree
│   ├── traits_data.csv             # Raw acoustic traits per vocalization
│   ├── traits_data_pc_gmm_8components_proba.csv  # GMM cluster probabilities
│   ├── species_traits_data.csv     # Species-level acoustic traits
│   ├── model_traits_data.csv       # Traits data for modeling
│   ├── model_traits_morpho_social_data.csv  # Morphological and social traits
│   ├── grid_species_lists.csv      # Species occurrence per grid cell
│   ├── grid_1.0deg_species_lists.csv  # Species lists at 1° resolution
│   ├── grid_1.0deg_coordID.gpkg    # Grid cell geometries
│   ├── geographic_model_data_with_biomes.csv  # Geographic data with biome info
│   ├── combined_tei_and_environmental_data.csv  # TEI and environmental variables
│   ├── ses_fdis_random_assembly_results_full.csv  # SES-FDis null model results
│   ├── spatial_mpd.csv             # Mean Pairwise Distance results
│   ├── matching_final_corrected.csv  # Taxonomy matching table
│   └── unique_families.txt         # List of passerine families
│
└── scripts/                        # Analysis scripts
    ├── data_parser/                # Data extraction utilities
    ├── mps/                        # Modulation Power Spectrum extraction
    ├── hypervolume/                # Acoustic space construction and motif clustering
    ├── phylogeny/                  # Phylogenetic analyses
    ├── geo models/                 # Spatial regression models
    ├── maps/                       # Global mapping visualizations
    └── species level model/        # Species-level trait analyses
```

## Scripts

### `scripts/data_parser/` - Data Extraction Utilities

| File | Description |
|------|-------------|
| `xeno_canto_extractor.py` | Extracts and processes metadata from xeno-canto recordings. Handles taxonomy matching, data cleaning, and preparation of acoustic datasets for analysis. |

### `scripts/mps/` - Modulation Power Spectrum Extraction

| File | Description |
|------|-------------|
| `extract_mps.py` | Computes Modulation Power Spectra (MPS) from audio recordings. MPS quantifies spectro-temporal modulations encoding information such as species identity, individual identity, and singer quality. Uses a 500 ms window with 67% overlap and 2D Fast Fourier Transform. |

### `scripts/hypervolume/` - Acoustic Space and Motif Clustering

| File | Description |
|------|-------------|
| `acoustic_space_500ms.ipynb` | Jupyter notebook for constructing the 37-dimensional acoustic space from 116,792 passerine vocalizations using weighted PCA. Includes dimensionality reduction validation and UMAP visualization. |
| `gmm_grid_analyzer.py` | Gaussian Mixture Model clustering to identify the eight fundamental acoustic motifs (Flat Whistles, Slow/Fast/Ultrafast Trills, Slow/Fast Modulated Whistles, Harmonic Stacks, Chaotic Notes). Implements AIC/BIC model selection. |
| `dendogram.py` | Computes and visualizes Euclidean distances between acoustic motif clusters in the 37-PC space. |
| `species_cluster_distance_analysis.py` | Analyzes species-level acoustic motif usage patterns, including specialization indices and motif diversity per species. |

### `scripts/phylogeny/` - Phylogenetic Analyses

| File | Description |
|------|-------------|
| `consensus_tree.py` | Constructs majority-rule consensus tree from 1,000 phylogenetic trees (Jetz et al. 2012). Prunes tree to species in the acoustic dataset. |
| `phylo_signals.R` | Calculates phylogenetic signal (Pagel's Lambda, Blomberg's K) for acoustic motif usage. Compares signals between oscine and suboscine passerines. |
| `tree_refined.R` | Reconstructs ancestral states for dominant acoustic motifs and generates the circular phylogram visualization (Fig. 2A). |

### `scripts/geo models/` - Spatial Regression Models

| File | Description |
|------|-------------|
| `tei_geo_models copy.Rmd` | Spatial regression models (INLA SPDE) for the Transmission Efficiency Index (TEI). Tests effects of climate (temperature, humidity), vegetation structure, topography, human footprint, and phylogenetic diversity on global TEI patterns. |
| `fdis_geo_models_new.Rmd` | Spatial regression models for Functional Dispersion (FDis) of acoustic traits within species assemblages. |
| `Biomes_congruence.Rmd` | Tests alignment between acoustic motif distributions and terrestrial biome boundaries. Compares baseline spatial models with biome-effect models using WAIC and LCPO. |

### `scripts/maps/` - Global Mapping

| File | Description |
|------|-------------|
| `richness_map.py` | Maps passerine species richness and motif-specific richness per 1° x 1° grid cell globally. |
| `SES_maps.py` | Computes and maps Standardized Effect Sizes (SES) for motif prevalence using realm-constrained null models. Identifies over/under-representation of each acoustic motif relative to species richness. |
| `dominant_strategy_map.py` | Maps the dominant (most over-represented) acoustic motif per grid cell based on rank-normalized assemblage profiles. |

### `scripts/species level model/` - Species-Level Analyses

| File | Description |
|------|-------------|
| `species_model.Rmd` | Bayesian phylogenetic Dirichlet regression models testing how functional traits (social organization, body size, beak morphology, mating system) predict acoustic motif composition across species (Fig. 2B). |

## The Eight Acoustic Motifs

The study identifies eight fundamental acoustic motifs that structure passerine songs:

1. **Flat Whistles** - Pure tones with no frequency modulation; highest transmission efficiency
2. **Slow Modulated Whistles** - Whistles with slow frequency sweeps
3. **Fast Modulated Whistles** - Whistles with rapid frequency modulation; associated with smaller body size and polygyny
4. **Slow Trills** - Rhythmic repetitions at low rates
5. **Fast Trills** - Rhythmic repetitions at moderate rates
6. **Ultrafast Trills** - Extremely rapid rhythmic patterns
7. **Harmonic Stacks** - Complex harmonic structures; lowest transmission efficiency
8. **Chaotic Notes** - Atonal, broadband sounds

## Data Availability

Acoustic data were acquired from [xeno-canto](https://xeno-canto.org/). The processed acoustic features and metadata are archived at Zenodo (DOI: 10.5281/zenodo.XXXXXXX).

The `data/` folder contains:
- **Phylogenetic trees**: Source trees (`AllBirdsEricson1.tre`) and consensus tree (`consensus_sumtrees.tre`)
- **Acoustic traits**: Raw traits per vocalization (`traits_data.csv`), GMM cluster probabilities (`traits_data_pc_gmm_8components_proba.csv`), and species-level summaries
- **Geographic data**: Species occurrence grids, environmental variables, and biome classifications
- **Model outputs**: SES-FDis results, spatial MPD calculations


## Requirements

### Python
- Python 3.11+
- numpy, scipy, scikit-learn
- librosa (audio processing)
- soundsig (MPS computation)
- umap-learn (dimensionality reduction)
- matplotlib, seaborn (visualization)

### R
- R 4.4.3+
- ape, phytools (phylogenetics)
- brms (Bayesian modeling)
- R-INLA (spatial regression)
- mFD (functional diversity)
- sf, terra (spatial data)

## Citation

```bibtex
@article{bacquele2025biogeography,
  title={The global biogeography of passerine songs},
  author={Bacquel{\'e}, Quentin and Barnagaud, Jean-Yves and Violle, Cyrille and Theunissen, Fr{\'e}d{\'e}ric and Mathevon, Nicolas},
  journal={},
  year={2025}
}
```

## Authors

**Quentin Bacquelé**<sup>1,2,*</sup>, **Jean-Yves Barnagaud**<sup>2,†</sup>, **Cyrille Violle**<sup>2</sup>, **Frédéric Theunissen**<sup>3</sup>, **Nicolas Mathevon**<sup>1,4,5,†</sup>

<br>

**Affiliations**
<br>
<sup>1</sup> ENES Bioacoustics Research Lab, CRNL, CNRS, Inserm, University of Saint-Etienne; Saint-Etienne, France.<br>
<sup>2</sup> CEFE, Univ Montpellier, CNRS, EPHE-PSL University, IRD; 1919 route de Mende, 34293 Montpellier, France.<br>
<sup>3</sup> Department of Neuroscience, University of California, Berkeley; Berkeley, CA 94720, USA.<br>
<sup>4</sup> École Pratique des Hautes Études - PSL, CHArt Lab, University Paris-Sciences-Lettres; Paris, France.<br>
<sup>5</sup> Institut Universitaire de France; Paris, France.

<br>

<sup>*</sup> **Correspondence:** qbacquele@gmail.com<br>
<sup>†</sup> These authors contributed equally to this work.

## Funding

- University of Saint-Etienne
- Ecole Pratique des Hautes Etudes
- University Paris-Sciences-Lettres (PhD stipend to QB)
- Labex CeLyA (NM)
- CNRS, Inserm
- Institut Universitaire de France (NM)
- ACOUCENE group (CESAB, French Foundation for Research on Biodiversity)

## License

Creative Commons Attribution 4.0 International.

## Acknowledgements

We are deeply grateful to the [xeno-canto](https://xeno-canto.org/) citizen science database and all its contributors.
