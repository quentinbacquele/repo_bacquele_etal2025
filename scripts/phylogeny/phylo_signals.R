#!/usr/bin/env Rscript

# ==============================================================================
# Enhanced Phylogenetic Signal Analysis with Parallel Processing
# Purpose: Comprehensive phylogenetic signal analysis with Pagel's Lambda and 
#          Blomberg's K, including per-strategy and oscine/non-oscine splits
# ==============================================================================

# --- Configuration ---
# Set working directory to project root (adjust as needed)
# setwd("path/to/project")

# File paths
tree_files <- c("./output/consensus_pruned_tree_plots/consensus_sumtrees.tre")
tree_names <- c("AllBirdTree")
trait_file <- "./data/model_traits_data.csv"
synonym_file <- "./matching_final_corrected.csv"
oscine_file <- "./data/unique_families.txt"
output_dir <- "./output/phylogenetic_signal/"
Family_col_name <- "family"
species_column_name <- "species"

# Parallel processing configuration
n_cores <- parallel::detectCores() - 1  # Leave one core free
parallel_threshold <- 4  # Minimum strategies to use parallel processing
set.seed(42)

# Analysis options
SKIP_OVERALL_ANALYSIS <- FALSE  # Set to TRUE to skip overall and per-strategy analyses
SKIP_PER_STRATEGY_ANALYSIS <- FALSE  # Set to TRUE to skip per-strategy analysis (but keep overall)
ONLY_OSCINE_ANALYSIS <- FALSE  # Set to TRUE to run ONLY the oscine/non-oscine analysis

# Print analysis configuration
cat("Analysis Configuration:\n")
cat(sprintf("- Skip overall analysis: %s\n", SKIP_OVERALL_ANALYSIS))
cat(sprintf("- Skip per-strategy analysis: %s\n", SKIP_PER_STRATEGY_ANALYSIS)) 
cat(sprintf("- Only oscine analysis: %s\n", ONLY_OSCINE_ANALYSIS))
cat("\n")

# --- Create Output Directory ---
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# --- Load Required Packages ---
required_packages <- c("ape", "readr", "dplyr", "tidyr", "phytools", "parallel", 
                       "foreach", "doParallel", "stringr", "tibble", "purrr",
                       "ggplot2", "ggtree", "ggtreeExtra", "ggnewscale", "RColorBrewer",
                       "fmsb", "scales")

for(pkg in required_packages) {
  if(!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "http://cran.us.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

#' Propagate values from tips to internal nodes (Ancestral State Reconstruction)
#' @param tree Phylogenetic tree
#' @param tip_values_df Data frame with tip values
#' @param species_col_in_df Column name for species in the data frame
#' @param value_cols Vector of value column names
#' @param value_type Type of values being propagated
#' @return Matrix of values for all nodes
propagate_values <- function(tree, tip_values_df, species_col_in_df, value_cols, value_type = "values") {
  n_tips <- length(tree$tip.label)
  n_nodes <- tree$Nnode
  n_total <- n_tips + n_nodes
  num_value_cols <- length(value_cols)
  
  all_values <- matrix(NA, nrow = n_total, ncol = num_value_cols)
  colnames(all_values) <- value_cols
  
  tip_map <- match(tree$tip.label, tip_values_df[[species_col_in_df]])
  valid_tip_indices_in_df <- !is.na(tip_map)
  valid_tip_indices_in_tree <- (1:n_tips)[valid_tip_indices_in_df]
  mapped_df_rows <- tip_map[valid_tip_indices_in_df]
  
  if (length(mapped_df_rows) > 0 && nrow(tip_values_df) > 0 && all(value_cols %in% names(tip_values_df))) {
    values_matrix <- tryCatch({
      as.matrix(tip_values_df[mapped_df_rows, value_cols, drop = FALSE])
    }, error = function(e) {
      matrix(NA, nrow = length(mapped_df_rows), ncol = num_value_cols)
    })
    
    if(nrow(values_matrix) == length(valid_tip_indices_in_tree)){
      all_values[valid_tip_indices_in_tree, ] <- values_matrix
    }
  }
  
  if (all(is.na(all_values[1:n_tips, ]))) {
    return(all_values)
  }
  
  tree_reordered <- ape::reorder.phylo(tree, "postorder")
  edge <- tree_reordered$edge
  
  node_sum_values <- matrix(0, nrow = n_total, ncol = num_value_cols)
  node_child_count <- numeric(n_total)
  
  tip_values_subset <- all_values[1:n_tips, , drop = FALSE]
  is_na_tip_row <- apply(tip_values_subset, 1, function(r) all(is.na(r)))
  
  node_sum_values[1:n_tips,] <- tip_values_subset
  node_sum_values[1:n_tips,][is.na(node_sum_values[1:n_tips,])] <- 0 
  node_child_count[1:n_tips] <- ifelse(is_na_tip_row, 0, 1)
  
  internal_nodes_ordered <- unique(edge[,1][edge[,1] > n_tips])
  
  for (parent_node in internal_nodes_ordered) {
    children <- edge[edge[, 1] == parent_node, 2]
    valid_children <- children[node_child_count[children] > 0]
    
    if (length(valid_children) > 0) {
      summed_child_values <- colSums(node_sum_values[valid_children, , drop = FALSE], na.rm = TRUE)
      total_child_contributors = sum(node_child_count[valid_children], na.rm = TRUE)
      
      if (total_child_contributors > 0) {
        node_sum_values[parent_node,] <- summed_child_values
        node_child_count[parent_node] <- total_child_contributors
        all_values[parent_node,] <- node_sum_values[parent_node,] / node_child_count[parent_node]
      } else {
        all_values[parent_node,] <- NA
        node_child_count[parent_node] <- 0
        node_sum_values[parent_node,] <- 0
      }
    } else {
      all_values[parent_node,] <- NA
      node_child_count[parent_node] <- 0
      node_sum_values[parent_node,] <- 0
    }
  }
  
  return(all_values)
}

#' Load and preprocess oscine/non-oscine classification data
#' @param file_path Path to the oscine classification file
#' @return Data frame with family and category columns
load_oscine_data <- function(file_path) {
  if (!file.exists(file_path)) {
    warning("Oscine classification file not found: ", file_path)
    return(NULL)
  }
  
  lines <- readr::read_lines(file_path, skip_empty_rows = TRUE)
  lines <- lines[!grepl("^\\s*$", lines)]  # Remove empty lines
  
  # Drop header if present
  if (length(lines) && grepl("^\\s*family\\s+category\\s*$", lines[1], ignore.case = TRUE)) {
    lines <- lines[-1]
  }
  
  # Parse each line more robustly
  fam <- character(length(lines))
  catg <- character(length(lines))
  
  for (i in seq_along(lines)) {
    line <- trimws(lines[i])
    
    # Check if line ends with "Oscines" or "Non-Oscines"
    if (grepl("\\s+Oscines\\s*$", line)) {
      catg[i] <- "Oscines"
      fam[i] <- sub("\\s+Oscines\\s*$", "", line)
    } else if (grepl("\\s+Non-Oscines\\s*$", line)) {
      catg[i] <- "Non-Oscines"  
      fam[i] <- sub("\\s+Non-Oscines\\s*$", "", line)
    } else {
      # Try with hyphenated version
      if (grepl("\\s+Non\\-Oscines\\s*$", line)) {
        catg[i] <- "Non-Oscines"
        fam[i] <- sub("\\s+Non\\-Oscines\\s*$", "", line)
      } else {
        warning("Could not parse line: ", line)
        next
      }
    }
    
    # Clean up family name: KEEP parentheses content, just normalize spaces to underscores
    fam[i] <- gsub("\\s+", "_", trimws(fam[i]))
  }
  
  # Create output dataframe, filtering out any failed parses
  valid_indices <- fam != "" & catg != "" & !is.na(fam) & !is.na(catg)
  
  out <- tibble::tibble(
    family = fam[valid_indices], 
    category = catg[valid_indices]
  ) %>%
    dplyr::filter(family != "", !is.na(category))
  
  return(out)
}

#' Calculate phylogenetic signal with robust error handling
#' @param tree Phylogenetic tree
#' @param trait Named vector of trait values
#' @param method Either "lambda" or "K"
#' @param test Whether to perform significance testing
#' @param nsim Number of simulations for significance testing
#' @return List with signal statistics
`%||%` <- function(a, b) if (!is.null(a)) a else b

calculate_phylo_signal <- function(tree,
                                   trait,                  # named numeric vector; names = tip labels
                                   method = c("lambda", "K"),
                                   test   = TRUE,
                                   nsim   = 999) {
  method <- match.arg(method)
  
  # Basic checks
  if (is.null(trait) || !is.numeric(trait)) {
    return(list(method = method, value = NA_real_, P = NA_real_,
                logL = if (method == "lambda") NA_real_ else NULL,
                logL0 = if (method == "lambda") NA_real_ else NULL,
                message = "Trait must be a named numeric vector"))
  }
  
  # Align to tree tips by name (safe even if already aligned)
  shared <- intersect(names(trait), tree$tip.label)
  if (length(shared) < 3L) {
    return(list(method = method, value = NA_real_, P = NA_real_,
                logL = if (method == "lambda") NA_real_ else NULL,
                logL0 = if (method == "lambda") NA_real_ else NULL,
                message = "Insufficient overlap between tree and trait (<3 tips)"))
  }
  tree2   <- ape::keep.tip(tree, shared)
  trait2  <- trait[tree2$tip.label]
  trait2  <- trait2[!is.na(trait2)]
  if (length(trait2) < 3L) {
    return(list(method = method, value = NA_real_, P = NA_real_,
                logL = if (method == "lambda") NA_real_ else NULL,
                logL0 = if (method == "lambda") NA_real_ else NULL,
                message = "Too few non-NA tips (<3)"))
  }
  if (length(unique(trait2)) < 2L) {
    return(list(method = method, value = NA_real_, P = NA_real_,
                logL = if (method == "lambda") NA_real_ else NULL,
                logL0 = if (method == "lambda") NA_real_ else NULL,
                message = "No variance in trait after alignment"))
  }
  
  # Attempt with test (permutations), then fallback to no-test
  res <- try(phytools::phylosig(tree2, trait2, method = method, test = test, nsim = nsim),
             silent = TRUE)
  if (inherits(res, "try-error")) {
    if (test) {
      res <- try(phytools::phylosig(tree2, trait2, method = method, test = FALSE),
                 silent = TRUE)
      if (inherits(res, "try-error")) {
        return(list(method = method, value = NA_real_, P = NA_real_,
                    logL = if (method == "lambda") NA_real_ else NULL,
                    logL0 = if (method == "lambda") NA_real_ else NULL,
                    message = "Computation failed (with and without test)"))
      } else {
        # Success without test
        if (method == "lambda") {
          val  <- res$lambda %||% res[["lambda"]]
          logL <- res$logL %||% res[["logL.lambda"]] %||% NA_real_
          logL0<- res$logL0 %||% res[["logL.null"]]  %||% NA_real_
          return(list(method = "lambda", value = val, P = NA_real_,
                      logL = logL, logL0 = logL0, message = "Success (no test)"))
        } else {
          val <- res$K %||% res[["K"]]
          return(list(method = "K", value = val, P = NA_real_, message = "Success (no test)"))
        }
      }
    } else {
      return(list(method = method, value = NA_real_, P = NA_real_,
                  logL = if (method == "lambda") NA_real_ else NULL,
                  logL0 = if (method == "lambda") NA_real_ else NULL,
                  message = "Computation failed"))
    }
  }
  
  # Normalize outputs across phytools versions
  if (method == "lambda") {
    val  <- res$lambda %||% res[["lambda"]]
    P    <- res$P %||% NA_real_
    logL <- res$logL %||% res[["logL.lambda"]] %||% NA_real_
    logL0<- res$logL0 %||% res[["logL.null"]]  %||% NA_real_
    return(list(method = "lambda", value = val, P = P,
                logL = logL, logL0 = logL0, message = "Success"))
  } else {
    val <- res$K %||% res[["K"]]
    P   <- res$P %||% NA_real_
    return(list(method = "K", value = val, P = P, message = "Success"))
  }
}


#' Process phylogenetic signal for multiple strategies
#' @param tree Phylogenetic tree
#' @param trait_data Data frame with species and trait columns
#' @param strategy_cols Vector of strategy column names
#' @param tree_label_col Column name for tree labels
#' @param parallel Whether to use parallel processing
#' @return List of results for each strategy
process_strategy_signals <- function(tree, trait_data, strategy_cols, 
                                     tree_label_col = "tree_label", 
                                     parallel = FALSE) {
  
  results <- list()
  
  # Function to process single strategy
  process_single_strategy <- function(strat_col) {
    # Prepare trait vector
    trait_vals <- trait_data[[strat_col]]
    names(trait_vals) <- trait_data[[tree_label_col]]
    
    # Align with tree tips and remove NAs
    aligned_trait <- trait_vals[tree$tip.label]
    valid_tips <- !is.na(aligned_trait)
    
    if (sum(valid_tips) < 3) {
      return(list(
        strategy = strat_col,
        n_tips = sum(valid_tips),
        lambda = list(value = NA, P = NA, message = "Too few valid tips"),
        K = list(value = NA, P = NA, message = "Too few valid tips")
      ))
    }
    
    # Subset tree
    tree_subset <- keep.tip(tree, tree$tip.label[valid_tips])
    trait_subset <- aligned_trait[valid_tips]
    
    # Calculate signals
    lambda_res <- calculate_phylo_signal(tree_subset, trait_subset, method = "lambda")
    k_res <- calculate_phylo_signal(tree_subset, trait_subset, method = "K")
    
    return(list(
      strategy = strat_col,
      n_tips = length(tree_subset$tip.label),
      lambda = lambda_res,
      K = k_res
    ))
  }
  
  if (parallel && length(strategy_cols) >= parallel_threshold) {
    # Parallel processing
    cl <- makeCluster(n_cores)
    registerDoParallel(cl)
    on.exit(try(parallel::stopCluster(cl)), add = TRUE)
    
    results <- foreach(strat = strategy_cols, 
                       .packages = c("phytools", "ape"),
                       .export = c("calculate_phylo_signal")) %dopar% {
                         process_single_strategy(strat)
                       }
    
    stopCluster(cl)
    names(results) <- strategy_cols
    
  } else {
    # Sequential processing
    for (strat in strategy_cols) {
      cat(sprintf("  Processing strategy: %s\n", strat))
      results[[strat]] <- process_single_strategy(strat)
    }
  }
  
  return(results)
}

#' Calculate overall phylogenetic signal using maximum probability strategy
#' @param tree Phylogenetic tree  
#' @param trait_data Data frame with species and strategy probability columns
#' @param strategy_cols Vector of strategy column names
#' @param tree_label_col Column name for tree labels
#' @return List with overall signal results
calculate_overall_signal <- function(tree, trait_data, strategy_cols, 
                                     tree_label_col = "tree_label") {
  
  # Calculate maximum probability value for each species
  max_prob_values <- apply(trait_data[, strategy_cols, drop = FALSE], 1, 
                           function(row) {
                             if (all(is.na(row))) NA else max(row, na.rm = TRUE)
                           })
  
  names(max_prob_values) <- trait_data[[tree_label_col]]
  
  # Align with tree and remove NAs
  aligned_trait <- max_prob_values[tree$tip.label]
  valid_tips <- !is.na(aligned_trait)
  
  if (sum(valid_tips) < 3) {
    return(list(
      n_tips = sum(valid_tips),
      lambda = list(value = NA, P = NA, message = "Too few valid tips"),
      K = list(value = NA, P = NA, message = "Too few valid tips")
    ))
  }
  
  # Subset tree
  tree_subset <- keep.tip(tree, tree$tip.label[valid_tips])
  trait_subset <- aligned_trait[valid_tips]
  
  # Calculate signals
  lambda_res <- calculate_phylo_signal(tree_subset, trait_subset, method = "lambda")
  k_res <- calculate_phylo_signal(tree_subset, trait_subset, method = "K")
  
  return(list(
    n_tips = length(tree_subset$tip.label),
    lambda = lambda_res,
    K = k_res
  ))
}

#' Format results into a data frame
#' @param results List of phylogenetic signal results
#' @param analysis_type Type of analysis (e.g., "overall", "per_strategy")
#' @param subset_name Name of subset (e.g., "all", "oscines", "non_oscines")
#' @return Data frame with formatted results
format_results <- function(results, analysis_type, subset_name = "all") {
  
  if (analysis_type == "overall") {
    return(data.frame(
      analysis = analysis_type,
      subset = subset_name,
      strategy = "overall",
      n_tips = results$n_tips,
      lambda_value = results$lambda$value,
      lambda_P = results$lambda$P,
      K_value = results$K$value,
      K_P = results$K$P,
      stringsAsFactors = FALSE
    ))
  } else if (analysis_type == "per_strategy") {
    df_list <- lapply(results, function(r) {
      data.frame(
        analysis = analysis_type,
        subset = subset_name,
        strategy = r$strategy,
        n_tips = r$n_tips,
        lambda_value = r$lambda$value,
        lambda_P = r$lambda$P,
        K_value = r$K$value,
        K_P = r$K$P,
        stringsAsFactors = FALSE
      )
    })
    return(do.call(rbind, df_list))
  }
}

# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

cat("\n==============================================================================\n")
cat("PHYLOGENETIC SIGNAL ANALYSIS WITH PARALLEL PROCESSING\n")
cat("==============================================================================\n")

# --- Load Data ---
cat("\n--- Loading data files ---\n")

# Load trait data
model_traits_data <- readr::read_csv(trait_file, show_col_types = FALSE) %>%
  mutate(across(where(is.character), trimws)) %>%
  mutate(!!sym(species_column_name) := stringr::str_replace_all(!!sym(species_column_name), " ", "_"))

species_in_traits <- unique(model_traits_data[[species_column_name]])

# Process family column
if (Family_col_name %in% names(model_traits_data)) {
  model_traits_data <- model_traits_data %>%
    mutate(!!sym(Family_col_name) := stringr::str_replace_all(!!sym(Family_col_name), " ", "_")) %>%
    mutate(!!sym(Family_col_name) := ifelse(
      is.na(!!sym(Family_col_name)) | !!sym(Family_col_name) == "", 
      "Unknown_Family", 
      !!sym(Family_col_name)))
}

# Load oscine classification
oscine_data <- load_oscine_data(oscine_file)
if (!is.null(oscine_data)) {
  cat(sprintf("Loaded oscine classification for %d families\n", nrow(oscine_data)))
  
  # VALIDATION: Print first few entries to confirm correct parsing
  cat("\n--- Oscine Classification Validation ---\n")
  cat("First few entries from oscine classification file:\n")
  print(head(oscine_data, 10))
  
  # Check category distribution
  cat("\nCategory distribution:\n")
  print(table(oscine_data$category))
  
  # Check for any potential parsing issues
  problematic_families <- oscine_data$family[grepl("\\s", oscine_data$family)]
  if (length(problematic_families) > 0) {
    cat("\nWarning: Found families with spaces (should be underscores):\n")
    print(problematic_families)
  }
  
  # Check if family names match expected format
  cat(sprintf("\nFamily name format check: %d families have underscores, %d have spaces\n",
              sum(grepl("_", oscine_data$family)),
              sum(grepl(" ", oscine_data$family))))
  
  cat("--- End Validation ---\n\n")
  
  # Merge with trait data
  model_traits_data <- model_traits_data %>%
    left_join(oscine_data, by = setNames("family", Family_col_name))
    
  # VALIDATION: Check merge results
  if ("category" %in% names(model_traits_data)) {
    cat("\n--- Merge Validation ---\n")
    cat("Category assignment after merge:\n")
    merge_table <- table(model_traits_data$category, useNA = "ifany")
    print(merge_table)
    
    cat(sprintf("\nSpecies with category assignments: %d/%d (%.1f%%)\n",
                sum(!is.na(model_traits_data$category)),
                nrow(model_traits_data),
                100 * sum(!is.na(model_traits_data$category)) / nrow(model_traits_data)))
    
    # Show some examples of successful matches
    cat("\nSample of successful family-category matches:\n")
    sample_matches <- model_traits_data %>%
      filter(!is.na(category)) %>%
      select(all_of(c(Family_col_name, "category"))) %>%
      distinct() %>%
      head(10)
    print(sample_matches)
    
    cat("--- End Merge Validation ---\n\n")
  } else {
    cat("Warning: 'category' column not found after merge\n")
  }
} else {
  cat("Warning: Could not load oscine classification data\n")
}

# Load synonym data
synonym_data_raw <- readr::read_csv(synonym_file, col_types = readr::cols(.default = "c"), 
                                    show_col_types = FALSE)
synonym_data <- synonym_data_raw %>% 
  mutate(across(everything(), ~stringr::str_replace_all(., " ", "_")))

# Process synonym rows for matching
synonym_rows_list <- apply(synonym_data, 1, function(row) {
  unique(as.character(row[!is.na(row) & row != "" & row != "_"]))
})

# Identify strategy columns
desired_strategy_order <- c(3, 7, 4, 2, 1, 6, 5, 0)
prob_cols <- paste0("gmm_prob_", desired_strategy_order, "_mean")
available_prob_cols <- grep("gmm_prob_\\d+_mean$", names(model_traits_data), value = TRUE)
prob_cols <- prob_cols[prob_cols %in% available_prob_cols]

num_clusters <- length(prob_cols)
cat(sprintf("Found %d strategy probability columns\n", num_clusters))

# Pause to review validation before starting computation
cat("\n==============================================================================\n")
cat("DATA LOADING AND VALIDATION COMPLETE\n")
cat("==============================================================================\n")

if (ONLY_OSCINE_ANALYSIS) {
  cat("*** FAST MODE: Running only Oscine vs Non-Oscine analysis ***\n")
} else if (SKIP_OVERALL_ANALYSIS && SKIP_PER_STRATEGY_ANALYSIS) {
  cat("*** PARTIAL MODE: Skipping overall and per-strategy analyses ***\n")
} else if (SKIP_OVERALL_ANALYSIS) {
  cat("*** PARTIAL MODE: Skipping overall analysis ***\n")
} else if (SKIP_PER_STRATEGY_ANALYSIS) {
  cat("*** PARTIAL MODE: Skipping per-strategy analysis ***\n")
} else {
  cat("*** FULL MODE: Running all analyses ***\n")
}

cat("Please review the validation output above.\n")
cat("Press Enter to continue with phylogenetic signal analysis, or Ctrl+C to stop...\n")
if (interactive()) {
  readline()
} else {
  cat("Non-interactive mode - continuing automatically\n")
}

# Initialize results storage
all_results <- list()

# --- Process Each Tree ---
for (i_tree in seq_along(tree_files)) {
  
  current_tree_file <- tree_files[i_tree]
  current_tree_name <- tree_names[i_tree]
  
  cat(sprintf("\n=== Processing Tree: %s ===\n", current_tree_name))
  
  # Load tree
  full_tree <- ape::read.tree(current_tree_file)
  if (inherits(full_tree, "multiPhylo")) {
    full_tree <- full_tree[[1]]
  }
  full_tree$tip.label <- stringr::str_replace_all(full_tree$tip.label, " ", "_")
  
  # Get cleaned tree labels
  cleaned_original_tree_labels <- full_tree$tip.label
  
  # Initial direct matches
  initial_matches <- intersect(species_in_traits, cleaned_original_tree_labels)
  species_missing_in_tree <- setdiff(species_in_traits, initial_matches)
  tree_labels_still_available <- setdiff(cleaned_original_tree_labels, initial_matches)
  
  # Synonym matching (EXACT logic from original script)
  synonym_match_results <- list()
  found_via_synonym_count <- 0
  
  if (length(species_missing_in_tree) > 0 && 
      length(tree_labels_still_available) > 0 && 
      length(synonym_rows_list) > 0) {
    
    # Track which tree labels have been used
    available_tree_label_set <- setNames(rep(TRUE, length(tree_labels_still_available)), 
                                         nm = tree_labels_still_available)
    
    for (idx_s in seq_along(species_missing_in_tree)) {
      missing_species <- species_missing_in_tree[idx_s]
      
      # Find synonym rows containing this species
      relevant_row_indices <- which(sapply(synonym_rows_list, function(syn_row) {
        missing_species %in% syn_row
      }))
      
      if (length(relevant_row_indices) > 0) {
        # Get all potential synonyms from relevant rows
        potential_synonyms <- unique(unlist(synonym_rows_list[relevant_row_indices]))
        potential_synonyms <- potential_synonyms[potential_synonyms != "" & 
                                                   !is.na(potential_synonyms) & 
                                                   potential_synonyms != "_"]
        
        # Try to find a match in available tree labels
        for (syn in potential_synonyms) {
          if (syn %in% names(available_tree_label_set) && 
              isTRUE(available_tree_label_set[[syn]])) {
            synonym_match_results[[missing_species]] <- syn
            available_tree_label_set[[syn]] <- FALSE
            found_via_synonym_count <- found_via_synonym_count + 1
            break
          }
        }
      }
    }
  }
  
  # Create mapping dataframe
  mapping_df_this_tree <- data.frame(
    trait_species = initial_matches, 
    tree_label = initial_matches, 
    match_type = "direct", 
    stringsAsFactors = FALSE
  )
  
  if (length(synonym_match_results) > 0) {
    synonym_df <- data.frame(
      trait_species = names(synonym_match_results), 
      tree_label = unlist(synonym_match_results), 
      match_type = "synonym", 
      stringsAsFactors = FALSE
    )
    mapping_df_this_tree <- bind_rows(mapping_df_this_tree, synonym_df)
  }
  
  # Get final tree labels to keep
  final_tree_labels_to_keep <- unique(mapping_df_this_tree$tree_label)
  
  cat(sprintf("Matching results: %d direct matches, %d synonym matches, %d total\n",
              length(initial_matches), found_via_synonym_count, 
              length(final_tree_labels_to_keep)))
  
  if (length(final_tree_labels_to_keep) < 3) {
    cat(sprintf("Warning: Only %d matches found. Skipping tree.\n", 
                length(final_tree_labels_to_keep)))
    next
  }
  
  # Prune tree
  pruned_tree <- ape::keep.tip(full_tree, final_tree_labels_to_keep)
  if (!is.binary(pruned_tree)) {
    pruned_tree <- multi2di(pruned_tree)
  }
  
  # Fix zero or near-zero branch lengths
  min_bl_threshold <- .Machine$double.eps^0.5
  if (any(pruned_tree$edge.length < min_bl_threshold, na.rm = TRUE)) {
    zero_indices <- which(pruned_tree$edge.length < min_bl_threshold)
    min_pos_bl <- min(pruned_tree$edge.length[pruned_tree$edge.length >= min_bl_threshold], 
                      na.rm = TRUE)
    small_val <- if (is.finite(min_pos_bl) && min_pos_bl > 0) {
      min_pos_bl * 0.001
    } else {
      1e-8
    }
    pruned_tree$edge.length[zero_indices] <- pruned_tree$edge.length[zero_indices] + small_val
  }
  
  # Prepare trait data using mapping
  additional_cols <- c(Family_col_name)
  if ("category" %in% names(model_traits_data)) {
    additional_cols <- c(additional_cols, "category")
  }
  
  trait_data <- mapping_df_this_tree %>%
    filter(tree_label %in% pruned_tree$tip.label) %>%
    inner_join(model_traits_data %>% 
                 select(all_of(species_column_name), all_of(prob_cols), 
                        all_of(additional_cols)), 
               by = c("trait_species" = species_column_name)) %>%
    select(tree_label, all_of(prob_cols), all_of(additional_cols)) %>%
    distinct(tree_label, .keep_all = TRUE)
  
  cat(sprintf("Tree has %d tips with trait data\n", nrow(trait_data)))
  
  # --- 1. OVERALL PHYLOGENETIC SIGNAL ---
  if (!SKIP_OVERALL_ANALYSIS && !ONLY_OSCINE_ANALYSIS) {
    cat("\n--- Calculating overall phylogenetic signal ---\n")
    
    overall_results <- calculate_overall_signal(pruned_tree, trait_data, prob_cols)
    
    cat(sprintf("Overall Lambda: %.4f (P = %.4f)\n", 
                ifelse(is.na(overall_results$lambda$value), NA, overall_results$lambda$value),
                ifelse(is.na(overall_results$lambda$P), NA, overall_results$lambda$P)))
    cat(sprintf("Overall Blomberg's K: %.4f (P = %.4f)\n",
                ifelse(is.na(overall_results$K$value), NA, overall_results$K$value),
                ifelse(is.na(overall_results$K$P), NA, overall_results$K$P)))
    
    all_results[[paste0(current_tree_name, "_overall")]] <- 
      format_results(overall_results, "overall", "all")
  } else {
    cat("\n--- Skipping overall phylogenetic signal analysis ---\n")
  }
  
  # --- 2. PER-STRATEGY PHYLOGENETIC SIGNAL ---
  if (!SKIP_PER_STRATEGY_ANALYSIS && !ONLY_OSCINE_ANALYSIS) {
    cat("\n--- Calculating per-strategy phylogenetic signals ---\n")
    
    use_parallel <- (num_clusters >= parallel_threshold) && (n_cores > 1)
    if (use_parallel) {
      cat(sprintf("Using parallel processing with %d cores\n", n_cores))
    }
    
    strategy_results <- process_strategy_signals(pruned_tree, trait_data, prob_cols, 
                                                 parallel = use_parallel)
    
    # Print summary
    for (strat_name in names(strategy_results)) {
      res <- strategy_results[[strat_name]]
      cat(sprintf("  %s: Lambda = %.4f (P = %.4f), K = %.4f (P = %.4f), n = %d\n",
                  strat_name,
                  ifelse(is.na(res$lambda$value), NA, res$lambda$value),
                  ifelse(is.na(res$lambda$P), NA, res$lambda$P),
                  ifelse(is.na(res$K$value), NA, res$K$value),
                  ifelse(is.na(res$K$P), NA, res$K$P),
                  res$n_tips))
    }
    
    all_results[[paste0(current_tree_name, "_per_strategy")]] <- 
      format_results(strategy_results, "per_strategy", "all")
  } else {
    cat("\n--- Skipping per-strategy phylogenetic signal analysis ---\n")
  }
  
  # --- 3. OSCINE/NON-OSCINE SPLIT ANALYSIS ---
  if (!is.null(oscine_data) && "category" %in% names(trait_data)) {
    
    if (ONLY_OSCINE_ANALYSIS) {
      cat("\n--- RUNNING ONLY OSCINE/NON-OSCINE ANALYSIS ---\n")
    } else {
      cat("\n--- Calculating phylogenetic signals for Oscines vs Non-Oscines ---\n")
    }
    
    # Add validation before starting oscine analysis
    cat("\n--- Pre-analysis validation for oscine/non-oscine data ---\n")
    category_counts <- table(trait_data$category, useNA = "ifany")
    print(category_counts)
    
    oscines_count <- sum(trait_data$category == "Oscines", na.rm = TRUE)
    non_oscines_count <- sum(trait_data$category == "Non-Oscines", na.rm = TRUE)
    
    cat(sprintf("Available for analysis: %d Oscines, %d Non-Oscines\n", 
                oscines_count, non_oscines_count))
    
    if (oscines_count < 3) {
      cat("Warning: Insufficient Oscines data (< 3 species)\n")
    }
    if (non_oscines_count < 3) {
      cat("Warning: Insufficient Non-Oscines data (< 3 species)\n")
    }
    cat("--- End pre-analysis validation ---\n")
    
    for (bird_category in c("Oscines", "Non-Oscines")) {
      
      cat(sprintf("\n  Processing %s:\n", bird_category))
      
      # Filter data
      category_data <- trait_data %>%
        filter(category == bird_category)
      
      if (nrow(category_data) < 3) {
        cat(sprintf("    Insufficient data for %s (n = %d)\n", 
                    bird_category, nrow(category_data)))
        next
      }
      
      # Subset tree
      category_tree <- keep.tip(pruned_tree, category_data$tree_label)
      
      cat(sprintf("    Subset has %d tips\n", length(category_tree$tip.label)))
      
      # Overall signal for subset
      cat(sprintf("    Calculating overall signal for %s...\n", bird_category))
      overall_category <- calculate_overall_signal(category_tree, category_data, prob_cols)
      
      all_results[[paste0(current_tree_name, "_overall_", bird_category)]] <- 
        format_results(overall_category, "overall", bird_category)
      
      cat(sprintf("    %s Overall - Lambda: %.4f (P = %.4f), K: %.4f (P = %.4f)\n",
                  bird_category,
                  ifelse(is.na(overall_category$lambda$value), NA, overall_category$lambda$value),
                  ifelse(is.na(overall_category$lambda$P), NA, overall_category$lambda$P),
                  ifelse(is.na(overall_category$K$value), NA, overall_category$K$value),
                  ifelse(is.na(overall_category$K$P), NA, overall_category$K$P)))
      
      # Per-strategy signal for subset
      cat(sprintf("    Calculating per-strategy signals for %s...\n", bird_category))
      
      use_parallel <- (num_clusters >= parallel_threshold) && (n_cores > 1)
      strategy_category <- process_strategy_signals(category_tree, category_data, 
                                                    prob_cols, parallel = use_parallel)
      
      all_results[[paste0(current_tree_name, "_per_strategy_", bird_category)]] <- 
        format_results(strategy_category, "per_strategy", bird_category)
    }
    
    # --- 4. STATISTICAL COMPARISON BETWEEN OSCINES AND NON-OSCINES ---
    cat("\n--- Comparing phylogenetic signals between Oscines and Non-Oscines ---\n")
    
    oscine_data_only <- trait_data %>% filter(category == "Oscines")
    non_oscine_data_only <- trait_data %>% filter(category == "Non-Oscines")
    
    if (nrow(oscine_data_only) >= 3 && nrow(non_oscine_data_only) >= 3) {
      comparison_results <- data.frame(
        strategy = character(),
        oscine_lambda = numeric(),
        non_oscine_lambda = numeric(),
        lambda_diff = numeric(),
        oscine_K = numeric(),
        non_oscine_K = numeric(),
        K_diff = numeric(),
        stringsAsFactors = FALSE
      )
      
      for (strat_col in prob_cols) {
        # Get oscine results
        osc_key <- paste0(current_tree_name, "_per_strategy_Oscines")
        non_osc_key <- paste0(current_tree_name, "_per_strategy_Non-Oscines")
        
        if (osc_key %in% names(all_results) && non_osc_key %in% names(all_results)) {
          osc_df <- all_results[[osc_key]]
          non_osc_df <- all_results[[non_osc_key]]
          
          osc_row <- osc_df[osc_df$strategy == strat_col, ]
          non_osc_row <- non_osc_df[non_osc_df$strategy == strat_col, ]
          
          if (nrow(osc_row) > 0 && nrow(non_osc_row) > 0) {
            comparison_results <- rbind(comparison_results, data.frame(
              strategy = strat_col,
              oscine_lambda = osc_row$lambda_value,
              non_oscine_lambda = non_osc_row$lambda_value,
              lambda_diff = osc_row$lambda_value - non_osc_row$lambda_value,
              oscine_K = osc_row$K_value,
              non_oscine_K = non_osc_row$K_value,
              K_diff = osc_row$K_value - non_osc_row$K_value,
              stringsAsFactors = FALSE
            ))
          }
        }
      }
      
      if (nrow(comparison_results) > 0) {
        cat("\nComparison Summary:\n")
        print(comparison_results)
        
        # Save comparison
        comparison_file <- file.path(output_dir, 
                                     paste0("oscine_nonoscine_comparison_", 
                                            current_tree_name, ".csv"))
        write.csv(comparison_results, comparison_file, row.names = FALSE)
        cat(sprintf("\nComparison saved to: %s\n", comparison_file))
      }
    }
  } else {
    cat("\n--- Oscine/Non-Oscine analysis skipped (no classification data) ---\n")
  }
}

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

cat("\n--- Saving all results ---\n")

# Combine all results into single data frame
if (length(all_results) > 0) {
  combined_results <- do.call(rbind, all_results)
  rownames(combined_results) <- NULL
  
  # Add significance interpretation
  combined_results$lambda_sig <- ifelse(combined_results$lambda_P < 0.001, "***",
                                        ifelse(combined_results$lambda_P < 0.01, "**",
                                               ifelse(combined_results$lambda_P < 0.05, "*", "ns")))
  
  combined_results$K_sig <- ifelse(combined_results$K_P < 0.001, "***",
                                   ifelse(combined_results$K_P < 0.01, "**",
                                          ifelse(combined_results$K_P < 0.05, "*", "ns")))
  
  # Save main results
  main_results_file <- file.path(output_dir, "phylogenetic_signal_results.csv")
  write.csv(combined_results, main_results_file, row.names = FALSE)
  cat(sprintf("Main results saved to: %s\n", main_results_file))
  
  # Create summary statistics
  summary_stats <- combined_results %>%
    group_by(analysis, subset) %>%
    summarise(
      n_strategies = n(),
      mean_lambda = mean(lambda_value, na.rm = TRUE),
      sd_lambda = sd(lambda_value, na.rm = TRUE),
      mean_K = mean(K_value, na.rm = TRUE),
      sd_K = sd(K_value, na.rm = TRUE),
      prop_sig_lambda = mean(lambda_P < 0.05, na.rm = TRUE),
      prop_sig_K = mean(K_P < 0.05, na.rm = TRUE),
      .groups = "drop"
    )
  
  summary_file <- file.path(output_dir, "phylogenetic_signal_summary.csv")
  write.csv(summary_stats, summary_file, row.names = FALSE)
  cat(sprintf("Summary statistics saved to: %s\n", summary_file))
  
  # Create visualization of phylogenetic signal results
  if (require("ggplot2", quietly = TRUE)) {
    cat("\nGenerating visualization plots...\n")
    
    # Plot 1: Lambda vs K values by strategy
    if (any(combined_results$analysis == "per_strategy")) {
      p1 <- ggplot(combined_results %>% filter(analysis == "per_strategy"),
                   aes(x = lambda_value, y = K_value, color = subset)) +
        geom_point(size = 3, alpha = 0.7) +
        geom_hline(yintercept = 1, linetype = "dashed", color = "gray50") +
        geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
        scale_color_manual(values = c("all" = "#0072B2", 
                                      "Oscines" = "#009E73", 
                                      "Non-Oscines" = "#E69F00")) +
        labs(x = "Pagel's Lambda", y = "Blomberg's K",
             title = "Phylogenetic Signal Comparison",
             subtitle = "Per-strategy values across different bird groups") +
        theme_minimal() +
        theme(legend.position = "bottom")
      
      plot_file1 <- file.path(output_dir, "phylogenetic_signal_scatter.pdf")
      ggsave(plot_file1, p1, width = 8, height = 6)
      cat(sprintf("Scatter plot saved to: %s\n", plot_file1))
    }
    
    # Plot 2: Bar plot of signal strength by strategy
    if (any(combined_results$analysis == "per_strategy")) {
      plot_data <- combined_results %>%
        filter(analysis == "per_strategy") %>%
        select(subset, strategy, lambda_value, K_value) %>%
        pivot_longer(cols = c(lambda_value, K_value),
                     names_to = "measure",
                     values_to = "value") %>%
        mutate(measure = ifelse(measure == "lambda_value", "Pagel's λ", "Blomberg's K"))
      
      p2 <- ggplot(plot_data, aes(x = strategy, y = value, fill = subset)) +
        geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
        facet_wrap(~ measure, scales = "free_y") +
        scale_fill_manual(values = c("all" = "#0072B2", 
                                     "Oscines" = "#009E73", 
                                     "Non-Oscines" = "#E69F00")) +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1),
              legend.position = "bottom") +
        labs(x = "Strategy", y = "Signal Strength",
             title = "Phylogenetic Signal by Strategy and Bird Group")
      
      plot_file2 <- file.path(output_dir, "phylogenetic_signal_barplot.pdf")
      ggsave(plot_file2, p2, width = 10, height = 6)
      cat(sprintf("Bar plot saved to: %s\n", plot_file2))
    }
  }
  
  # Print summary to console
  cat("\n==============================================================================\n")
  cat("ANALYSIS SUMMARY\n")
  cat("==============================================================================\n")
  print(summary_stats)
}

# Save session info for reproducibility
session_info_file <- file.path(output_dir, "session_info.txt")
writeLines(capture.output(sessionInfo()), session_info_file)

cat("\n==============================================================================\n")
cat("PHYLOGENETIC SIGNAL ANALYSIS COMPLETE\n")
cat(sprintf("Results saved to: %s\n", output_dir))
cat("==============================================================================\n")

# --- 4. STATISTICAL COMPARISON BETWEEN OSCINES AND NON-OSCINES ---
cat("\n--- Comparing phylogenetic signals between Oscines and Non-Oscines ---\n")

oscine_data_only <- trait_data %>% filter(category == "Oscines")
non_oscine_data_only <- trait_data %>% filter(category == "Non-Oscines")

if (nrow(oscine_data_only) >= 3 && nrow(non_oscine_data_only) >= 3) {
  comparison_results <- data.frame(
    strategy = character(),
    oscine_lambda = numeric(),
    non_oscine_lambda = numeric(),
    lambda_diff = numeric(),
    oscine_K = numeric(),
    non_oscine_K = numeric(),
    K_diff = numeric(),
    wilcox_p_lambda = numeric(),
    wilcox_p_K = numeric(),
    cohen_d_lambda = numeric(),
    cohen_d_K = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Helper function to calculate Cohen's d
  calculate_cohens_d <- function(x1, x2) {
    n1 <- length(x1)
    n2 <- length(x2)
    if (n1 < 2 || n2 < 2) return(NA)
    
    m1 <- mean(x1, na.rm = TRUE)
    m2 <- mean(x2, na.rm = TRUE)
    s1 <- sd(x1, na.rm = TRUE)
    s2 <- sd(x2, na.rm = TRUE)
    
    # Pooled standard deviation
    pooled_sd <- sqrt(((n1 - 1) * s1^2 + (n2 - 1) * s2^2) / (n1 + n2 - 2))
    
    if (pooled_sd == 0) return(NA)
    
    cohens_d <- (m1 - m2) / pooled_sd
    return(cohens_d)
  }
  
  # Helper function to safely perform Wilcoxon test
  safe_wilcox_test <- function(x1, x2) {
    if (length(x1) < 3 || length(x2) < 3 || all(is.na(x1)) || all(is.na(x2))) {
      return(NA)
    }
    
    # Remove NAs
    x1_clean <- x1[!is.na(x1)]
    x2_clean <- x2[!is.na(x2)]
    
    if (length(x1_clean) < 2 || length(x2_clean) < 2) {
      return(NA)
    }
    
    # Check if there's any variation
    if (length(unique(c(x1_clean, x2_clean))) < 2) {
      return(NA)
    }
    
    tryCatch({
      test_result <- wilcox.test(x1_clean, x2_clean)
      return(test_result$p.value)
    }, error = function(e) {
      return(NA)
    })
  }
  
  # --- OVERALL COMPARISON ---
  cat("\n  Comparing overall phylogenetic signals...\n")
  
  osc_overall_key <- paste0(current_tree_name, "_overall_Oscines")
  non_osc_overall_key <- paste0(current_tree_name, "_overall_Non-Oscines")
  
  if (osc_overall_key %in% names(all_results) && non_osc_overall_key %in% names(all_results)) {
    osc_overall <- all_results[[osc_overall_key]]
    non_osc_overall <- all_results[[non_osc_overall_key]]
    
    # Calculate maximum probability values for each group (same as overall analysis)
    oscine_max_probs <- apply(oscine_data_only[, prob_cols, drop = FALSE], 1, 
                              function(row) {
                                if (all(is.na(row))) NA else max(row, na.rm = TRUE)
                              })
    non_oscine_max_probs <- apply(non_oscine_data_only[, prob_cols, drop = FALSE], 1, 
                                  function(row) {
                                    if (all(is.na(row))) NA else max(row, na.rm = TRUE)
                                  })
    
    # Perform statistical tests on the maximum probability values
    wilcox_p_overall <- safe_wilcox_test(oscine_max_probs, non_oscine_max_probs)
    cohen_d_overall <- calculate_cohens_d(oscine_max_probs, non_oscine_max_probs)
    
    # Add overall comparison to results
    comparison_results <- rbind(comparison_results, data.frame(
      strategy = "overall",
      oscine_lambda = osc_overall$lambda_value,
      non_oscine_lambda = non_osc_overall$lambda_value,
      lambda_diff = osc_overall$lambda_value - non_osc_overall$lambda_value,
      oscine_K = osc_overall$K_value,
      non_oscine_K = non_osc_overall$K_value,
      K_diff = osc_overall$K_value - non_osc_overall$K_value,
      wilcox_p_lambda = wilcox_p_overall,
      wilcox_p_K = wilcox_p_overall,  # Same test for both metrics
      cohen_d_lambda = cohen_d_overall,
      cohen_d_K = cohen_d_overall,    # Same effect size for both metrics
      stringsAsFactors = FALSE
    ))
  }
  
  # --- PER-STRATEGY COMPARISONS ---
  cat("  Comparing per-strategy phylogenetic signals...\n")
  
  for (strat_col in prob_cols) {
    # Get oscine and non-oscine results for this strategy
    osc_key <- paste0(current_tree_name, "_per_strategy_Oscines")
    non_osc_key <- paste0(current_tree_name, "_per_strategy_Non-Oscines")
    
    if (osc_key %in% names(all_results) && non_osc_key %in% names(all_results)) {
      osc_df <- all_results[[osc_key]]
      non_osc_df <- all_results[[non_osc_key]]
      
      osc_row <- osc_df[osc_df$strategy == strat_col, ]
      non_osc_row <- non_osc_df[non_osc_df$strategy == strat_col, ]
      
      if (nrow(osc_row) > 0 && nrow(non_osc_row) > 0) {
        
        # Prepare trait vectors for statistical testing
        oscine_traits <- oscine_data_only[[strat_col]]
        names(oscine_traits) <- oscine_data_only$tree_label
        oscine_traits_clean <- oscine_traits[!is.na(oscine_traits)]
        
        non_oscine_traits <- non_oscine_data_only[[strat_col]]
        names(non_oscine_traits) <- non_oscine_data_only$tree_label
        non_oscine_traits_clean <- non_oscine_traits[!is.na(non_oscine_traits)]
        
        # Perform statistical tests
        wilcox_p_strategy <- safe_wilcox_test(oscine_traits_clean, non_oscine_traits_clean)
        cohen_d_strategy <- calculate_cohens_d(oscine_traits_clean, non_oscine_traits_clean)
        
        comparison_results <- rbind(comparison_results, data.frame(
          strategy = strat_col,
          oscine_lambda = osc_row$lambda_value,
          non_oscine_lambda = non_osc_row$lambda_value,
          lambda_diff = osc_row$lambda_value - non_osc_row$lambda_value,
          oscine_K = osc_row$K_value,
          non_oscine_K = non_osc_row$K_value,
          K_diff = osc_row$K_value - non_osc_row$K_value,
          wilcox_p_lambda = wilcox_p_strategy,
          wilcox_p_K = wilcox_p_strategy,    # Same test for both metrics
          cohen_d_lambda = cohen_d_strategy,
          cohen_d_K = cohen_d_strategy,      # Same effect size for both metrics
          stringsAsFactors = FALSE
        ))
      }
    }
  }
  
  if (nrow(comparison_results) > 0) {
    # Reorder so overall comes first
    comparison_results <- comparison_results[order(comparison_results$strategy == "overall", 
                                                   comparison_results$strategy, 
                                                   decreasing = c(TRUE, FALSE), 
                                                   method = "radix"), ]
    
    # Add significance flags
    comparison_results$wilcox_sig_lambda <- ifelse(comparison_results$wilcox_p_lambda < 0.001, "***",
                                                   ifelse(comparison_results$wilcox_p_lambda < 0.01, "**",
                                                          ifelse(comparison_results$wilcox_p_lambda < 0.05, "*", "ns")))
    comparison_results$wilcox_sig_K <- ifelse(comparison_results$wilcox_p_K < 0.001, "***",
                                              ifelse(comparison_results$wilcox_p_K < 0.01, "**",
                                                     ifelse(comparison_results$wilcox_p_K < 0.05, "*", "ns")))
    
    cat("\nComparison Summary with Statistical Tests:\n")
    print(comparison_results)
    
    # Print summary statistics
    total_comparisons <- nrow(comparison_results)
    strategy_comparisons <- total_comparisons - 1  # Subtract the overall comparison
    
    cat(sprintf("\nSummary of statistical comparisons:\n"))
    cat(sprintf("- Overall comparison: 1\n"))
    cat(sprintf("- Per-strategy comparisons: %d\n", strategy_comparisons))
    
    # Overall results
    overall_row <- comparison_results[comparison_results$strategy == "overall", ]
    if (nrow(overall_row) > 0) {
      cat(sprintf("\nOverall trait comparison:\n"))
      cat(sprintf("- Wilcoxon p-value: %.4f (%s)\n", 
                  overall_row$wilcox_p_lambda, overall_row$wilcox_sig_lambda))
      cat(sprintf("- Cohen's d: %.3f\n", overall_row$cohen_d_lambda))
    }
    
    # Strategy-specific results
    strategy_rows <- comparison_results[comparison_results$strategy != "overall", ]
    if (nrow(strategy_rows) > 0) {
      cat(sprintf("\nPer-strategy trait comparisons:\n"))
      cat(sprintf("- Significant differences (p < 0.05): %d/%d strategies\n",
                  sum(strategy_rows$wilcox_p_lambda < 0.05, na.rm = TRUE),
                  sum(!is.na(strategy_rows$wilcox_p_lambda))))
      cat(sprintf("- Mean |Cohen's d|: %.3f (range: %.3f to %.3f)\n",
                  mean(abs(strategy_rows$cohen_d_lambda), na.rm = TRUE),
                  min(abs(strategy_rows$cohen_d_lambda), na.rm = TRUE),
                  max(abs(strategy_rows$cohen_d_lambda), na.rm = TRUE)))
    }
    
    # Save comparison with statistical tests
    comparison_file <- file.path(output_dir, 
                                 paste0("oscine_nonoscine_comparison_", 
                                        current_tree_name, ".csv"))
    write.csv(comparison_results, comparison_file, row.names = FALSE)
    cat(sprintf("\nComparison with statistical tests saved to: %s\n", comparison_file))
  }
}