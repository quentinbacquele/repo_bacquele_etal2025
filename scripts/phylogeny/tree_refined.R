#!/usr/bin/env Rscript
# Refined script for circular tree with family-aggregated heatmap
# - Heatmap alpha based on RAW MEAN family proportions per strategy.
# - Branch coloring based on RAW MAX probability (not percentile).
# - Generates spider charts for each family showing RAW mean strategy proportions.
# - Generates a duplicate tree plot annotated with family names.

# --- Configuration ---
# Set working directory to project root (adjust as needed)
# setwd("path/to/project")

# Tree files
tree_files <- c("./output/consensus_pruned_tree_plots/consensus_sumtrees.tre")
tree_names <- c("AllBirdTree")
trait_file <- "./data/model_traits_data.csv"
synonym_file <- "./matching_final_corrected.csv"
Family_col_name <- "family"
output_dir <- "./output/phylogenetic_data/"
output_plot_base_final <- file.path(output_dir, "pruned_circular_tree_family_heatmap_raw_mean_alpha0_v9")
unmatched_species_file <- file.path(output_dir, "unmatched_species_report_AllBirdTree.csv")
species_column_name <- "species"

# --- Create Output Directory ---
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# --- Load Required Packages ---
required_packages <- c("ape", "readr", "dplyr", "tidyr", "ggtree", "ggplot2",
                       "RColorBrewer", "stringr", 'ggtreeExtra', 'ggnewscale',
                       'fmsb', 'scales', 'phytools')

for(pkg in required_packages) {
  if(!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "http://cran.us.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

# --- Helper Function for Ancestral State Reconstruction ---
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
    if(nrow(values_matrix) == length(valid_tip_indices_in_tree)) {
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
      total_child_contributors <- sum(node_child_count[valid_children], na.rm = TRUE)
      
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

# --- Load Data ---
cat("Loading trait data...\n")
model_traits_data <- readr::read_csv(trait_file, show_col_types = FALSE) %>%
  mutate(across(where(is.character), trimws)) %>%
  mutate(!!sym(species_column_name) := stringr::str_replace_all(!!sym(species_column_name), " ", "_"))

family_col_exists <- Family_col_name %in% names(model_traits_data)
if (family_col_exists) {
  cat(paste("Found family column '", Family_col_name, "'.\n", sep = ""))
  if (is.character(model_traits_data[[Family_col_name]])) {
    model_traits_data <- model_traits_data %>%
      mutate(!!sym(Family_col_name) := stringr::str_replace_all(!!sym(Family_col_name), " ", "_")) %>%
      mutate(!!sym(Family_col_name) := stringr::str_replace_all(!!sym(Family_col_name), "[^[:alnum:]_]", "_"))
  }
  model_traits_data <- model_traits_data %>%
    mutate(!!sym(Family_col_name) := ifelse(
      is.na(!!sym(Family_col_name)) | !!sym(Family_col_name) == "",
      "Unknown_Family",
      !!sym(Family_col_name)
    ))
} else {
  stop(paste("Error: Family column '", Family_col_name, "' not found in trait_file.", sep = ""))
}

cat("Loading synonym data...\n")
synonym_data_raw <- readr::read_csv(synonym_file, col_types = readr::cols(.default = "c"), show_col_types = FALSE)
synonym_data <- synonym_data_raw %>%
  mutate(across(everything(), ~stringr::str_replace_all(., " ", "_")))
synonym_rows_list <- apply(synonym_data, 1, function(row) {
  unique(as.character(row[!is.na(row) & row != "" & row != "_"]))
})

desired_strategy_order <- c(3, 7, 4, 2, 1, 6, 5, 0)
prob_cols <- paste0("gmm_prob_", desired_strategy_order, "_mean")
available_prob_cols <- grep("gmm_prob_\\d+_mean$", names(model_traits_data), value = TRUE)
prob_cols <- prob_cols[prob_cols %in% available_prob_cols]

num_clusters <- length(prob_cols)
if (num_clusters == 0) {
  stop("Error: No probability columns (gmm_prob_N_mean) found in trait data.")
}
cat(paste("Detected", num_clusters, "strategy probability columns in order:\n"))
print(prob_cols)

species_in_traits <- unique(model_traits_data[[species_column_name]])

strategy_colors_vec <- c('#e1d314', '#444444', '#0072B2', '#009E73', '#E69F00', '#CC79A7', '#D55E00', '#785EF0')
if (num_clusters > length(strategy_colors_vec)) {
  strategy_colors <- rep_len(strategy_colors_vec, num_clusters)
} else {
  strategy_colors <- strategy_colors_vec[1:num_clusters]
}
strategy_colors_named_for_ggtree <- setNames(strategy_colors, as.character(0:(num_clusters - 1)))
heatmap_strategy_fill_colors <- setNames(strategy_colors, paste0("Strategy ", 0:(num_clusters - 1)))

# --- Process Each Tree ---
all_unmatched_results <- list()

for (i_tree in seq_along(tree_files)) {
  current_tree_file <- tree_files[i_tree]
  current_tree_name <- tree_names[i_tree]
  
  cat(paste("\n===== Processing Tree:", current_tree_name, "(File:", current_tree_file, ") =====\n"))
  
  full_tree <- ape::read.tree(current_tree_file)
  if (inherits(full_tree, "multiPhylo")) {
    full_tree <- full_tree[[1]]
  }
  full_tree$tip.label <- stringr::str_replace_all(full_tree$tip.label, " ", "_")
  
  cleaned_original_tree_labels <- full_tree$tip.label
  
  initial_matches <- intersect(species_in_traits, cleaned_original_tree_labels)
  species_missing_in_tree <- setdiff(species_in_traits, initial_matches)
  tree_labels_still_available <- setdiff(cleaned_original_tree_labels, initial_matches)
  
  synonym_match_results <- list()
  found_via_synonym_count <- 0
  
  if (length(species_missing_in_tree) > 0 && length(tree_labels_still_available) > 0 && length(synonym_rows_list) > 0) {
    available_tree_label_set <- setNames(rep(TRUE, length(tree_labels_still_available)), nm = tree_labels_still_available)
    
    for (idx_s in seq_along(species_missing_in_tree)) {
      missing_species <- species_missing_in_tree[idx_s]
      relevant_row_indices <- which(sapply(synonym_rows_list, function(syn_row) missing_species %in% syn_row))
      
      if (length(relevant_row_indices) > 0) {
        potential_synonyms <- unique(unlist(synonym_rows_list[relevant_row_indices]))
        potential_synonyms <- potential_synonyms[potential_synonyms != "" & !is.na(potential_synonyms) & potential_synonyms != "_"]
        
        for (syn in potential_synonyms) {
          if (syn %in% names(available_tree_label_set) && isTRUE(available_tree_label_set[[syn]])) {
            synonym_match_results[[missing_species]] <- syn
            available_tree_label_set[[syn]] <- FALSE
            found_via_synonym_count <- found_via_synonym_count + 1
            break
          }
        }
      }
    }
  }
  
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
  
  final_tree_labels_to_keep <- unique(mapping_df_this_tree$tree_label)
  
  if (length(final_tree_labels_to_keep) < 3) {
    warning(paste("Skipping plot for", current_tree_name, ": < 3 tips."))
    next
  }
  
  pruned_tree <- ape::keep.tip(full_tree, final_tree_labels_to_keep)
  if (!is.binary(pruned_tree)) {
    pruned_tree <- multi2di(pruned_tree)
  }
  
  min_bl_threshold <- .Machine$double.eps^0.5
  if (any(pruned_tree$edge.length < min_bl_threshold, na.rm = TRUE)) {
    zero_indices <- which(pruned_tree$edge.length < min_bl_threshold)
    min_pos_bl <- min(pruned_tree$edge.length[pruned_tree$edge.length >= min_bl_threshold], na.rm = TRUE)
    small_val <- if (is.finite(min_pos_bl) && min_pos_bl > 0) min_pos_bl * 0.001 else 1e-8
    pruned_tree$edge.length[zero_indices] <- pruned_tree$edge.length[zero_indices] + small_val
  }
  
  # --- Prepare tip probabilities for ASR (RAW, not percentiles) ---
  tip_probabilities_for_asr <- mapping_df_this_tree %>%
    filter(tree_label %in% pruned_tree$tip.label) %>%
    inner_join(model_traits_data %>% select(all_of(species_column_name), all_of(prob_cols)),
               by = c("trait_species" = species_column_name)) %>%
    select(tree_label, all_of(prob_cols)) %>%
    distinct(tree_label, .keep_all = TRUE)
  
  # --- Propagate RAW probabilities and determine dominant cluster by MAX ---
  if (nrow(tip_probabilities_for_asr) > 0 && ncol(tip_probabilities_for_asr) > 1 && length(prob_cols) > 0) {
    
    # Propagate raw probabilities (no percentile conversion)
    node_probabilities <- propagate_values(pruned_tree, tip_probabilities_for_asr, "tree_label", prob_cols, "probabilities")
    
    # Determine dominant cluster based on raw max probability
    dominant_clusters <- apply(node_probabilities, 1, function(probs) {
      if (all(is.na(probs))) return(NA)
      max_indices <- which(probs == max(probs, na.rm = TRUE))
      if (length(max_indices) == 0) return(NA)
      return(min(max_indices) - 1)
    })
    
    node_data <- data.frame(
      node = 1:(length(pruned_tree$tip.label) + pruned_tree$Nnode),
      dominant_cluster = dominant_clusters
    ) %>%
      mutate(dominant_cluster_factor = factor(dominant_cluster, levels = names(strategy_colors_named_for_ggtree)))
    
  } else {
    node_data <- data.frame(
      node = 1:(length(pruned_tree$tip.label) + pruned_tree$Nnode),
      dominant_cluster_factor = factor(NA, levels = names(strategy_colors_named_for_ggtree))
    )
    cat("Not enough data for ASR, branches will not be colored by dominant strategy.\n")
  }
  
  # --- Build base tree plot ---
  p <- ggtree(pruned_tree, layout = 'fan', open.angle = 10, size = 0.25,
              branch.length = 'none', ladderize = TRUE) %<+% node_data +
    aes(color = dominant_cluster_factor) +
    scale_color_manual(
      values = strategy_colors_named_for_ggtree,
      name = "Dominant Strategy\n(Max Tip Probability)",
      guide = guide_legend(keywidth = 0.8, keyheight = 0.8, order = 1),
      na.translate = FALSE,
      drop = FALSE
    )
  
  # --- Prepare Family-Specific Heatmap Data (Using RAW Family Means) ---
  heatmap_data_family_long <- data.frame()
  
  tip_data_for_heatmap_means <- mapping_df_this_tree %>%
    filter(tree_label %in% pruned_tree$tip.label) %>%
    inner_join(model_traits_data %>% select(all_of(species_column_name), all_of(Family_col_name), all_of(prob_cols)),
               by = c("trait_species" = species_column_name)) %>%
    select(tree_label, !!sym(Family_col_name), all_of(prob_cols))
  
  if (nrow(tip_data_for_heatmap_means) == 0) {
    cat("No species data from pruned tree for family mean calculation. Skipping heatmap alpha.\n")
  } else {
    cat("Preparing family-specific heatmap data (using raw family means)...\n")
    
    family_mean_probs <- tip_data_for_heatmap_means %>%
      group_by(!!sym(Family_col_name)) %>%
      summarise(across(all_of(prob_cols), .fns = \(x) mean(x, na.rm = TRUE)), .groups = 'drop') %>%
      filter(rowSums(is.na(select(., all_of(prob_cols)))) < num_clusters)
    
    if (nrow(family_mean_probs) > 0) {
      plot_data_for_heatmap_final <- tip_data_for_heatmap_means %>%
        select(tree_label, !!sym(Family_col_name)) %>%
        distinct() %>%
        left_join(family_mean_probs, by = Family_col_name)
      
      if (nrow(plot_data_for_heatmap_final) > 0 && all(prob_cols %in% names(plot_data_for_heatmap_final))) {
        heatmap_data_family_long <- plot_data_for_heatmap_final %>%
          tidyr::pivot_longer(
            cols = all_of(prob_cols),
            names_to = "StrategyMetric",
            values_to = "AlphaValue"
          ) %>%
          mutate(
            Strategy = factor(
              StrategyMetric,
              levels = prob_cols,
              labels = paste0("Strategy ", 0:(length(prob_cols) - 1))
            )
          ) %>%
          rename(ID = tree_label) %>%
          filter(!is.na(AlphaValue))
      } else {
        cat("Not enough data or columns missing after joining means for family heatmap.\n")
      }
    } else {
      cat("No valid family mean probabilities to use for heatmap.\n")
    }
  }
  
  # --- Add heatmap layer ---
  if (nrow(heatmap_data_family_long) > 0) {
    cat("Adding family-specific heatmap layer to the plot...\n")
    
    desired_data_min_for_alpha <- 0.0
    desired_data_max_for_alpha <- 0.25
    visual_alpha_min <- 0.0
    visual_alpha_max <- 1.0
    
    p <- p +
      new_scale_fill() +
      ggtreeExtra::geom_fruit(
        data = heatmap_data_family_long,
        geom = geom_tile,
        mapping = aes(y = ID, x = Strategy, fill = Strategy, alpha = AlphaValue, height = 1.05),
        color = NA,
        linewidth = 0,
        offset = 0.008,
        pwidth = 0.3,
        grid.params = list(linetype = 0, color = NA)
      ) +
      scale_fill_manual(values = heatmap_strategy_fill_colors, guide = "none", drop = FALSE) +
      scale_alpha_continuous(
        name = paste0("Family Strategy Usage\n(Mean Prop. data range [",
                      desired_data_min_for_alpha, "-", desired_data_max_for_alpha,
                      "]\nmapped to alpha [", visual_alpha_min, "-", visual_alpha_max, "])"),
        range = c(visual_alpha_min, visual_alpha_max),
        limits = c(desired_data_min_for_alpha, desired_data_max_for_alpha),
        oob = scales::squish,
        guide = guide_legend(title.position = "top", title.hjust = 0.5, order = 2),
        na.value = 0
      ) +
      labs(caption = paste0(
        "Branches: Dominant strategy (max tip probability).\n",
        "Heatmap Rings: Color denotes strategy, intensity (alpha) shows mean proportion\n",
        "of the strategy within the family (data range ", desired_data_min_for_alpha, "-",
        desired_data_max_for_alpha, " mapped to full alpha intensity range [",
        visual_alpha_min, "-", visual_alpha_max, "])."
      ))
  } else {
    p <- p + labs(caption = "Branches: Dominant strategy (max tip probability). No family heatmap data.")
  }
  
  # --- Finalize main plot ---
  p_final <- p +
    theme_void() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_blank(),
      panel.background = element_blank(),
      axis.line = element_blank(),
      axis.ticks = element_blank(),
      axis.text = element_blank(),
      axis.title = element_blank(),
      legend.position = "right",
      legend.box = "vertical",
      legend.spacing.y = unit(0.3, "cm"),
      legend.title = element_text(size = 9, face = "bold"),
      legend.text = element_text(size = 8),
      legend.background = element_rect(fill = "white", color = NA),
      legend.key = element_rect(fill = NA, color = NA),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.caption = element_text(size = 8, hjust = 0.5, margin = margin(t = 10)),
      plot.margin = margin(10, 10, 10, 10),
      plot.background = element_rect(fill = "white", color = NA)
    ) +
    ggtitle(paste("Acoustic Strategies & Family Specificity:", current_tree_name))
  
  # --- Save main plot ---
  output_filename_final_png <- paste0(output_plot_base_final, "_", current_tree_name, ".svg")
  output_filename_final_pdf <- paste0(output_plot_base_final, "_", current_tree_name, ".pdf")
  
  cat(paste("Saving refined plot to:", output_filename_final_png, "\n"))
  tryCatch({
    ggsave(output_filename_final_png, plot = p_final, width = 12, height = 11, units = "in", dpi = 600, bg = "white")
  }, error = function(e_save) {
    cat("Error saving PNG:", e_save$message, "\n")
  })
  
  cat(paste("Saving refined plot to:", output_filename_final_pdf, "\n"))
  tryCatch({
    ggsave(output_filename_final_pdf, plot = p_final, width = 12, height = 11, units = "in", device = "pdf", bg = "white")
  }, error = function(e_save) {
    cat("Error saving PDF:", e_save$message, "\n")
  })
  
  # --- Create Duplicate Plot with Family Name Annotations ---
  if (exists("tip_data_for_heatmap_means") && nrow(tip_data_for_heatmap_means) > 0 &&
      Family_col_name %in% names(tip_data_for_heatmap_means)) {
    
    cat("\nGenerating duplicate plot with family name annotations...\n")
    
    family_col_sym <- sym(Family_col_name)
    
    plot_family_annotation_data <- tip_data_for_heatmap_means %>%
      select(tree_label, !!family_col_sym) %>%
      filter(!is.na(!!family_col_sym)) %>%
      distinct(tree_label, .keep_all = TRUE) %>%
      rename(label = tree_label)
    
    common_labels <- intersect(pruned_tree$tip.label, plot_family_annotation_data$label)
    
    if (length(common_labels) == 0 && nrow(plot_family_annotation_data) > 0) {
      cat("ERROR: No common labels found. Cannot annotate families.\n")
      p_annotated_final <- NULL
    } else {
      plot_family_annotation_data_for_merge <- plot_family_annotation_data %>%
        filter(label %in% common_labels)
      
      p_annot_merged_base <- ggtree(pruned_tree, layout = 'fan', open.angle = 10,
                                    ladderize = TRUE, branch.length = 'none', size = 0.25)
      
      if (exists("node_data") && nrow(node_data) > 0 && "dominant_cluster_factor" %in% names(node_data)) {
        p_annot_merged_base <- p_annot_merged_base %<+% node_data
      }
      
      if (nrow(plot_family_annotation_data_for_merge) > 0) {
        p_annot_merged_base <- p_annot_merged_base %<+% plot_family_annotation_data_for_merge
      }
      
      p_built_annot <- p_annot_merged_base
      
      if (exists("node_data") && "dominant_cluster_factor" %in% names(p_built_annot$data)) {
        p_built_annot <- p_built_annot +
          aes(color = dominant_cluster_factor) +
          scale_color_manual(
            values = strategy_colors_named_for_ggtree,
            name = "Dominant Strategy",
            na.translate = FALSE,
            drop = FALSE,
            guide = guide_legend(keywidth = 0.7, keyheight = 0.7, order = 1)
          )
      }
      
      if (Family_col_name %in% names(p_built_annot$data)) {
        p_built_annot <- p_built_annot +
          ggtree::geom_tiplab(
            aes(label = !!family_col_sym),
            align = TRUE,
            linetype = NULL,
            size = 1.5,
            offset = 0.03,
            fontface = 'italic',
            na.rm = TRUE
          )
      }
      
      p_annotated_final <- p_built_annot +
        theme_void() +
        theme(
          legend.position = "right",
          legend.box = "vertical",
          legend.spacing.y = unit(0.2, "cm"),
          legend.title = element_text(size = 8, face = "bold"),
          legend.text = element_text(size = 7),
          legend.background = element_rect(fill = "white", color = NA),
          legend.key = element_rect(fill = NA, color = NA),
          plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
          plot.margin = margin(20, 20, 20, 20),
          plot.background = element_rect(fill = "white", color = NA)
        ) +
        ggtitle(paste("Family-Annotated Tree:", current_tree_name))
    }
    
    if (!is.null(p_annotated_final)) {
      output_filename_annotated_png <- paste0(output_plot_base_final, "_", current_tree_name, "_FAMILY_NAMES.png")
      output_filename_annotated_pdf <- paste0(output_plot_base_final, "_", current_tree_name, "_FAMILY_NAMES.pdf")
      
      cat(paste("Saving family-annotated plot to:", output_filename_annotated_png, "\n"))
      tryCatch({
        ggsave(output_filename_annotated_png, plot = p_annotated_final,
               width = 12, height = 12, units = "in", dpi = 300, bg = "white", limitsize = FALSE)
      }, error = function(e_save) {
        cat("Error saving family-annotated PNG:", e_save$message, "\n")
      })
      
      cat(paste("Saving family-annotated plot to:", output_filename_annotated_pdf, "\n"))
      tryCatch({
        ggsave(output_filename_annotated_pdf, plot = p_annotated_final,
               width = 12, height = 12, units = "in", device = "pdf", bg = "white", limitsize = FALSE)
      }, error = function(e_save) {
        cat("Error saving family-annotated PDF:", e_save$message, "\n")
      })
    }
  }
  
  # --- Generate Spider Charts ---
  cat(paste("\n--- Generating Spider Charts for Families in Tree:", current_tree_name, "---\n"))
  
  spider_chart_dir <- file.path(output_dir, paste0("spider_charts_", current_tree_name))
  dir.create(spider_chart_dir, showWarnings = FALSE, recursive = TRUE)
  
  if (family_col_exists && num_clusters > 0) {
    species_in_pruned_tree_df <- mapping_df_this_tree %>%
      filter(tree_label %in% pruned_tree$tip.label) %>%
      inner_join(model_traits_data %>% select(all_of(species_column_name), all_of(Family_col_name), all_of(prob_cols)),
                 by = c("trait_species" = species_column_name)) %>%
      select(!!sym(Family_col_name), all_of(prob_cols))
    
    if (nrow(species_in_pruned_tree_df) > 0) {
      family_mean_probs_for_spider <- species_in_pruned_tree_df %>%
        group_by(!!sym(Family_col_name)) %>%
        summarise(across(all_of(prob_cols), .fns = \(x) mean(x, na.rm = TRUE)), .groups = 'drop') %>%
        filter(rowSums(is.na(select(., all_of(prob_cols)))) < num_clusters)
      
      if (nrow(family_mean_probs_for_spider) > 0) {
        strategy_spider_labels <- paste0("S", 0:(num_clusters - 1))
        
        for (j_fam in 1:nrow(family_mean_probs_for_spider)) {
          family_name <- family_mean_probs_for_spider[[j_fam, Family_col_name]]
          family_data_values <- as.numeric(family_mean_probs_for_spider[j_fam, prob_cols])
          
          if (all(is.na(family_data_values))) next
          
          data_matrix <- rbind(rep(1, num_clusters), rep(0, num_clusters), family_data_values)
          data_for_radar <- as.data.frame(data_matrix)
          colnames(data_for_radar) <- strategy_spider_labels
          
          safe_family_name <- stringr::str_replace_all(family_name, "[^[:alnum:]_]", "_")
          spider_plot_file <- file.path(spider_chart_dir, paste0("spider_", safe_family_name, ".png"))
          
          png(spider_plot_file, width = 8, height = 8, units = "in", res = 150)
          op <- par(mar = c(1, 2, 2, 1))
          
          tryCatch({
            fmsb::radarchart(
              data_for_radar,
              axistype = 1,
              pcol = strategy_colors[1],
              pfcol = scales::alpha(strategy_colors[1], 0.3),
              plwd = 2,
              plty = 1,
              cglcol = "grey",
              cglty = 1,
              axislabcol = "grey40",
              caxislabels = seq(0, 1, 0.25),
              cglwd = 0.8,
              vlcex = 1.0,
              title = paste("Strategy Profile for Family:", family_name)
            )
          }, error = function(e_rad) {
            cat("Error generating spider chart for", family_name, ":", e_rad$message, "\n")
          })
          
          par(op)
          dev.off()
        }
        cat("Finished generating spider charts for families.\n")
      } else {
        cat("No valid family mean probability data for spider charts.\n")
      }
    } else {
      cat("No species from pruned tree found for spider charts.\n")
    }
  } else {
    cat("Family column missing or no probability columns. Skipping spider charts.\n")
  }
  
  cat(paste("===== Finished Plotting Tree:", current_tree_name, "=====\n"))
}

# --- Generate Unmatched Species Report ---
cat("\n--- Generating Unmatched Species Report ---\n")

if (length(all_unmatched_results) > 0) {
  max_len <- 0
  for (list_name in names(all_unmatched_results)) {
    max_len <- max(max_len, length(all_unmatched_results[[list_name]]))
  }
  
  padded_lists <- list()
  for (list_name in names(all_unmatched_results)) {
    current_list <- all_unmatched_results[[list_name]]
    if (is.null(current_list)) current_list <- character(0)
    length(current_list) <- max_len
    padded_lists[[list_name]] <- current_list
  }
  
  unmatched_df <- as.data.frame(padded_lists)
  if (ncol(unmatched_df) > 0 && all(names(all_unmatched_results) %in% names(unmatched_df))) {
    unmatched_df <- unmatched_df[, names(all_unmatched_results), drop = FALSE]
  }
  
  cat(paste("Saving unmatched species report to:", unmatched_species_file, "\n"))
  tryCatch({
    write.csv(unmatched_df, unmatched_species_file, row.names = FALSE, na = "")
  }, error = function(e_csv) {
    cat("Error saving unmatched species report:", e_csv$message, "\n")
  })
} else {
  cat("No unmatched species results collected.\n")
}

cat("\n--- Script finished successfully! ---\n")