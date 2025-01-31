################################################################################
#                             Data preprocessing
################################################################################

# process gradshap results
process_gradshap <- function(data, model = "DeepSurv") {
  processed_data <- data %>%
    group_by(id, feature) %>%
    summarise(average_value = mean(value, na.rm = TRUE), .groups = "drop") %>%
    group_by(id) %>%
    mutate(rank = rank(-abs(average_value))) %>%
    ungroup()
  
  # Count frequencies of ranks for each feature
  result_data <- processed_data %>%
    group_by(feature, rank) %>%
    summarise(frequency = n(), .groups = "drop") %>%
    mutate(ranking = case_when(
      rank == 1 ~ "1st",
      rank == 2 ~ "2nd",
      rank == 3 ~ "3rd",
      rank == 4 ~ "4th",
      rank == 5 ~ "5th",
      TRUE ~ as.character(rank)
    )) %>%
    select(feature, ranking, frequency)
  
  # Convert ranking to factor
  result_data$ranking <- factor(result_data$ranking, levels = c("5th", "4th", "3rd", "2nd", "1st"))
  
  # Insert model column
  result_data$model <- model
  
  # Return result
  return(result_data)
}


# process survshap results
process_survshap <- function(data, model = "DeepSurv") {
  # Get aggregates
  aggregates <- data$aggregate
  
  # Function to calculate rankings for each element
  calculate_rankings <- function(vec) {
    abs_values <- abs(vec)                    
    ranking <- rank(-abs_values, ties.method = "first") 
    return(ranking)
  }
  
  # Apply the function to each element in the list
  rankings <- lapply(aggregates, calculate_rankings)
  
  # Convert each vector into a data frame and combine them
  rankings_df <- bind_rows(lapply(rankings, function(r) {
    data.frame(feature = names(r), ranking = r)
  }))
  
  # Calculate frequency of each feature for each ranking
  rankings_df <- rankings_df %>%
    group_by(feature, ranking) %>%
    summarise(frequency = n(), .groups = "drop") %>%
    mutate(ranking = case_when(
      ranking == 1 ~ "1st",
      ranking == 2 ~ "2nd",
      ranking == 3 ~ "3rd",
      ranking == 4 ~ "4th",
      ranking == 5 ~ "5th",
      TRUE ~ as.character(ranking)
    ))
  
  # Expand to include all combinations of features and rankings
  result <- expand.grid(
    feature = unique(rankings_df$feature),
    ranking = unique(rankings_df$ranking)
  ) %>%
    left_join(rankings_df, by = c("feature", "ranking")) %>%
    mutate(frequency = replace_na(frequency, 0))
  
  # Convert ranking to factor
  result$ranking <- factor(result$ranking, levels = c("5th", "4th", "3rd", "2nd", "1st"))
  
  # Insert model column
  result$model <- model
  
  
  # Return result
  return(result)
}



################################################################################
#                          Plot functions
################################################################################


# plot global importance
plot_gimp <- function(plot_data) {
  ggplot(plot_data, aes(x = frequency, y = ranking, fill = feature)) +
    geom_bar(stat = "identity", position = "stack") +
    labs(
      title = "",
      x = "",
      y = "Importance ranking"
    ) +
    facet_wrap(vars(model),
               scales = "free_x",
               labeller = as_labeller(function(a)
                 paste0(a))) +
    scale_fill_viridis_d() +
    theme_minimal() +
    theme(
      axis.title.y = element_text(size = 12),
      axis.title.x = element_text(size = 12),
      plot.title = element_text(size = 14, face = "bold"),
      legend.title = element_blank()
    )
}

# Plot survival curves
plot_surv_pred <- function(x, model = "DeepSurv") {
  dat <- as.data.frame(x)
  dat$id <- as.factor(dat$id)
  
  p <- ggplot(dat, aes(x = .data$time)) + NULL +
    geom_line(aes(
      y = .data$pred,
      group = .data$id,
      color = .data$id,
      linetype = .data$id
    )) +
    scale_color_manual(values = c("skyblue", "orange")) +
    theme_minimal(base_size = 16) +
    theme(legend.position = "bottom",
          plot.margin = margin(25, 5, 5, 5)) +
    labs(
      x = "Time",
      y = "Survival Prediction",
      color = "ID",
      linetype = "ID"
    )
  return(p)
}

# Plot attribution curves
plot_attribution <- function(x,
                             normalize = FALSE,
                             normalize_abs = FALSE,
                             add_comp = FALSE,
                             label = "") {
  dat <- as.data.frame(x)
  dat$id <- as.factor(dat$id)
  
  # Normalize values if requested
  if (normalize) {
    dat$value <- ave(
      dat$value,
      dat$id,
      dat$time,
      FUN = function(x)
        x / sum(x)
    )
  }
  
  # Normalize values using absolute values if requested
  if (normalize_abs) {
    dat$value <- ave(
      dat$value,
      dat$id,
      dat$time,
      FUN = function(x)
        abs(x) / sum(abs(x))
    )
  }
  
  # Add comparison curves (reference, prediction, sum, difference)
  if (add_comp) {
    dat$pred_ref <- dat$pred - dat$pred_diff
    
    dat_comp <- pivot_longer(
      dat[,c("id", "time", "feature", "pred", "pred_diff", "pred_ref")],
      cols = c("pred", "pred_diff", "pred_ref"),
      names_to = "Comparison",
      values_to = "value"
    )
    
    comp <- geom_line(
      data = dat_comp,
      aes(
        x = time,
        y = value,
        group = Comparison,
        linetype = Comparison
      ),
      color = "grey",
      linewidth = 1
    )
  } else {
    comp <- NULL
  }
  
  p <- ggplot(dat, aes(x = .data$time)) + NULL +
    geom_line(aes(
      y = .data$value,
      group = .data$feature,
      color = .data$feature
    ), na.rm = TRUE) +
    geom_point(aes(
      y = .data$value,
      group = .data$feature,
      color = .data$feature
    ),
    na.rm = TRUE,
    size = 0.75) +
    comp +
    geom_hline(yintercept = 0, color = "black") +
    facet_wrap(vars(.data$id),
               scales = "free_x",
               labeller = as_labeller(function(a)
                 paste0("Instance ID: ", a))) +
    theme_minimal(base_size = 16) +
    theme(legend.position = "bottom") +
    labs(
      x = "Time",
      y = paste0("Attribution S(t|x): ", label),
      color = "Feature",
      linetype = NULL
    ) +
    scale_linetype_manual("Comparison",
                          values = c("solid", "dotdash", "dotted")) +
    scale_color_viridis_d()
  return(p)
}

# Plot absolute contribution of features
plot_contribution <- function(x, scale = 0.7, aggregate = FALSE, label = "") {
  # Convert input to a data frame and calculate derived columns
  dat <- as.data.frame(x)
  
  # Aggregate values if requested
  if (aggregate) {
    dat <- dat %>%
      group_by(time, feature) %>%
      summarise(value = mean(abs(value)), pred = mean(pred), pred_diff = mean(pred_diff), method = unique(method), .groups = "drop")
    dat$id <- "Aggregated"
  }
  
  dat$id <- as.factor(dat$id)
  dat$pred_ref <- dat$pred - dat$pred_diff
  integer_times <- seq(from = ceiling(min(dat$time)), to = floor(max(dat$time)), by = 2)
  
  # Process data to compute ratios, cumulative ratios, and ymin
  dat <- dat %>%
    group_by(id, time) %>%
    mutate(sum = sum(abs(value)),
           ratio = abs(value) / abs(sum) * 100) %>%
    arrange(id, time, desc(feature)) %>%
    group_by(id, time) %>%
    mutate(cumulative_ratio = cumsum(ratio)) %>%
    arrange(id, time, feature) %>%
    group_by(id, time) %>%
    mutate(ymin = lead(cumulative_ratio, default = 0))
  
  # Obtain average ratio over features and positions for the barchart
  avg_contribution <- dat %>%
    group_by(id, feature) %>%
    summarise(mean_ratio = round(mean(ratio),2), .groups = "drop") %>%
    group_by(id) %>%
    mutate(pos = rev(cumsum(rev(mean_ratio)))*scale)
  
  # Generate the plot
  p <- ggplot(dat, aes(x = .data$time)) +
    # Line plot for cumulative_ratio
    geom_line(aes(y = .data$cumulative_ratio, group = .data$feature),
              color = "black") +
    # Ribbon for cumulative_ratio
    geom_ribbon(
      aes(
        ymin = .data$ymin,
        ymax = .data$cumulative_ratio,
        group = .data$feature,
        fill = .data$feature
      ),
      alpha = 0.4
    ) +
    # Bar plot for mean_ratio
    geom_bar(
      data = avg_contribution, # Unique rows for bar plot
      aes(
        x = max(dat$time) + 1, # Offset x to place bars next to lines
        y = mean_ratio,
        fill = feature
      ),
      stat = "identity",
      alpha = 0.6,
      width = 0.8
    ) +
    # Add percentage labels
    geom_text(
      data = avg_contribution,
      aes(
        x = max(dat$time) + 1,
        y = pos,
        label = paste0(round(mean_ratio, 1), "%")
      ),
      color = "black",
      size = 3
    ) +
    # Facet for each instance ID
    facet_wrap(vars(.data$id),
               scales = "free_x",
               labeller = as_labeller(function(a) {
                 if (aggregate) {
                   "Global"
                 } else {
                   paste0("Instance ID: ", a)
                 }
               })) +
    # Minimal theme
    theme_minimal(base_size = 16) +
    theme(legend.position = "bottom") +
    # Suppress x-axis tick label for bar offset
    scale_x_continuous(
      breaks = integer_times,
      labels = integer_times
    ) +
    # Labels
    labs(
      x = "Time",
      y = paste0("% Contribution: ", label),
      color = "Feature",
      fill = "Feature"
    ) +
    scale_fill_viridis_d(alpha = 0.4)
  
  return(p)
}

# Force plot for contributions
plot_force <- function(x,
                       num_samples = 10,
                       zero_feature = "x3",
                       pos_neg = FALSE,
                       upper_distance = 0.02,
                       lower_distance = 0.02,
                       lower_distance_x1 = 0,
                       intgrad0_td_cox = FALSE,
                       intgrad0_td_deepsurv = FALSE, 
                       intgradmean_td_cox = FALSE,
                       intgradmean_td_deephit = FALSE, 
                       intgradmean_td_deepsurv = FALSE, 
                       gradshap_td_cox = FALSE,
                       gradshap_td_deephit = FALSE,
                       gradshap_td_deepsurv = FALSE,
                       label = "") {
  # Convert input to a data frame and calculate derived columns
  dat <- as.data.frame(x)
  dat$id <- as.factor(dat$id)
  dat$pred_ref <- dat$pred - dat$pred_diff
  
  # Process data to compute sum of attributions
  dat <- dat %>%
    group_by(id, time) %>%
    mutate(sum = sum(value))
  
  # Sample time points for visualization
  t_interest <- sort(unique(dat$time))
  target_points <- seq(min(t_interest), max(t_interest), length.out = num_samples)
  selected_points <- sapply(target_points, function(x)
    t_interest[which.min(abs(t_interest - x))])
  dat_small <- dat[dat$time %in% selected_points, ]
  
  
  # Create position variable for plotting attribution values
  dat_small <- dat_small %>%
    group_by(id, time) %>%
    mutate(
      pos = case_when(
        feature == "x4" ~ NA,
        # For x4, pos is 0.05
        feature == "x3" ~ value,
        # For x3, pos is value of x3
        feature == "x2" &
          sign(value) == sign(value[feature == "x3"]) ~ value + value[feature == "x3"],
        # For x2, check sign with x3
        feature == "x2" &
          sign(value) != sign(value[feature == "x3"]) ~ value,
        # For x2, if signs differ from x3
        feature == "x1" &
          sign(value) == sign(value[feature == "x2"]) &
          sign(value) == sign(value[feature == "x3"]) ~ value + value[feature == "x2"] + value[feature == "x3"],
        # For x1, if sign matches both x2 and x3
        feature == "x1" &
          sign(value) == sign(value[feature == "x2"]) ~ value + value[feature == "x2"],
        # For x1, if sign matches x2
        feature == "x1" &
          sign(value) == sign(value[feature == "x3"]) ~ value + value[feature == "x3"],
        # For x1, if sign matches x3
        feature == "x1" &
          sign(value) != sign(value[feature == "x2"]) &
          sign(value) != sign(value[feature == "x3"]) ~ value,
        # For x1, if sign doesn't match x2 or x3
        TRUE ~ NA_real_
      )
    ) %>%
    ungroup()
  
  # Additional position variable for the arrows
  dat_small$pos_a <- ifelse(dat_small$pos > 0, dat_small$pos + 0.04, dat_small$pos)
  dat_small$pos_a <- ifelse(dat_small$pos_a < 0, dat_small$pos_a - 0.04, dat_small$pos_a)
  
  # Adjustments for improved plotting of attribution values
  dat_small$pos <- ifelse((dat_small$pos <= 0.025) &
                            (dat_small$feature == "x2"),
                          dat_small$pos - lower_distance,
                          dat_small$pos
  )
  dat_small$pos <- ifelse((dat_small$pos >= 0.025) &
                            (dat_small$pos <= 0.08) &
                            (dat_small$feature == "x1"),
                          dat_small$pos + upper_distance,
                          dat_small$pos
  )
  dat_small$pos <- ifelse((dat_small$value >= -0.04) &
                            (dat_small$value < 0) &
                            (dat_small$feature == "x1"),
                          dat_small$pos - lower_distance_x1,
                          dat_small$pos
  )
  dat_small$pos <- ifelse((dat_small$value <= 0.025) &
                            (dat_small$value > 0) &
                            (dat_small$feature == "x1"),
                          dat_small$pos + lower_distance_x1,
                          dat_small$pos
  )
  dat_small$pos <- ifelse((dat_small$value > 0) &
                            (dat_small$value < 0.04) &
                            (dat_small$feature == "x1"),
                          dat_small$pos + lower_distance_x1,
                          dat_small$pos
  )
  
  # Adjustments for intgrad0_td_cox
  if (intgrad0_td_cox) {
    dat_small[(dat_small$id == 79) &
                (round(dat_small$time, 2) == 0.99) &
                (dat_small$feature == "x2"), "pos"] <- 0.05
    dat_small[(dat_small$id == 428) &
                (round(dat_small$time, 1) == 1.7) &
                (dat_small$feature == "x2"), "pos"] <- 0.14
  }
  
  # Adjustments for intgrad0_td_deepsurv
  if (intgrad0_td_deepsurv) {
    dat_small[((round(dat_small$time, 1) == 1.6) |
                 (round(dat_small$time, 1) == 7)) &
                (dat_small$feature == "x1") &
                (dat_small$id == "79"), "pos"] <- -0.1
    dat_small[((round(dat_small$time, 1) == 0.8) |
                 (round(dat_small$time, 1) == 7)) &
                (dat_small$feature == "x1") &
                (dat_small$id == "428"), "pos"] <- -0.1
    dat_small[((round(dat_small$time, 1) == 0.8) |
                 (round(dat_small$time, 1) == 7)) &
                (dat_small$feature == "x2") &
                (dat_small$id == "428"),"pos"] <- 0
    dat_small[((round(dat_small$time, 1) == 1.6) |
                 (round(dat_small$time, 1) == 7)) &
                (dat_small$feature == "x2") &
                (dat_small$id == "79"),"pos"] <- 0
    dat_small[(round(dat_small$time, 1) == 7) &
                (dat_small$feature == "x3") &
                (dat_small$id == "79"),"pos"] <- 0.1
    dat_small[(round(dat_small$time, 1) == 0.8) &
                (dat_small$feature == "x3") &
                (dat_small$id == "428"),"pos"] <- 0.1
  }
  
  if (intgradmean_td_cox) {
    dat_small[(round(dat_small$time, 1) == 1.7) &
                (dat_small$feature == "x2") &
                (dat_small$id == "79"), "pos"] <- 0.04
    dat_small[(round(dat_small$time, 1) == 5.6) &
                (dat_small$feature == "x1") &
                (dat_small$id == "79"), "pos"] <- 0.07
    dat_small[(round(dat_small$time, 1) == 5.6) &
                (dat_small$feature == "x3") &
                (dat_small$id == "79"), "pos"] <- -0.07
    dat_small[(round(dat_small$time, 2) == 0.99) &
                (dat_small$feature == "x1") &
                (dat_small$id == "428"), "pos"] <- 0.07
    dat_small[(round(dat_small$time, 1) == 3.3) &
                (dat_small$feature == "x1") &
                (dat_small$id == "428"), "pos"] <- -0.25
    dat_small[(round(dat_small$time, 1) == 4) &
                (dat_small$feature == "x1") &
                (dat_small$id == "428"), "pos"] <- -0.11
    dat_small[(round(dat_small$time, 1) == 4) &
                (dat_small$feature == "x3") &
                (dat_small$id == "428"), "pos"] <- -0.05
    dat_small[(round(dat_small$time, 1) == 4) &
                (dat_small$feature == "x2") &
                (dat_small$id == "428"), "pos"] <- 0.03
  }
  
  if (intgradmean_td_deephit) {
    dat_small[(round(dat_small$time, 1) == 2.4) &
                (dat_small$feature == "x1") &
                (dat_small$id == "79"), "pos"] <- 0.05
    dat_small[(round(dat_small$time, 1) > 6) &
                (dat_small$feature == "x3") &
                (dat_small$id == "79"), "pos"] <- -0.03
    dat_small[(round(dat_small$time, 1) == 1.4) &
                (dat_small$feature == "x1") &
                (dat_small$id == "428"), "pos"] <- 0.05
    dat_small[(round(dat_small$time, 1) > 5) &
                (dat_small$feature == "x3") &
                (dat_small$id == "428"), "pos"] <- 0
    dat_small[(round(dat_small$time, 1) > 5) &
                (dat_small$feature == "x1") &
                (dat_small$id == "428"), "pos"] <- -0.03
  }
  
  if (intgradmean_td_deepsurv){
    dat_small[((round(dat_small$time, 1) == 1.6) | 
                 (round(dat_small$time, 1) == 5.5)) &
                (dat_small$feature == "x1") &
                (dat_small$id == "79"), "pos"] <- 0.07
    dat_small[((round(dat_small$time, 1) == 1.6) | 
                 (round(dat_small$time, 1) == 5.5)) &
                (dat_small$feature == "x3") &
                (dat_small$id == "79"), "pos"] <- -0.06
    dat_small[(round(dat_small$time, 1) == 0.8) &
                (dat_small$feature == "x2") &
                (dat_small$id == "428"), "pos"] <- 0.03
    dat_small[(round(dat_small$time, 1) == 1.6) &
                (dat_small$feature == "x1") &
                (dat_small$id == "428"), "pos"] <- -0.4
    dat_small[(round(dat_small$time, 1) == 4.1) &
                (dat_small$feature == "x2") &
                (dat_small$id == "428"), "pos"] <- 0.02
    dat_small[(round(dat_small$time, 1) == 4.1) &
                (dat_small$feature == "x3") &
                (dat_small$id == "428"), "pos"] <- -0.04
    dat_small[(round(dat_small$time, 1) == 4.1) &
                (dat_small$feature == "x1") &
                (dat_small$id == "428"), "pos"] <- -0.1
  }
  
  if (gradshap_td_cox) {
    dat_small[(round(dat_small$time, 1) > 4) &
                (dat_small$feature == "x3") , "pos"] <- dat_small[(round(dat_small$time, 1) > 4) &
                                                                    (dat_small$feature == "x3") , "pos"] + 0.04
    
    dat_small[(round(dat_small$time, 1) == 4) &
                (dat_small$feature == "x3") &
                (dat_small$id == "428"), "pos"] <- dat_small[(round(dat_small$time, 1) == 4) &
                                                               (dat_small$feature == "x3") &
                                                               (dat_small$id == "428"), "pos"] + 0.04
    dat_small[(round(dat_small$time, 1) == 1) &
                (dat_small$feature == "x1") &
                (dat_small$id == "428"), "pos"] <- 0.08
    dat_small[(round(dat_small$time, 1) == 1) &
                (dat_small$feature == "x2") &
                (dat_small$id == "79"), "pos"] <- 0.12
    dat_small[(round(dat_small$time, 1) == 1.7) &
                (dat_small$feature == "x2") &
                (dat_small$id == "428"), "pos"] <- 0.07
  }
  
  if (gradshap_td_deephit){
    dat_small[(round(dat_small$time, 1) > 4.5) &
                (dat_small$feature == "x2") &
                (dat_small$id == "79"), "pos"] <- dat_small[(round(dat_small$time, 1) > 4.5) &
                                                              (dat_small$feature == "x2") &
                                                              (dat_small$id == "79"), "pos"] - 0.02
    dat_small[(round(dat_small$time, 1) < 2) &
                (dat_small$feature == "x2") &
                (dat_small$id == "79"), "pos"] <- dat_small[(round(dat_small$time, 1) < 2) &
                                                              (dat_small$feature == "x2") &
                                                              (dat_small$id == "79"), "pos"] + 0.03
    dat_small[(round(dat_small$time, 1) > 4) &
                (dat_small$feature == "x2") &
                (dat_small$id == "428"), "pos"] <- dat_small[(round(dat_small$time, 1) > 4) &
                                                               (dat_small$feature == "x2") &
                                                               (dat_small$id == "428"), "pos"] - 0.02
    dat_small[(round(dat_small$time, 1) > 4) &
                (dat_small$feature == "x1") &
                (dat_small$id == "428"), "pos"] <- dat_small[(round(dat_small$time, 1) > 4) &
                                                               (dat_small$feature == "x1") &
                                                               (dat_small$id == "428"), "pos"] - 0.01
  }
  
  if (gradshap_td_deepsurv) {
    dat_small[(round(dat_small$time, 1) > 4) &
                (dat_small$feature == "x3") , "pos"] <- dat_small[(round(dat_small$time, 1) > 4) &
                                                                    (dat_small$feature == "x3") , "pos"] + 0.03
    dat_small[(round(dat_small$time, 1) == 3.1) &
                (dat_small$feature == "x3") &
                (dat_small$id == 428), "pos"] <- dat_small[(round(dat_small$time, 1) == 3.1) &
                                                             (dat_small$feature == "x3") &
                                                             (dat_small$id == 428), "pos"] + 0.03
    dat_small[(round(dat_small$time, 1) == 1.6) &
                (dat_small$feature == "x3") &
                (dat_small$id == 428), "pos"] <- dat_small[(round(dat_small$time, 1) == 1.6) &
                                                             (dat_small$feature == "x3") &
                                                             (dat_small$id == 428), "pos"] + 0.03
    dat_small[(round(dat_small$time, 1) == 0.8) &
                (dat_small$feature == "x1") &
                (dat_small$id == 428), "pos"] <- 0.04
    dat_small[(round(dat_small$time, 1) == 0.8) &
                (dat_small$feature == "x3") &
                (dat_small$id == 428), "pos"] <- -0.04
    dat_small[(round(dat_small$time, 1) == 0.8) &
                (dat_small$feature == "x2") &
                (dat_small$id == 428), "pos"] <- 0
    dat_small[(round(dat_small$time, 1) == 0.8) &
                (dat_small$feature == "x2") &
                (dat_small$id == 79), "pos"] <- 0.06
    dat_small[(round(dat_small$time, 1) == 0.8) &
                (dat_small$feature == "x3") &
                (dat_small$id == 79), "pos"] <- 0.01
    dat_small[(round(dat_small$time, 1) == 1.6) &
                (dat_small$feature == "x3") &
                (dat_small$id == 79), "pos"] <- 0.06
    dat_small[(round(dat_small$time, 1) == 1.6) &
                (dat_small$feature == "x2") &
                (dat_small$id == 79), "pos"] <- 0.11
    dat_small[(round(dat_small$time, 1) == 1.6) &
                (dat_small$feature == "x1") &
                (dat_small$id == 79), "pos"] <- 0.16
    dat_small[(round(dat_small$time, 1) == 0.8) &
                (dat_small$feature == "x3") &
                (dat_small$id == 428), "pos"] <- -0.04
    dat_small[(round(dat_small$time, 1) == 0.8) &
                (dat_small$feature == "x1") &
                (dat_small$id == 428), "pos"] <- 0.05
  }
  
  # Add sign legend
  dat_small$sign <- ifelse(dat_small$value > 0, "Positive", "Negative")
  dat_small$sign <- factor(dat_small$sign, levels = c("Positive", "Negative"))
  
  # Plot
  if (pos_neg) {
    p <- ggplot() +
      geom_line(
        data = dat,
        mapping = aes(x = .data$time, y = .data$sum),
        color = "black"
      ) +
      geom_bar(
        data = dat_small,
        mapping = aes(
          x = .data$time,
          y = .data$value,
          fill = .data$feature,
          color = .data$feature
        ),
        stat = "identity",
        position = "stack"
      ) +
      scale_color_viridis_d(name = "Feature") +
      scale_fill_viridis_d(alpha = 0.4, name = "Feature") +
      new_scale_color() +
      geom_segment(
        data = dat_small[(dat_small$feature != zero_feature) &
                           (round(dat_small$value, 2) != 0), ],
        mapping = aes(
          x = .data$time,
          xend = .data$time,
          y = .data$pos_a,
          yend = .data$pos_a + (.data$value) * 0.01,
          color = .data$sign  # Color arrows based on the sign
        ),
        arrow = arrow(type = "closed", length = unit(0.1, "inches")),
        size = 6
      ) +
      geom_label(
        data = dat_small[round(dat_small$value, 2) != 0, ],
        mapping = aes(
          x = .data$time,
          y = .data$pos,
          label = round(.data$value, 2)
        ),
        color = "black",
        size = 3,
        vjust = 0.5,
        hjust = 0.5,
        na.rm = TRUE
      ) +
      facet_wrap(vars(.data$id),
                 scales = "free_x",
                 labeller = as_labeller(function(a)
                   paste0("Instance ID: ", a))) +
      theme_minimal(base_size = 16) +
      theme(legend.position = "bottom") +
      labs(
        x = "Time",
        y = paste0("Contribution: ", label),
        color = "Feature",
        fill = "Feature"
      ) +
      scale_color_manual(
        values = c(
          "Positive" = "darkgreen",
          "Negative" = "darkred"
        ),
        name = "Sign"
      )
    
  } else {
    p <- ggplot() +
      geom_line(
        data = dat,
        mapping = aes(x = .data$time, y = .data$sum),
        color = "black"
      ) +
      geom_bar(
        data = dat_small,
        mapping = aes(
          x = .data$time,
          y = .data$value,
          fill = .data$feature,
          color = .data$feature
        ),
        stat = "identity",
        position = "stack"
      ) +
      scale_color_viridis_d(name = "Feature") +
      scale_fill_viridis_d(alpha = 0.4, name = "Feature") +
      geom_segment(
        data = dat_small[(dat_small$feature != zero_feature) &
                           (round(dat_small$value, 2) != 0), ],
        mapping = aes(
          x = .data$time,
          xend = .data$time,
          y = .data$pos_a,
          yend = .data$pos_a + (.data$value) * 0.01,
          color = .data$feature
        ),
        arrow = arrow(type = "closed", length = unit(0.1, "inches")),
        linewidth = 6
      ) +
      geom_label(
        data = dat_small[round(dat_small$value, 2) != 0, ],
        mapping = aes(
          x = .data$time,
          y = .data$pos,
          label = round(.data$value, 2)
        ),
        color = "black",
        size = 3,
        vjust = 0.5,
        hjust = 0.5,
        na.rm = TRUE
      ) +
      facet_wrap(vars(.data$id),
                 scales = "free_x",
                 labeller = as_labeller(function(a)
                   paste0("Instance ID: ", a))) +
      theme_minimal(base_size = 16) +
      theme(legend.position = "bottom") +
      labs(
        x = "Time",
        y = paste0("Contribution: ", label),
        color = "Feature",
        fill = "Feature"
      )
  }
  
  return(p)
}
