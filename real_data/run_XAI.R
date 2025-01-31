library(Survinng)
library(torch)
library(torchvision)
library(ggplot2)
library(data.table)
library(dplyr)
library(here)

set.seed(42)
torch_manual_seed(42)



# Define multi modal model
MultiModalModel <- nn_module(
  "MultiModalModel",
  initialize = function(net_images, tabular_features, n_out = 1, n_img_out = 64,
                        include_time = FALSE, out_bias = FALSE) {
    self$net_images <- net_images
    
    input_dim <- n_img_out + length(tabular_features) + ifelse(include_time, 1, 0)
    self$fc1 <- nn_linear(input_dim, 256)
    self$fc2 <- nn_linear(256, 128)
    self$relu <- nn_relu()
    self$drop_0_3 <- nn_dropout(0.3)
    self$drop_0_4 <- nn_dropout(0.3)
    self$out <- nn_linear(128, n_out, bias = out_bias)
  },
  
  forward = function(input, time = NULL) {
    img <- input[[1]]
    tab <- input[[2]]
    img <- self$net_images(img)
    img <- self$drop_0_4(img)
    
    if (!is.null(time)) {
      x <- torch_cat(list(img, tab, time), dim = 2)
    } else {
      x <- torch_cat(list(img, tab), dim = 2)
    }
    
    x <- self$drop_0_3(x)
    x <- self$relu(self$fc1(x))
    x <- self$drop_0_3(x)
    x <- self$relu(self$fc2(x))
    #x <- self$drop_0_3(x)
    x <- self$out(x)
    
    x
  }
)


# Load model--------------------------------------------------------------------

# Load model metadata
model_metadata <- read.csv(here("real_data_new/results/deephit/metadata.csv"))
n_img_out <- model_metadata$n_img_out
n_out <- model_metadata$n_out
out_bias <- as.logical(model_metadata$out_bias)
n_tab_feat <- model_metadata$Number.of.tabular.features

# Load model state dict
model_state_dict <- load_state_dict(here("real_data_new/results/deephit/model.pt"))

# Replicate model
net_image <- torchvision::model_resnet34(num_classes = n_img_out)
model <- MultiModalModel(net_image, tabular_features = rep(1, n_tab_feat),
                         n_out = n_out, n_img_out = n_img_out, out_bias = out_bias)

# Load model state dict
model <- model$load_state_dict(model_state_dict)
model$eval()

# Load data to be explained-----------------------------------------------------
data <- read.csv(here("real_data_new/data/deephit_tabular_data.csv"))
test_images <- list.files(here("real_data_new/data/test"), full.names = FALSE)

# Filter data
data <- data[data$full_path %in% test_images, ]
data_tab <- torch_tensor(as.matrix(data[, c(1, 2, 3, 4)]))
data_img <- torch_stack(lapply(data$full_path, function(x) {
  base_loader(paste0("real_data_new/data/test/", x)) %>%
    (function(x) x[,,1:3]) %>%
    transform_to_tensor() %>%
    transform_center_crop(226) %>%
    transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
}), dim = 1)


# Create force plot
plot_force_real <- function(x, num_samples = 10, zero_feature = "Sex (male)", label = "") {
  # Convert input to a data frame and calculate derived columns
  dat <- as.data.frame(x)
  dat$id <- as.factor(dat$id)
  dat$feature <- factor(
    dat$feature,
    levels = c(
      "IDH Mutation (mutant)",
      "Image (sum)",
      "1p19q Codeletion (no)",
      "Age (43)",
      "Sex (male)"
    )
  )
  
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
        feature == "Sex (male)" ~ NA,
        feature == "Age (43)" ~ value + value[feature == "Sex (male)"],
        feature == "1p19q Codeletion (no)"  ~ value,
        feature == "IDH Mutation (mutant)" ~ value + value[feature == "Age (43)"] + value[feature == "Sex (male)"],
        #feature == "IDH Mutation (mutant)" &
        #  sign(value[feature == "Image (sum)"]) < 0 ~ value + value[feature == "Age (43)"] + value[feature == "Sex (male)"],
        #feature == "Image (sum)" &
        #  sign(value) < 0 ~ value,
        feature == "Image (sum)" ~ value + value[feature == "1p19q Codeletion (no)"],
        TRUE ~ NA_real_
      )
    ) %>%
    ungroup()
  
  # Additional position variable for the arrows
  dat_small$pos_a <- ifelse(dat_small$pos > 0, dat_small$pos + 0.03, dat_small$pos)
  dat_small$pos_a <- ifelse(dat_small$pos_a < 0, dat_small$pos_a - 0.03, dat_small$pos_a)
  
  # Adjust label position manually
  # dat_small[(round(dat_small$time, 2) == 15.21) &
  #             (dat_small$feature == "Image (sum)"), "pos"] <- -0.011
  dat_small[(round(dat_small$value, 2) == 0.01), "pos"] <- 0.018
  
  # Plot
  p <- ggplot() +
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
    scale_color_viridis_d(name = "Feature", guide = guide_legend(override.aes = list(linewidth = 7))) +
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
    geom_line(
      data = dat,
      mapping = aes(x = .data$time, y = .data$sum),
      color = "black",
      linewidth = 1.5
    ) +
    facet_wrap(vars(.data$id),
               scales = "free_x",
               labeller = as_labeller(function(a)
                 paste0("Instance ID: ", a))) +
    theme_minimal(base_size = 13) +
    theme(legend.position = "bottom") +
    ylim(-0.15, 0.16) +
    scale_x_continuous(expand = c(0,0)) +
  labs(
    x = "Time",
    y = paste0("Force Plot: ", label),
    color = "Feature",
    fill = "Feature"
  )
  
  return(p)
}

# Plot function
plot_bars_real <- function(x, num_samples = 40, zero_feature = "Sex (male)", label = "") {
  # Convert input to a data frame and calculate derived columns
  dat <- as.data.frame(x)
  dat$id <- as.factor(dat$id)
  dat$feature <- factor(
    dat$feature,
    levels = c(
      "IDH Mutation (mutant)",
      "Image (sum)",
      "1p19q Codeletion (no)",
      "Age (43)",
      "Sex (male)"
    )
  )
  
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
  
  # Plot
  p <- ggplot() +
    geom_line(
      data = dat,
      mapping = aes(x = .data$time, y = .data$sum),
      color = "black",
      linewidth = 1
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
    facet_wrap(vars(.data$id),
               scales = "free_x",
               labeller = as_labeller(function(a)
                 paste0("Instance ID: ", a))) +
    theme_minimal(base_size = 15) +
    theme(legend.position = "bottom") +
    labs(
      x = "Time",
      y = paste0("Contribution: ", label),
      color = "Feature",
      fill = "Feature"
    )
  
  return(p)
}

plot_result <- function(result, img, path = NULL, name = "res", num_images = 7, as_force = TRUE) {
  df_img <- result[[1]]
  df_tab <- result[[2]]
  
  # Get prediction data.table
  col_idx <- colnames(df_tab)[colnames(df_tab) %in% c("time", "pred", "pred_diff")]
  df_pred <- unique(df_tab[, ..col_idx])
  
  # Summarize image features
  df_img_sum <- df_img %>%
    group_by(id, time, method) %>%
    summarise(value = sum(value))
  
  # Add image summary to table
  df_tab <- rbind(df_tab,
                  cbind(df_img_sum, feature = "Image (sum)"), fill = TRUE)
  df_tab$feature <- factor(df_tab$feature, levels = unique(df_tab$feature),
                           labels = c("Age (43)", "Sex (male)", "IDH Mutation (mutant)", "1p19q Codeletion (no)", "Image (sum)"))
  
  # Plot force plot ------------------------------------------------------------
  if (as_force) {
    p_force <- plot_force_real(df_tab, label = "GradSHAP(t)", num_samples = 20)
  } else {
    p_force <- NULL
  }
  
  # Plot bar plot --------------------------------------------------------------
  p_bar <- plot_bars_real(df_tab, label = "GradSHAP(t)")
  
  # Plot as lines
  if ("pred_diff" %in% colnames(df_pred)) {
    df_pred$pred <- df_pred$pred_diff
  }
  p_line <- ggplot(df_tab, aes(x = time, y = value, color = feature)) +
    geom_line(linewidth = 1) +
    geom_line(data = df_pred, aes(y = pred), color = "black", linewidth = 1) +
    geom_hline(yintercept = 0, linetype = "dashed", linewidth = 1) +
    facet_wrap(vars(id), scales = "free_x") +
    scale_color_viridis_d() +
    theme_minimal(base_size = 13) +
    theme(legend.position = "bottom",
          legend.box.margin = margin(),
          plot.margin = margin()) +
    labs(x = "Time (months)", y = NULL, color = "Feature")
  
  # Aggregate over channels
  df_img <- df_img %>%
    group_by(id, time, height, width) %>%
    summarise(value = sum(value), pred = unique(pred))
  
  # Normalize over time
  fun <- function(x) {
    q1 <- quantile(x, 0.005)
    q2 <- quantile(x, 0.995)
    pmax(pmin(x, q2), q1)
  }
  
  df_img <- df_img %>%
    group_by(id, time) %>%
    mutate(value = fun(value))
  
  # Plot image explanation
  time_bins <- unique(df_img$time)
  times <- as.integer(seq(0, length(time_bins), length.out = num_images + 2))[-c(1, num_images + 2)]
  time_bins <- time_bins[times]
  
  df <- df_img[df_img$time %in% time_bins, ]
  p_img_exp <- ggplot() +
    geom_tile(aes(x = width, y = height, fill = value), data = df) +
    facet_grid(cols = vars(time)) +
    scale_x_discrete(expand = c(0,0)) +
    scale_y_discrete(expand = c(0,0)) +
    scale_fill_gradient2(low = "blue", mid = "white", 
                         high = "red", transform = "pseudo_log") +
    theme_minimal() +
    guides(fill = "none") +
    labs(x = NULL, y = NULL) +
    theme(axis.text = element_blank(),
          plot.margin = margin(0, 0, 0, 0),
          strip.text = element_blank())
  
  # Load image
  library(ggmap)
  p_img <- ggimage(img)
  
  # Save figures
  if (!is.null(path)) {
    if (!dir.exists(dirname(path))) {
      dir.create(dirname(path))
    }
    ggsave(p_bar + scale_x_continuous(expand = c(0,0)), filename = paste0(path, name, "_bar.pdf"), width = 10, height = 5)
    ggsave(p_img_exp, filename = paste0(path, name, "_img_exp.pdf"), width = 7, height = 1)
    ggsave(p_img, filename = paste0(path, name, "_img_exp.pdf"), width = 1, height = 1)
    if (!is.null(p1_force)) {
      ggsave(p_force, filename = paste0(path, name, "_force.pdf"), width = 9, height = 5)
    }
  }
  
  list(p1, p2, p_img, p1_force)
}


# Explain model-----------------------------------------------------------------
torch_set_num_threads(30L)
exp_deephit <- Survinng::explain(model, list(data_img, data_tab), model_type = "deephit",
                       time_bins = seq(0, 17, length.out = n_out))

ids <- 230 # which(rownames(data) %in% c(1428, 789))
orig_img <- base_loader(paste0("data/test/", data[ids, ]$full_path))

grad <- surv_grad(exp_deephit, instance = ids, batch_size = 4000) |>
  as.data.table()
sgrad <- surv_smoothgrad(exp_deephit, instance = ids, n = 100, 
                         noise_level = 0.3, batch_size = 1000)  |>
  as.data.table()
intgrad <- surv_intgrad(exp_deephit, instance = ids, n = 100, batch_size = 1000)  |>
  as.data.table()
grad_shap <- surv_gradSHAP(exp_deephit, instance = ids, n = 20, num_samples = 100, 
                           batch_size = 1000, replace = FALSE)  |>
  as.data.table()


# Save results -----------------------------------------------------------------
saveRDS(grad, "results/deephit/grad.rds")
saveRDS(sgrad, "results/deephit/sgrad.rds")
saveRDS(intgrad, "results/deephit/intgrad.rds")
saveRDS(grad_shap, "results/deephit/grad_shap.rds")

# Read results -----------------------------------------------------------------
grad <- readRDS(here("real_data_new/results/deephit/grad.rds"))
sgrad <- readRDS(here("real_data_new/results/deephit/sgrad.rds"))
intgrad <- readRDS(here("real_data_new/results/deephit/intgrad.rds"))
grad_shap <- readRDS(here("real_data_new/results/deephit/grad_shap.rds"))


# Create and save plots --------------------------------------------------------
path <- here("figures_paper/")
plots <- plot_result(grad, orig_img, path = path, name = "real_data_grad", as_force = FALSE)
plots <- plot_result(sgrad, orig_img, path = path, name = "real_data_sgrad", as_force = FALSE)
plots <- plot_result(intgrad, orig_img, path = path, name = "real_data_intgrad", as_force = TRUE)
plots <- plot_result(grad_shap, orig_img, path = path, name = "real_data_grad_shap", as_force = TRUE)

