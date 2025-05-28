################################################################################
#     This script reproduces the results of Figure 9 (real data on
#     multimodal medical data) in the ICML 2025 paper "Gradient-based 
#     explanations for Deep Survival Models". 
################################################################################
library(survinng)
library(torch)
library(torchvision)
library(ggplot2)
library(data.table)
library(dplyr)
library(cli)
library(here)

# Load helper functions
source(here("real_data/utils.R"))

# Set seeds for reproducibility
set.seed(42)
torch_manual_seed(42)

# Load model--------------------------------------------------------------------
cli_progress_step("Loading model")

# Load model metadata
model_metadata <- read.csv(here("real_data/results/deephit/metadata.csv"))
n_img_out <- model_metadata$n_img_out
n_out <- model_metadata$n_out
out_bias <- as.logical(model_metadata$out_bias)
n_tab_feat <- model_metadata$Number.of.tabular.features

# Load model state dict
model_state_dict <- load_state_dict(here("real_data/results/deephit/model.pt"))

# Replicate model
net_image <- torchvision::model_resnet34(num_classes = n_img_out)
model <- MultiModalModel(net_image, tabular_features = rep(1, n_tab_feat),
                         n_out = n_out, n_img_out = n_img_out, out_bias = out_bias)

# Load model state dict
model <- model$load_state_dict(model_state_dict)
model$eval()

# Load data to be explained-----------------------------------------------------
cli_progress_step("Loading data")

data <- read.csv(here("real_data/data/deephit_tabular_data.csv"))
test_images <- list.files(here("real_data/data/full/test"), full.names = FALSE)

# Filter data
data <- data[data$full_path %in% test_images, ]
data_tab <- torch_tensor(as.matrix(data[, c(1, 2, 3, 4)]))
data_img <- torch_stack(lapply(data$full_path, function(x) {
  base_loader(here(paste0("real_data/data/full/test/", x))) %>%
    (function(x) x[,,1:3]) %>%
    transform_to_tensor() %>%
    transform_center_crop(226) %>%
    transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
}), dim = 1)

# Explain model-----------------------------------------------------------------
torch_set_num_threads(30L)
exp_deephit <- survinng::explain(model, list(data_img, data_tab), 
                                 model_type = "deephit",
                                 time_bins = seq(0, 17, length.out = n_out))

# Instance to be explained
ids <- 230
orig_img <- base_loader(here(paste0("real_data/data/full/test/", data[ids, ]$full_path)))

# Run Grad(t)
cli_progress_step("Running Grad(t)")
grad <- surv_grad(exp_deephit, instance = ids, batch_size = 5000) |>
  as.data.table()

# Run SG(t)
cli_progress_step("Running SG(t)")
sgrad <- surv_smoothgrad(exp_deephit, instance = ids, n = 100, 
                         noise_level = 0.3, batch_size = 5000)  |>
  as.data.table()

# Run IntGrad(t)
cli_progress_step("Running IntGrad(t)")
intgrad <- surv_intgrad(exp_deephit, instance = ids, n = 100, batch_size = 5000)  |>
  as.data.table()

# Run GradSHAP(t)
cli_progress_step("Running GradSHAP(t)")
grad_shap <- surv_gradSHAP(exp_deephit, instance = ids, n = 20, num_samples = 100, 
                           batch_size = 5000, replace = FALSE)  |>
  as.data.table()


# Save results -----------------------------------------------------------------
saveRDS(grad, here("real_data/results/deephit/grad.rds"))
saveRDS(sgrad, here("real_data/results/deephit/sgrad.rds"))
saveRDS(intgrad, here("real_data/results/deephit/intgrad.rds"))
saveRDS(grad_shap, here("real_data/results/deephit/grad_shap.rds"))

# Read results -----------------------------------------------------------------
grad <- readRDS(here("real_data/results/deephit/grad.rds"))
sgrad <- readRDS(here("real_data/results/deephit/sgrad.rds"))
intgrad <- readRDS(here("real_data/results/deephit/intgrad.rds"))
grad_shap <- readRDS(here("real_data/results/deephit/grad_shap.rds"))


# Create and save plots --------------------------------------------------------
cli_progress_step("Creating plots")

# Create dir
if (!dir.exists(here("figures_paper"))) dir.create(here("figures_paper"))
if (!dir.exists(here("figures_paper/other_plots"))) dir.create(here("figures_paper/other_plots"))

path <- here("figures_paper/")
path_other <- here("figures_paper/other_plots/")
plots <- plot_result(grad, orig_img, path = path_other, name = "real_data_grad", as_force = FALSE)
plots <- plot_result(sgrad, orig_img, path = path_other, name = "real_data_sgrad", as_force = FALSE)
plots <- plot_result(intgrad, orig_img, path = path_other, name = "real_data_intgrad", as_force = TRUE)
plots <- plot_result(grad_shap, orig_img, path = path, name = "real_data_grad_shap", as_force = TRUE)

