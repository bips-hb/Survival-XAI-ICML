################################################################################
#                        Explain model with SurvSHAP
################################################################################
# Note: This part is very very time-consuming and requires a lot of memory!
#       We aborted this simulation on our workstation after 1 week of runtime
#       and it took almost 800 GB of RAM.
library(Survinng)
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

# Explain model ----------------------------------------------------------------
cli_progress_step("Running SurvSHAP")

model <- exp_deephit$model
model$eval()
times <- exp_deephit$model$time_bins
newdata <- as.data.frame(as.matrix(torch_cat(list(data_img$view(c(dim(data_img)[1], -1)), data_tab), dim = 2)))
y <- Surv(data$Survival.months, data$censored)

# Transform image to a tabular format (required for SurvSHAP)
new_datapoint <- as.array(torch_cat(list(
  data_img[ids, ]$view(c(-1)), 
  data_tab[ids, ]),
  dim = 1)$view(c(1, -1)))
new_datapoint <- as.data.frame(new_datapoint)

# Create survex explainer
survex_exp <- explain_survival(
  model = model,
  data = newdata,
  verbose = TRUE,
  y = y,
  predict_function = function(model, newdata) predict_function(model, newdata, dim = c(-1, 3, 226, 226)),
  label = "DeepHit",
  times = times,
  predict_survival_function = function(model, newdata, times) predict_survival_function(model, newdata, times, dim = c(-1, 3, 226, 226)),
  predict_cumulative_hazard_function = NULL
)

# Run SurvSHAP
cli_progress_step("Running SurvSHAP (N = 1)")
res_1 <- model_survshap(survex_exp, new_datapoint, N = 1)
cli_progress_done()