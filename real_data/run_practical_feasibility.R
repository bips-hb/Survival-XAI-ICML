################################################################################
#     This script compares the runtime of GradSHAP(t) and SurvSHAP(t) on 
#     real dataset with a smaller image size and ResNet model. It is
#     part of the ICML 2025 paper "Gradient-based explanations for
#     Deep Survival Models".
################################################################################
library(Survinng)
library(torch)
library(torchvision)
library(ggplot2)
library(data.table)
library(dplyr)
library(here)
library(survival)
library(survex)
library(microbenchmark)
library(geomtextpath)
library(cli)

# Set number of threads
num <- 32L
Sys.setenv(OMP_NUM_THREADS = num)
Sys.setenv(OMP_THREAD_LIMIT = num)
Sys.setenv(MKL_NUM_THREADS = num) # Unsure, MKL is an Intel-specific thing
Sys.setenv(MC_CORES = num)

# Package-specific settings
try(data.table::setDTthreads(num))
try(RhpcBLASctl::blas_set_num_threads(num))
try(RhpcBLASctl::omp_set_num_threads(num))
try(torch::torch_set_num_threads(num))
try(torch::torch_set_num_interop_threads(num))


# Set seeds for reproducibility
set.seed(123)
torch_manual_seed(123)

# Load helper functions
source(here("real_data/utils.R"))


# Load model--------------------------------------------------------------------

# Load model metadata
model_metadata <- read.csv(here("real_data/results/deephit_small/metadata.csv"))
n_img_out <- model_metadata$n_img_out
n_out <- model_metadata$n_out
out_bias <- as.logical(model_metadata$out_bias)
n_tab_feat <- model_metadata$Number.of.tabular.features

# Load model state dict
model_state_dict <- load_state_dict(here("real_data/results/deephit_small/model.pt"))

# Replicate model
net_image <- torchvision::model_resnet18(num_classes = n_img_out)
model <- MultiModalModel(net_image, tabular_features = rep(1, n_tab_feat),
                         n_out = n_out, n_img_out = n_img_out, out_bias = out_bias)

# Load model state dict
model <- model$load_state_dict(model_state_dict)
model$eval()

# Load data to be explained-----------------------------------------------------
data <- read.csv(here("real_data/data/deephit_tabular_data_small.csv"))
test_images <- list.files(here("real_data/data/small/test"), full.names = FALSE)

# Filter data
data <- data[data$full_path %in% test_images, ]
data_tab <- torch_tensor(as.matrix(data[, c(1, 2, 3, 4)]))
data_img <- torch_stack(lapply(data$full_path, function(x) {
  base_loader(here(paste0("real_data/data/small/test/", x))) %>%
    (function(x) x[,,1:3]) %>%
    transform_to_tensor() %>%
    transform_center_crop(32) %>%
    transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
}), dim = 1)

# Load data point to be explained
ids <- 230 
orig_img <- base_loader(here(paste0("real_data/data/small/test/", data[ids, ]$full_path)))


# Prepare SurvSHAP explainer ---------------------------------------------------
# Create explainer
exp_deephit <- Survinng::explain(model, list(data_img, data_tab), model_type = "deephit",
                                 time_bins = seq(0, 17, length.out = n_out))


# Preparations for SurvSHAP
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
  predict_function = predict_function,
  label = "DeepHit",
  times = times,
  predict_survival_function = predict_survival_function,
  predict_cumulative_hazard_function = NULL
)

# Define functions for the benchmark -------------------------------------------

# Function to run GradSHAP (ours)
run_survGRAD <- function(explainer, ids, n, num_samples, reps = 1) {
  res <- rbindlist(lapply(1:reps, function(x) {
    cli_progress_step("Running GradSHAP(t): {x}|{reps} --- N = {n} --- samples = {num_samples}")
    res_time <- microbenchmark(
      res = {res = surv_gradSHAP(explainer, instance = ids, n = n, num_samples = num_samples, 
                                 batch_size = 10000, replace = TRUE)},
      times = 1L, unit = "sec")
    res <- as.data.table(res)
    res <- rbind(res[[1]][, c("time", "value", "pred", "pred_diff", "method")], 
                 res[[2]][, c("time", "value", "pred", "pred_diff", "method")])
    res <- res[, .(sum_attr = sum(value)), by = c("time", "pred", "pred_diff")]
    res$method <- paste0("GradSHAP(t)\n (n = ", n, "; samples = ", num_samples, ")")
    res$runtime <- res_time$time / 1e9
    gc()
    res
  }))
  
  res
}

# Function to run SurvSHAP
run_survSHAP <- function(explainer, instance, N, reps = 1) {
  res <- rbindlist(lapply(1:reps, function(x) {
    cli_progress_step("Running SurvSHAP(t): {x}|{reps} --- N = {N}")
    res_time <- microbenchmark(
      res = {res = model_survshap(explainer, instance, N = N)},
      times = 1L, unit = "sec")
    res <-  as.data.table(cbind(
      expand.grid(time = res$eval_times, feature = colnames(res$result)),
      data.frame(value = c(as.matrix(res$result)), id = ids, 
                 method = paste0("SurvSHAP(t)\n(samples = ", N, ")"))
    ))
    res <- res[, .(sum_attr = sum(value)), by = c("time", "method")]
    res$runtime <- res_time$time / 1e9
    res$pred <- NA
    res$pred_diff <- NA
    gc()
    res
  }))
  
  res
}

# Start benchmark ---------------------------------------------------------------
result <- rbind(
  run_survGRAD(exp_deephit, ids = ids, n = 10, num_samples = 10, reps = 10),
  run_survGRAD(exp_deephit, ids = ids, n = 25, num_samples = 50, reps = 10),
  run_survGRAD(exp_deephit, ids = ids, n = 50, num_samples = 50, reps = 10),
  run_survSHAP(survex_exp, new_datapoint, N = 5, reps = 10),
  run_survSHAP(survex_exp, new_datapoint, N = 25, reps = 10),
  run_survSHAP(survex_exp, new_datapoint, N = 50, reps = 10)
)

# Postprocess results
res_pred <- unique(result[startsWith(method, "GradSHAP"), c("pred", "pred_diff", "time")])
result <- rbind(
  result[!is.na(pred), ], 
  merge(result[is.na(pred), -c("pred", "pred_diff")], res_pred, by = "time")
)

# Save results
saveRDS(result, here("real_data/results/res_practical_feasibility.rds"))

# Create plot ------------------------------------------------------------------
dt <- readRDS(here("real_data/results/res_practical_feasibility.rds"))

# Uncomment the following line to load the final results used in the paper
# dt <- readRDS(here("real_data/results/res_practical_feasibility_final.rds"))
dt$method <- factor(dt$method, levels = rev(unique(dt$method)))

# Aggregated overview
dt[, .(sum_attr = mean((pred_diff - sum_attr)^2 / pred),
       runtime = mean(runtime)), by = "method"]

# Create time data.table
dt_time <- unique(dt[, c("runtime", "method")])[, .(runtime = mean(runtime)), by = "method"]
dt_time$type <- ifelse(startsWith(as.character(dt_time$method), "GradSHAP"), "GradSHAP(t)", "SurvSHAP(t)")
dt_time$title <- "Runtime Comparison"

# Create local accuracy data.table
dt_res <- dt[, .(sum_attr = mean((pred_diff - sum_attr)^2 / pred)), by = "method"]
dt_res$type <- ifelse(startsWith(as.character(dt_res$method), "GradSHAP"), "GradSHAP(t)", "SurvSHAP(t)")
dt_res$title <- "Instance-wise Local Accuracy"

# Create local accuracy plot
p1 <- ggplot(dt_res) +
  geom_bar(aes(y = method, x = sum_attr, fill = type), stat = "identity") +
  theme_minimal() +
  scale_fill_viridis_d(direction = -1) +
  labs(y = NULL, x = "Local accuracy (average over timepoints)", fill = NULL) +
  scale_x_continuous(transform = "reverse") +
  scale_y_discrete(position = "right") +
  geom_vline(xintercept = 0, linetype = "solid") +
  facet_grid(cols = vars(title)) +
  theme(legend.position = "none",
        strip.text.x = element_text(size = 12, face = "bold"),
        axis.text.y = element_blank())


# Create runtime plot
p2 <- ggplot() +
  geom_bar(aes(y = method, x = runtime, fill = type), stat = "identity", data = dt_time) +
  theme_minimal() +
  scale_fill_viridis_d(direction = -1) +
  theme(axis.text.y = element_text(hjust = 0.5),
        strip.text.x = element_text(size = 12, face = "bold"),
        legend.position = "none") +
  labs(x = "Time (sec)", y = NULL , fill = NULL) +
  geom_textvline(xintercept = 60, label = "1 minute", linetype = "dashed", hjust = 0.8) +
  geom_textvline(xintercept = 10*60, label = "10 minutes", linetype = "dashed", hjust = 0.8) +
  geom_textvline(xintercept = 5*60, label = "5 minutes", linetype = "dashed", hjust = 0.8) +
  geom_textvline(xintercept = 10, label = "10 seconds", linetype = "dashed", hjust = 0.9) +
  geom_textvline(xintercept = 1, label = "1 second", linetype = "dashed", hjust = 0.9) +
  geom_vline(xintercept = 0, linetype = "solid") +
  scale_x_continuous(transform = "pseudo_log", 
                     breaks = c(0, 5, 30, 200, 400)) +
  facet_grid(cols = vars(title)) 


# Combine plots
library(patchwork)
combined <- p1 + p2 & theme(legend.position = "top", 
                            legend.margin = margin(b = -10),
                            plot.margin = margin(r = 8, l = 8, b = 2, t = 2))
p <- combined + plot_layout(guides = "collect")

# Save combined plot
ggsave(here("figures_paper/comparison_real_data.pdf"), plot = p, width = 9, height = 4)
