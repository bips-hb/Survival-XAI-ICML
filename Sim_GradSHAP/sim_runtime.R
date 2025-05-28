################################################################################
#     This is the replication script for the runtime comparison 
#     between GradSHAP(t) and SurvSHAP(t) on simulated data and part 
#     of the ICML 2025 paper "Gradient-based explanations for
#     Deep Survival Models".
################################################################################
library(data.table)
library(batchtools)
library(ggplot2)
library(here)

# Load setup file
source(here("setup.R"))

# Set seed for reproducibility
set.seed(42)

# Registry ----------------------------------------------------------------
reg_name <- "sim_runtime"
reg_dir <- here("Sim_GradSHAP/registries", reg_name)
dir.create(here("Sim_GradSHAP/registries"), showWarnings = FALSE)
unlink(reg_dir, recursive = TRUE)
makeExperimentRegistry(file.dir = reg_dir, 
                       conf.file = here("Sim_GradSHAP/config.R"),
                       packages = c("survinng", "survex", "survival", "simsurv",
                                    "torch", "survivalmodels", "callr", 
                                    "microbenchmark"),
                       source = c(here("utils/utils_nn_training.R"),
                                  here("Sim_GradSHAP/limit_cpus.R"),
                                  here("Sim_GradSHAP/algorithms.R")))

# Problems ----------------------------------------------------------------
generate_survival_data <- function(data, job, n_train, n_test, p) {
  x <- data.frame(matrix(rnorm((n_train + n_test) * p), n_train + n_test, p))
  colnames(x) <- paste0("x", seq_len(p))
  betas <- seq(0, 1, length.out = p) * rep(c(1, -1), length.out = p)
  names(betas) <- colnames(x)
  simdat <- simsurv(dist = "weibull", lambdas = 0.1, gammas = 2.5, 
                    betas = betas, x = x, maxt = 10)
  y <- simdat[, -1]
  colnames(y)[1] <- "time"
  dat <- cbind(y, x)
  
  list(train = dat[seq_len(n_train), ], test = dat[-seq_len(n_train), ])
}
addProblem(name = "runtime_no_td", fun = generate_survival_data, seed = 43)

# Algorithms ----------------------------------------------------------------
source(here("Sim_GradSHAP/algorithms.R"))

addAlgorithm(name = "runtime_deephit", fun = algo)
addAlgorithm(name = "runtime_coxtime", fun = algo)
addAlgorithm(name = "runtime_deepsurv", fun = algo)

# Experiments ----------------------------------------------------------------

# Runtime Comparison
runtime_prob_design <- list(
  runtime_no_td = expand.grid(
    n_train = 1000, n_test = 100, 
    p = c(5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160),
    stringsAsFactors = FALSE)
)

runtime_algo_design <- list(
  runtime_deepsurv = expand.grid(
    only_time = TRUE,
    n_times = 20L,
    model_type = "deepsurv",
    num_samples = list(c(25)),
    num_integration = list(c(5, 25, 50)),
    stringsAsFactors = FALSE 
  ),
  runtime_coxtime = expand.grid(
    only_time = TRUE,
    n_times = 20L,
    model_type = "coxtime",
    num_samples = list(c(25)),
    num_integration = list(c(5, 25, 50)),
    stringsAsFactors = FALSE 
  ),
  runtime_deephit = expand.grid(
    only_time = TRUE,
    n_times = 20L,    
    model_type = "deephit",
    num_cuts = 10,
    num_samples = list(c(25)),
    num_integration = list(c(5, 25, 50)),
    stringsAsFactors = FALSE
  )
)

# Add, test and submit experiments ---------------------------------------------
addExperiments(runtime_prob_design, runtime_algo_design, repls = 10)
summarizeExperiments()

# Submit jobs
submitJobs()
waitForJobs()

# Get results ------------------------------------------------------------------
loadRegistry(reg_dir)
res <- reduceResultsDataTable()
jobp <- flatten(getJobPars(res$job.id))[, c("job.id", "problem", "p", "model_type")]
res <- merge(jobp, res, by = "job.id")

# Save results
saveRDS(res, here("Sim_GradSHAP/final_results/sim_runtime.rds"))

# Load final results
#res <- readRDS(here("Sim_GradSHAP/final_results/sim_runtime.rds"))

# Postprocess: Runtime comparison ----------------------------------------------
res_runtime <- rbindlist(lapply(seq_len(nrow(res)), function(i) {
  cbind(res[i, -c("result", "problem")], res$result[[i]][[1]])
}))
res_runtime$method <- ifelse(is.na(res_runtime$num_integration), "SurvSHAP(t)", paste0("GradSHAP(t) (", res_runtime$num_integration, ")"))
res_runtime <- res_runtime[, .(runtime = mean(runtime)), by = c("p", "model_type", "method")]
res_runtime$method <- factor(res_runtime$method, levels = c("SurvSHAP(t)", "GradSHAP(t) (5)", "GradSHAP(t) (25)", "GradSHAP(t) (50)"))
res_runtime$model_type <- factor(res_runtime$model_type, levels = c("coxtime", "deephit", "deepsurv"), labels = c("CoxTime", "DeepHit", "DeepSurv"))


# Create Plots ----------------------------------------------------------------
library(cowplot)

# Runtime Comparison
ggplot(res_runtime, aes(x = p, y = runtime, color = method)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  scale_color_viridis_d() +
  facet_wrap(vars(model_type), scale = "free_y", nrow = 1) +
  scale_y_log10() +
  theme(legend.position = "top") +
  labs(x = "Number of features (p)", y = "Runtime (sec)", color = "Method")
ggsave(here("figures_paper/gradshapt_runtime.pdf"), width = 8, height = 4)


# The main plot in the paper
legend <- get_plot_component(
  ggplot(res_runtime[model_type == "DeepHit"], aes(x = p, y = runtime, color = method)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  scale_color_viridis_d() +
  theme(legend.position = "top") +
  labs(color = "Method") +
  scale_y_log10(), "guide-box", return_all = TRUE
)[[4]]

p <- ggplot(res_runtime[model_type != "CoxTime"], aes(x = p, y = runtime, color = method)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  scale_color_viridis_d() +
  facet_wrap(vars(model_type), scale = "free_y", nrow = 1) +
  scale_y_log10() +
  theme(legend.position = "none") +
  labs(x = "Number of features (p)", y = "Runtime (sec)", color = "Method")

# Load loca accuary result
res_locacc <- readRDS(here("Sim_GradSHAP/final_results/sim_locacc.rds"))
res_locacc_bins <- readRDS(here("Sim_GradSHAP/final_results/sim_locacc_bins.rds"))

p2 <- ggplot(res_locacc[model_type == "DeepSurv"], aes(x = time)) +
  geom_line(aes(y = locacc, color = method)) +
  geom_point(aes(y = locacc, color = method), alpha = 0.5) +
  theme_minimal(base_size = 14) +
  geom_rug(data = locacc_bins[model_type == "DeepSurv"], aes(x = time), alpha = 0.5, color = "black") +
  scale_color_viridis_d() +
  facet_wrap(vars(model_type), scale = "free_y", nrow = 1) +
  scale_y_log10() +
  theme(legend.position = "none",
        plot.margin = margin(0,0,0,0),
        legend.margin = margin(0,0,0,0),
        legend.box.margin = margin(0,0,0,0)
  ) +
  labs(x = "Time t", y = "Local accuracy", color = "Method")

# Combine Fig
plot_grid(legend, plot_grid(p, p2, rel_widths = c(2, 1.6)), ncol = 1, rel_heights = c(1, 10))
ggsave(here("figures_paper/gradshapt_fig.pdf"), width = 9, height = 4.5)
