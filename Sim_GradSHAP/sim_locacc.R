################################################################################
#     This is the replication script for the local accuracy comparison 
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
reg_name <- "sim_locacc"
reg_dir <- here("Sim_GradSHAP/registries", reg_name)
dir.create(here("Sim_GradSHAP/registries"), showWarnings = FALSE)
unlink(reg_dir, recursive = TRUE)
makeExperimentRegistry(file.dir = reg_dir, 
                       conf.file = here("Sim_GradSHAP/config.R"),
                       packages = c("Survinng", "survex", "survival", "simsurv",
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
addProblem(name = "locacc_no_td", fun = generate_survival_data, seed = 44)

# Algorithms ----------------------------------------------------------------
source(here("Sim_GradSHAP/algorithms.R"))

addAlgorithm(name = "locacc_deephit", fun = algo)
addAlgorithm(name = "locacc_coxtime", fun = algo)
addAlgorithm(name = "locacc_deepsurv", fun = algo)

# Experiments ----------------------------------------------------------------

# Local Accuracy 
locacc_prob_design <- list(
  locacc_no_td = expand.grid(n_train = 1000, n_test = 100, p = 20) 
)
locacc_algo_design <- list(
  locacc_deepsurv = expand.grid(
    calc_lime = FALSE,
    only_time = FALSE,
    model_type = "deepsurv",
    num_samples = list(c(99)),
    num_integration = list(c(5, 25, 50)),
    stringsAsFactors = FALSE 
  ),
  locacc_coxtime = expand.grid(
    calc_lime = FALSE,
    only_time = FALSE,
    model_type = "coxtime",
    num_samples = list(c(99)),
    num_integration = list(c(5, 25, 50)),
    stringsAsFactors = FALSE 
  ),
  locacc_deephit = expand.grid(
    calc_lime = FALSE,
    only_time = FALSE,
    model_type = "deephit",
    num_cuts = 12,
    num_samples = list(c(99)),
    num_integration = list(c(5, 25, 50)),
    stringsAsFactors = FALSE
  )
)


# Add, test and submit experiments ---------------------------------------------
addExperiments(locacc_prob_design, locacc_algo_design, repls = 1)
summarizeExperiments()

# Submit jobs
submitJobs()
waitForJobs()

# Get results ------------------------------------------------------------------
loadRegistry(reg_dir)
res <- reduceResultsDataTable()
jobp <- flatten(getJobPars(res$job.id))[, c("job.id", "problem", "p", "model_type")]
res <- merge(jobp, res, by = "job.id")

# Postprocess: Local accuracy --------------------------------------------------
local_acc_pred <- rbindlist(lapply(seq_len(nrow(res)), function(i) {
  cbind(res[i, -c("result", "problem")], res$result[[i]][[2]])
}))
local_acc_res <- rbindlist(lapply(seq_len(nrow(res)), function(i) {
  cbind(res[i, -c("result", "problem")], res$result[[i]][[1]])
}))
locacc_bins <- rbindlist(lapply(seq_len(nrow(res)), function(i) {
  cbind(res[i, -c("result", "problem")], time = res$result[[i]][[4]])
}))
res_locacc <- merge(local_acc_res, local_acc_pred, by = c("job.id", "id", "p", "model_type", "time"))
res_locacc_time <- unique(res_locacc[, c("model_type", "num_samples", "num_integration", "method", "runtime")])
res_locacc_time$method <- ifelse(is.na(res_locacc_time$num_integration), "SurvSHAP(t)", paste0("GradSHAP(t) (", res_locacc_time$num_integration, ")"))
res_locacc_time$model_type <- factor(res_locacc_time$model_type, levels = c("coxtime", "deephit", "deepsurv"), labels = c("CoxTime", "DeepHit", "DeepSurv"))
res_locacc <- res_locacc[, .(sum_attr = sum(value)), by = c("id", "model_type", "time", "num_samples", "num_integration", "method", "pred", "pred_diff")]
res_locacc <- res_locacc[, .(locacc = sqrt(mean((pred_diff - sum_attr)**2) / mean(pred**2))), by = c("time", "num_samples", "num_integration", "method", "model_type")]
res_locacc$method <- ifelse(is.na(res_locacc$num_integration), "SurvSHAP(t)", paste0("GradSHAP(t) (", res_locacc$num_integration, ")"))
res_locacc$method <- factor(res_locacc$method, levels = c("SurvSHAP(t)", "GradSHAP(t) (5)", "GradSHAP(t) (25)", "GradSHAP(t) (50)"))
res_locacc$model_type <- factor(res_locacc$model_type, levels = c("coxtime", "deephit", "deepsurv"), labels = c("CoxTime", "DeepHit", "DeepSurv"))
locacc_bins$model_type <- factor(locacc_bins$model_type, levels = c("coxtime", "deephit", "deepsurv"), labels = c("CoxTime", "DeepHit", "DeepSurv"))


# Save results
saveRDS(res_locacc, here("Sim_GradSHAP/final_results/sim_locacc.rds"))
saveRDS(res_locacc_time, here("Sim_GradSHAP/final_results/sim_locacc_time.rds"))
saveRDS(locacc_bins, here("Sim_GradSHAP/final_results/sim_locacc_bins.rds"))

# Create Plots -----------------------------------------------------------------
library(cowplot)

# Load final results
#res_locacc <- readRDS(here("Sim_GradSHAP/final_results/sim_locacc.rds"))
#res_locacc_time <- readRDS(here("Sim_GradSHAP/final_results/sim_locacc_time.rds"))
#locacc_bins <- readRDS(here("Sim_GradSHAP/final_results/sim_locacc_bins.rds"))

# Create dirs 
if (!dir.exists(here("figures_paper"))) dir.create(here("figures_paper"))
if (!dir.exists(here("figures_paper/other_plots"))) dir.create(here("figures_paper/other_plots"))

# Local Accuracy
ggplot(res_locacc, aes(x = time)) +
  geom_line(aes(y = locacc, color = method)) +
  geom_point(aes(y = locacc, color = method), alpha = 0.5) +
  theme_minimal(base_size = 14) +
  geom_rug(data = locacc_bins, aes(x = time), alpha = 0.5, color = "black") +
  scale_color_viridis_d() +
  facet_wrap(vars(model_type), scale = "free_y", nrow = 1) +
  scale_y_log10() +
  theme(legend.position = "top",
        plot.margin = margin(0,0,0,0),
        legend.margin = margin(0,0,0,0),
        legend.box.margin = margin(0,0,0,0)
  ) +
  labs(x = "Time t", y = "Local accuracy", color = "Method")
ggsave(here("figures_paper/gradshapt_localacc.pdf"), width = 10, height = 4)

# Runtime
res_locacc_time$method <- factor(res_locacc_time$method, levels = c("SurvSHAP(t)", "GradSHAP(t) (5)", "GradSHAP(t) (25)", "GradSHAP(t) (50)"))
ggplot(res_locacc_time, aes(x = method, y = runtime)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(x = NULL, y = "Runtime (sec)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 0.75)) +
  facet_wrap(vars(model_type), nrow = 1, scales = "free_y")
ggsave(here("figures_paper/other_plots/gradshapt_localacc_runtime.pdf"), width = 8, height = 5)
