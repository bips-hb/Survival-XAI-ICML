################################################################################
#     This is the replication script for Figure 7 (global importance 
#     ranking) of the ICML 2025 paper "Gradient-based explanations for
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

# Registry ---------------------------------------------------------------------
reg_name <- "sim_global"
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
addProblem(name = "global_no_td", fun = generate_survival_data, seed = 45)

# Algorithms ----------------------------------------------------------------
source(here("Sim_GradSHAP/algorithms.R"))
addAlgorithm(name = "global_deephit", fun = algo)
addAlgorithm(name = "global_coxtime", fun = algo)
addAlgorithm(name = "global_deepsurv", fun = algo)

# Experiments ----------------------------------------------------------------

# Global Importance Ranking
global_prob_design <- list(
  global_no_td = expand.grid(n_train = 2000, n_test = 300, p = 5) 
)

global_algo_design <- list(
  global_deepsurv = expand.grid(
    calc_lime = TRUE,
    only_time = FALSE,
    model_type = "deepsurv",
    num_samples = list(c(25)),
    num_integration = list(c(25)),
    stringsAsFactors = FALSE 
  ),
  global_coxtime = expand.grid(
    calc_lime = TRUE,
    only_time = FALSE,
    model_type = "coxtime",
    num_samples = list(c(25)),
    num_integration = list(c(25)),
    stringsAsFactors = FALSE 
  ),
  global_deephit = expand.grid(
    calc_lime = TRUE,
    only_time = FALSE,
    model_type = "deephit",
    num_cuts = 12,
    num_samples = list(c(25)),
    num_integration = list(c(25)),
    stringsAsFactors = FALSE
  )
)

# Add, test and submit experiments ---------------------------------------------
addExperiments(global_prob_design, global_algo_design, repls = 1)
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
saveRDS(res, here("Sim_GradSHAP/final_results/sim_global_imp.rds"))

# Uncomment to load results
#res <- readRDS(here("Sim_GradSHAP/final_results/sim_global_imp.rds"))

# Postprocess: Global Comparison ----------------------------------------------
res_survlime <- rbindlist(lapply(seq_len(nrow(res)), function(i) {
  res_i <- as.data.table(res$result[[i]][[3]])
  res_i$id <- seq_len(nrow(res_i))
  cbind(res[i, -c("result", "problem")], melt(res_i, id.vars = "id"))
}))
res_survlime <- res_survlime[, .(rank = rank(-value, ties.method = "first"), feature = variable), by = c("model_type", "id")]
res_survlime$rank <- factor(res_survlime$rank, levels = rev(c(1,2,3,4,5)), 
                            labels = rev(c("1st", "2nd", "3rd", "4th", "5th")))
res_survlime$method <- "SurvLIME"
res_global <- rbindlist(lapply(seq_len(nrow(res)), function(i) {
  cbind(res[i, -c("result", "problem")], res$result[[i]][[1]])
}))
res_global <- res_global[, -c("num_samples", "num_integration", "runtime")]
res_global <- res_global[, .(value = abs(mean(value))), by = c("model_type", "feature", "method", "id")]
res_global <- res_global[, .(rank = rank(-value, ties.method = "first"), feature = feature), by = c("model_type", "method", "id")]
res_global$rank <- factor(res_global$rank, levels = rev(c(1,2,3,4,5)), 
                          labels = rev(c("1st", "2nd", "3rd", "4th", "5th")))
res_global <- rbind(res_global, res_survlime)
res_global <- res_global[ , .(frequency = .N), by = c("method", "rank", "model_type", "feature")]
res_global$model_type <- factor(res_global$model_type, levels = c("coxtime", "deephit", "deepsurv"), labels = c("CoxTime", "DeepHit", "DeepSurv"))
res_global$method <- factor(res_global$method, levels = c("GradSHAP", "SurvSHAP", "SurvLIME"), 
                            labels = c("GradSHAP(t)", "SurvSHAP(t)", "SurvLIME"))

# Create Plots -----------------------------------------------------------------

# Global importance
ggplot(res_global, aes(x = frequency, y = rank, fill = feature)) +
  geom_bar(stat = "identity", position = "stack", alpha  = 0.8) +
  labs(title = "", x = "", y = "Importance ranking", fill = "Features (increasing importance)") +
  facet_grid(rows = vars(model_type), cols = vars(method),
             scales = "free_x",
             labeller = as_labeller(function(a)
               paste0(a))) +
  scale_fill_viridis_d() +
  scale_y_discrete(expand = c(0,0)) +
  scale_x_continuous(expand = c(0,0.2)) +
  theme_minimal(base_size = 17, base_line_size = 0) +
  geom_text(aes(label=ifelse(frequency < 20, "", paste0(frequency))), position = position_stack(vjust = 0.5)) +
  theme(
    legend.position = "top"
  )
ggsave(here("figures_paper/gradshapt_ranking.pdf"), width = 13, height = 6)

ggplot(res_global[model_type == "DeepSurv"], aes(x = frequency, y = rank, fill = feature)) +
  geom_bar(stat = "identity", position = "stack", alpha = 0.8) +
  labs(title = "", x = "", y = "Importance ranking", fill = "Features (increasing importance)") +
  facet_grid(rows = vars(model_type), cols = vars(method),
             scales = "free_x",
             labeller = as_labeller(function(a)
               paste0(a))) +
  scale_fill_viridis_d() +
  scale_y_discrete(expand = c(0,0)) +
  scale_x_continuous(expand = c(0,0.2)) +
  theme_minimal(base_size = 17, base_line_size = 0) +
  geom_text(aes(label=ifelse(frequency < 10, "", paste0(frequency))), position = position_stack(vjust = 0.5)) +
  theme(
    plot.margin = margin(0,0,0,0),
    legend.margin = margin(0,0,0,0),
    legend.box.margin = margin(0,0,0,0),
    legend.position = "top"
  )
ggsave(here("figures_paper/gradshapt_ranking_fig.pdf"), width = 12, height = 4.25)
