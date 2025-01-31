library(data.table)
library(batchtools)
library(ggplot2)
library(here)

# Load setup file
source(here("setup.R"))

# Set seed for reproducibility
set.seed(42)

# Registry ----------------------------------------------------------------
reg_name <- "sim_gradshapt"
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

addProblem(name = "runtime_no_td", fun = generate_survival_data, seed = 43)
addProblem(name = "locacc_no_td", fun = generate_survival_data, seed = 44)
addProblem(name = "global_no_td", fun = generate_survival_data, seed = 45)

# Algorithms ----------------------------------------------------------------
source(here("Sim_Shapley/algorithms.R"))

addAlgorithm(name = "runtime_deephit", fun = algo)
addAlgorithm(name = "runtime_coxtime", fun = algo)
addAlgorithm(name = "runtime_deepsurv", fun = algo)
addAlgorithm(name = "locacc_deephit", fun = algo)
addAlgorithm(name = "locacc_coxtime", fun = algo)
addAlgorithm(name = "locacc_deepsurv", fun = algo)
addAlgorithm(name = "global_deephit", fun = algo)
addAlgorithm(name = "global_coxtime", fun = algo)
addAlgorithm(name = "global_deepsurv", fun = algo)

# Experiments ----------------------------------------------------------------

############################# Runtime Comparison #############################
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

############################## Local Accuracy ################################
locacc_prob_design <- list(
  locacc_no_td = expand.grid(n_train = 1000, n_test = 100, p = 20) # 20
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

############################ Global Importance ###############################
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



# Add, test and submitt exoeriments -------------------------------------------
addExperiments(locacc_prob_design, locacc_algo_design, repls = 1)
addExperiments(global_prob_design, global_algo_design, repls = 1)
addExperiments(runtime_prob_design, runtime_algo_design, repls = 5)
summarizeExperiments()


#testJob(2, external = FALSE)


submitJobs()
waitForJobs()

loadRegistry(reg_dir)


# Get results ------------------------------------------------------------------
res <- reduceResultsDataTable()
jobp <- flatten(getJobPars(res$job.id))[, c("job.id", "problem", "p", "model_type")]
res <- merge(jobp, res, by = "job.id")


# Load final results
#res <- readRDS(here("Sim_GradSHAP/final_results/sim_gradshap_final.rds"))

# Postprocess: Runtime comparison ----------------------------------------------
res_runtime <- res[problem == "runtime_no_td", ]
res_runtime <- rbindlist(lapply(seq_len(nrow(res_runtime)), function(i) {
  cbind(res_runtime[i, -c("result", "problem")], res_runtime$result[[i]][[1]])
}))
res_runtime$method <- ifelse(is.na(res_runtime$num_integration), "SurvSHAP(t)", paste0("GradSHAP(t) (", res_runtime$num_integration, ")"))
res_runtime <- res_runtime[, .(runtime = mean(runtime)), by = c("p", "model_type", "method")]
res_runtime$method <- factor(res_runtime$method, levels = c("SurvSHAP(t)", "GradSHAP(t) (5)", "GradSHAP(t) (25)", "GradSHAP(t) (50)"))
res_runtime$model_type <- factor(res_runtime$model_type, levels = c("coxtime", "deephit", "deepsurv"), labels = c("CoxTime", "DeepHit", "DeepSurv"))

# Postprocess: Local accuracy --------------------------------------------------
res_locacc <- res[problem == "locacc_no_td", ]
local_acc_pred <- rbindlist(lapply(seq_len(nrow(res_locacc)), function(i) {
  cbind(res_locacc[i, -c("result", "problem")], res_locacc$result[[i]][[2]])
}))
local_acc_res <- rbindlist(lapply(seq_len(nrow(res_locacc)), function(i) {
  cbind(res_locacc[i, -c("result", "problem")], res_locacc$result[[i]][[1]])
}))
locacc_bins <- rbindlist(lapply(seq_len(nrow(res_locacc)), function(i) {
  cbind(res_locacc[i, -c("result", "problem")], time = res_locacc$result[[i]][[4]])
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


# Postprocess: Global Comparison ----------------------------------------------
res_global <- res[problem == "global_no_td", ]
res_survlime <- rbindlist(lapply(seq_len(nrow(res_global)), function(i) {
  res_i <- as.data.table(res_global$result[[i]][[3]])
  res_i$id <- seq_len(nrow(res_i))
  cbind(res_global[i, -c("result", "problem")], melt(res_i, id.vars = "id"))
}))
res_survlime <- res_survlime[, .(rank = rank(-value, ties.method = "first"), feature = variable), by = c("model_type", "id")]
res_survlime$rank <- factor(res_survlime$rank, levels = rev(c(1,2,3,4,5)), 
                            labels = rev(c("1st", "2nd", "3rd", "4th", "5th")))
res_survlime$method <- "SurvLIME"
res_global <- rbindlist(lapply(seq_len(nrow(res_global)), function(i) {
  cbind(res_global[i, -c("result", "problem")], res_global$result[[i]][[1]])
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

res_locacc_time$method <- factor(res_locacc_time$method, levels = c("SurvSHAP(t)", "GradSHAP(t) (5)", "GradSHAP(t) (25)", "GradSHAP(t) (50)"))
ggplot(res_locacc_time, aes(x = method, y = runtime)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(x = NULL, y = "Runtime (sec)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 0.75)) +
  facet_wrap(vars(model_type), nrow = 1, scales = "free_y")
ggsave(here("figures_paper/gradshapt_localacc_runtime.pdf"), width = 8, height = 5)

# Combine Fig
plot_grid(legend, plot_grid(p, p2, rel_widths = c(2, 1.6)), ncol = 1, rel_heights = c(1, 10))
ggsave(here("figures_paper/gradshapt_fig.pdf"), width = 9, height = 4.5)

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
