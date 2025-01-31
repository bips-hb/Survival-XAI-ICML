library(data.table)
library(batchtools)
library(ggplot2)
library(here)

# Load setup file
source(here("setup.R"))

# Set seed for reproducibility
set.seed(42)

# Simulation parameters ----------------------------------------------------------------
num_replicates <- 5
n_train <- 1000
n_test <- 100
p <- c(5, 10, 20, 30, 40, 50, 75, 100)

# Algorithm parameters ----------------------------------------------------------------
num_samples <- c(1, 25, 50)
hidden_nodes <- c(64)
num_integration <- c(5, 20, 50, 100)
model_type <- c("deepsurv", "deephit")

# Registry ----------------------------------------------------------------
reg_name <- "sim_shapley"
reg_dir <- here("Sim_Shapley/registries", reg_name)
dir.create(here("Sim_Shapley/registries"), showWarnings = FALSE)
unlink(reg_dir, recursive = TRUE)
makeExperimentRegistry(file.dir = reg_dir, 
                       conf.file = here("Sim_Shapley/config.R"),
                       packages = c("Survinng", "survex", "survival", "simsurv",
                                    "torch", "survivalmodels", "callr", 
                                    "microbenchmark"),
                       source = c(here("utils/utils_nn_training.R"),
                                  here("Sim_Shapley/limit_cpus.R"),
                                  here("Sim_Shapley/algorithms.R")))

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

addProblem(name = "no_time_dependency", fun = generate_survival_data, seed = 43)

# Algorithms ----------------------------------------------------------------
algo <- function(data, job, instance, hidden_nodes = 32, num_layers = 2,
                 num_samples = 20, num_integration = 10, num_cuts = 20,
                 model_type = "deepsurv", calc_lime = FALSE, only_time = TRUE) {
  
  # Fit the model
  if (model_type == "deephit") {
    res <- callr::r(fit_deephit,  
      args = list(
        instance = instance, 
        hidden_nodes = hidden_nodes,
        num_layers = num_layers,
        cuts = num_cuts,
        calc_lime = calc_lime
      ), show  = TRUE)
  } else if (model_type == "deepsurv") {
    res <- callr::r(fit_deepsurv,
      args = list(
        instance = instance, 
        hidden_nodes = hidden_nodes,
        num_layers = num_layers,
        calc_lime = calc_lime
      ), show = TRUE)
  }
  
  torch::torch_set_num_interop_threads(10L)
  torch::torch_set_num_threads(10L)
  
  
  # Run SurvSHAP
  res_survshap <- lapply(num_samples, run_survshap,
                         extracted_model = res[[1]], 
                         df_test = instance$test,
                         only_time = only_time)
  res_survshap <- do.call("rbind", res_survshap)
  
  # Run GradSHAP
  args <- expand.grid(num_samples = num_samples, num_int = num_integration)
  res_gradshap <- lapply(seq_len(nrow(args)), function(i) {
    run_gradshap(extracted_model = res[[1]],
                 df_test = instance$test,
                 num_samples = args$num_samples[i],
                 num_integration = args$num_int[i],
                 only_time = only_time)
  })
  
  # Combine results
  if (only_time) {
    res_gradshap <- do.call("rbind", res_gradshap)
    result <- rbind(res_gradshap, res_survshap)
  } else {
    preds <- unique(do.call("rbind", lapply(res_gradshap, `[[`, 2)))
    res_gradshap <- do.call("rbind", lapply(res_gradshap, `[[`, 1))
    
    result <- list(rbind(res_gradshap, res_survshap), preds)
  }
  
  result
}

addAlgorithm(name = "deephit", fun = algo)
addAlgorithm(name = "deepsurv", fun = algo)

# Experiments -----------------------------------------------------------
prob_design <- list(
  no_time_dependency = expand.grid(
    n_train = n_train, n_test = n_test, p = p,
    stringsAsFactors = FALSE)
)

algo_design <- list(
  deepsurv = expand.grid(
    hidden_nodes = hidden_nodes,
    num_layers = 2,
    only_time = TRUE,
    model_type = "deepsurv",
    num_samples = list(num_samples),
    num_integration = list(num_integration),
    stringsAsFactors = FALSE 
  ),
  deephit = expand.grid(
    hidden_nodes = hidden_nodes,
    num_layers = 2,
    only_time = TRUE,
    model_type = "deephit",
    num_cuts = 20,
    num_samples = list(num_samples),
    num_integration = list(num_integration),
    stringsAsFactors = FALSE
  )
)


addExperiments(prob_design, algo_design, repls = num_replicates)
summarizeExperiments()
testJob(1, external = TRUE)


submitJobs()


loadRegistry(reg_dir)


# Analysis ---------------------------------------------------------------------
res <- reduceResultsDataTable()


# Time Analysis ----------------------------------------------------------------
result <- rbindlist(lapply(seq_along(res$result), function(i) {
  cbind(job.id = res$job.id[i], res$result[[i]])
}))
jobp <- flatten(getJobPars())[, c(1,3,6,7,8,10)]

result <- merge(result, jobp, by = "job.id")

result$method <- ifelse(is.na(result$num_integration), "SurvSHAP", paste0("GradSHAP (", result$num_integration, ")"))
ggplot(result, aes(x = as.factor(p), y = runtime, color = method)) +
  #geom_smooth() +
  #geom_point() +
  geom_boxplot() +
  facet_grid(num_samples ~ model_type) +
  scale_y_log10()


result <- rbindlist(lapply(seq_along(res$result), function(i) {
  cbind(job.id = res$job.id[i], res$result[[i]][[1]])
}))
preds <- rbindlist(lapply(seq_along(res$result), function(i) {
  cbind(job.id = res$job.id[i], res$result[[i]][[2]])
}))

jobp <- flatten(getJobPars())[, c(1,3,6,7,8,9,12)]

result <- merge(result, jobp, by = "job.id")
preds <- merge(preds, jobp, by = "job.id")
time <- unique(result[, -c("value", "id", "feature", "time")])

library(ggplot2)
time$method <- ifelse(is.na(time$num_integration), "SurvSHAP", paste0("GradSHAP (", time$num_integration, ")"))
ggplot(time, aes(x = as.factor(p), y = runtime, fill = method)) +
  geom_boxplot() +
  facet_grid(num_samples ~ model_type) +
  scale_y_log10()
