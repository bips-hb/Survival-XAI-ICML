################################################################################
#                   Algorithms for 'batchtools' Simulation
################################################################################

# Main algorithm ---------------------------------------------------------------
algo <- function(data, job, instance, hidden_nodes = 32, num_layers = 2,
                 num_samples = 20, num_integration = 10, num_cuts = 20,
                 num_cpus = 10L, n_times = 1L,
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
  } else if (model_type == "coxtime") {
    res <- callr::r(fit_coxtime,
      args = list(
        instance = instance, 
        hidden_nodes = hidden_nodes,
        num_layers = num_layers,
        calc_lime = calc_lime
      ), show = TRUE)
  }
  
  torch::torch_set_num_interop_threads(num_cpus)
  torch::torch_set_num_threads(num_cpus)
  
  # Run SurvSHAP
  res_survshap <- lapply(num_samples, run_survshap,
                         extracted_model = res[[1]], 
                         df_test = instance$test,
                         only_time = only_time, 
                         n_times = n_times)
  res_survshap <- do.call("rbind", res_survshap)
  
  # Run GradSHAP
  args <- expand.grid(num_samples = num_samples, num_int = num_integration)
  res_gradshap <- lapply(seq_len(nrow(args)), function(i) {
    run_gradshap(extracted_model = res[[1]],
                 df_test = instance$test,
                 num_samples = args$num_samples[i],
                 num_integration = args$num_int[i],
                 only_time = only_time,
                 n_times = n_times)
  })
  
  # Combine results
  if (only_time) {
    res_gradshap <- do.call("rbind", res_gradshap)
    result <- list(rbind(res_gradshap, res_survshap), instance$test$time)
  } else {
    preds <- unique(do.call("rbind", lapply(res_gradshap, `[[`, 2)))
    res_gradshap <- do.call("rbind", lapply(res_gradshap, `[[`, 1))
    
    result <- list(rbind(res_gradshap, res_survshap), preds, res$survlime, instance$test$time)
  }
  
  result
}

# Model training ---------------------------------------------------------------

fit_deephit <- function(instance, hidden_nodes, num_layers, cuts, calc_lime) {
  library(survivalmodels)
  library(survival)
  source(here::here("utils/utils_survivalmodels.R"))
  reticulate::use_condaenv("Survinng_paper", required = TRUE)
  
  # Set seeds
  set.seed(1)
  set_seed(1)
  
  nodes <- as.integer(rep(hidden_nodes, num_layers))
  model <- deephit(Surv(time, status) ~ ., data = instance$train, 
                   verbose = FALSE, epochs = 100L, num_nodes = nodes,
                   early_stopping = TRUE, frac = 0.33, patience = 10L, cuts = cuts, 
                   batch_size = 500L, dropout = 0.1, num_workers = 1L, device = "cpu")

  # Run SurvLime
  test <- instance$test
  if (calc_lime) {
    survlime <- reticulate::import_from_path("survlime", here::here("Sim_GradSHAP/"))
    res <- survlime$run_survlime(test, model, num_samples = 50L,  model_type = "deephit")
    colnames(res) <- colnames(test)[-c(1,2)]
  } else {
    res <- NULL
  }
  
  # Make predictions
  pred <- predict(model, newdata = test, type = "survival")
  dat <- data.frame(
    pred = c(pred), 
    time = rep(as.numeric(colnames(pred)), each = nrow(pred)),
    id = rep(1:nrow(pred), ncol(pred))
  )
  list(Survinng::extract_model(model), pred = dat, survlime = res)
}

fit_deepsurv <- function(instance, hidden_nodes, num_layers, calc_lime) {
  library(survivalmodels)
  library(survival)
  reticulate::use_condaenv("Survinng_paper", required = TRUE)
  
  # Set seeds
  set.seed(1)
  set_seed(1)
  
  nodes <- as.integer(rep(hidden_nodes, num_layers))
  model <- deepsurv(Surv(time, status) ~ ., data = instance$train, 
                   verbose = FALSE, epochs = 100L, num_nodes = nodes,
                   early_stopping = TRUE, frac = 0.33, patience = 10L,
                   batch_size = 500L, dropout = 0.1, num_workers = 1L,
                   device = "cpu")
  # Run SurvLime
  test <- instance$test
  if (calc_lime) {
    survlime <- reticulate::import_from_path("survlime", here::here("Sim_GradSHAP/"))
    res <- survlime$run_survlime(test, model, num_samples = 50L, model_type = "deepsurv")
    colnames(res) <- colnames(test)[-c(1,2)]
  } else {
    res <- NULL
  }
  
  # Make predictions
  pred <- predict(model, newdata = test, type = "survival")
  dat <- data.frame(
    pred = c(pred), 
    time = rep(as.numeric(colnames(pred)), each = nrow(pred)),
    id = rep(1:nrow(pred), ncol(pred))
  )
  list(Survinng::extract_model(model, num_basehazard = 50L), pred = dat, survlime = res)
}


fit_coxtime <- function(instance, hidden_nodes, num_layers, calc_lime) {
  library(survivalmodels)
  library(survival)
  reticulate::use_condaenv("Survinng_paper", required = TRUE)
  
  # Set seeds
  set.seed(1)
  set_seed(1)
  
  nodes <- as.integer(rep(hidden_nodes, num_layers))
  model <- coxtime(Surv(time, status) ~ ., data = instance$train, 
                   verbose = FALSE, epochs = 100L, num_nodes = nodes,
                   early_stopping = TRUE, frac = 0.33, patience = 10L,
                   batch_size = 500L, dropout = 0.1, num_workers = 1L,
                   device = "cpu")
  # Run SurvLime
  test <- instance$test
  if (calc_lime) {
    survlime <- reticulate::import_from_path("survlime", here::here("Sim_GradSHAP/"))
    res <- survlime$run_survlime(test, model, num_samples = 50L, model_type = "coxtime")
    colnames(res) <- colnames(test)[-c(1,2)]
  } else {
    res <- NULL
  }
  
  # Make predictions
  pred <- predict(model, newdata = test, type = "survival")
  dat <- data.frame(
    pred = c(pred), 
    time = rep(as.numeric(colnames(pred)), each = nrow(pred)),
    id = rep(1:nrow(pred), ncol(pred))
  )
  list(Survinng::extract_model(model, num_basehazard = 50L), pred = dat, survlime = res)
}


# XAI Methods ------------------------------------------------------------------


# Run SHAP methods
run_survshap <- function(extracted_model, df_test, num_samples, only_time, n_times = 5L) {
  # Create explainer
  explainer <- Survinng::explain(extracted_model, data = df_test)
  
  # Run SurvSHAP
  survex_explainer <- to_survex(explainer)
  time_survshap <- microbenchmark({
    survshap <- model_survshap(survex_explainer, df_test[, -c(1, 2)], 
                               N = num_samples)
  }, times = n_times)$time / 1e9
  
  if (only_time) {
    result <- data.frame(
      runtime = median(time_survshap),
      method = "SurvSHAP",
      num_samples = num_samples,
      num_integration = NA
    )
  } else {
    # Combine results as data.frame
    res_survshap <- do.call("rbind", lapply(seq_along(survshap$result), function(i) {
      cbind(
        expand.grid(time = survshap$eval_times, feature = colnames(survshap$result[[i]])),
        data.frame(
          value = c(as.matrix(survshap$result[[i]])),
          id = i,
          num_integration = NA,
          num_samples = num_samples,
          runtime = median(time_survshap), 
          method = "SurvSHAP"
        )
      )
    }))
    
    result <- res_survshap[, c("id", "feature", "time", "value", "num_samples", "num_integration", "runtime", "method")]
  }
  
  result
}

run_gradshap <- function(extracted_model, df_test, num_samples, num_integration, 
                         only_time, dtype = "float", n_times = 5L) {
  # Create explainer
  explainer <- Survinng::explain(extracted_model, data = df_test)
  
  # Run GradSHAP
  time_gradshap <- microbenchmark({
    gradshap <- surv_gradSHAP(explainer, instance = seq_len(nrow(df_test)), 
                              n = num_integration, batch_size = 100000000,
                              replace = FALSE,
                              num_samples = num_samples, dtype = dtype)
  }, times = n_times)$time / 1e9
  
  if (only_time) {
    result <- data.frame(
      runtime = median(time_gradshap),
      method = "GradSHAP",
      num_samples = num_samples,
      num_integration = num_integration
    )
  } else {
    res_gradshap <- expand.grid(
      id = seq_len(dim(gradshap$res[[1]])[1]),
      feature = dimnames(gradshap$res[[1]])[[2]],
      time = gradshap$time
    )
    res_gradshap$value <- as.vector(gradshap$res[[1]])
    res_gradshap$num_samples <- num_samples
    res_gradshap$num_integration <- num_integration
    res_gradshap$runtime <- median(time_gradshap)
    res_gradshap$method <- "GradSHAP"
    
    # Get prediction and prediction difference
    res_pred <- expand.grid(
      id = seq_len(dim(gradshap$pred_diff)[1]),
      time = gradshap$time
    )
    res_pred$pred_diff <- as.vector(gradshap$pred_diff)
    res_pred$pred <- as.vector(gradshap$pred)
    
    result <- list(res_gradshap, res_pred)
  }
  
  result
}
