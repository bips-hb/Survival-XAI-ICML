################################################################################
#                  Utility functions for Neural network training
################################################################################

# Fit model function
fit_model <- function(model, train, test, num_layers = 2, num_hidden = 32L, cuts = 30,
                      num_basehazard = 80L) {
  callr::r(function(model, train, test, num_layers, num_hidden, cuts, num_basehazard) {
    library(survivalmodels)
    library(survival)
    source(here::here("utils/utils_survivalmodels.R"))
    reticulate::use_condaenv("Survinng_paper", required = TRUE)
    set.seed(1)
    set_seed(1)
    
    # Get the hidden nodes
    nodes <- rep(num_hidden, num_layers)
    
    # Fit model
    if (model == "CoxTime") {
      model <- coxtime(Surv(time, status) ~ ., data = train, verbose = FALSE, epochs = 500L,
                       early_stopping = TRUE, frac = 0.33, batch_size = 1024L, patience = 10L,
                       dropout = 0.1, num_workers = 1L)
    } else if (model == "DeepHit") {
      model <- deephit(Surv(time, status) ~ ., data = train, verbose = FALSE, epochs = 500L,
                       early_stopping = TRUE, frac = 0.33, patience = 10L, cuts = cuts, 
                       batch_size = 1024L, dropout = 0.1, num_workers = 1L)
    } else if (model == "DeepSurv") {
      model <- deepsurv(Surv(time, status) ~ ., data = train, verbose = FALSE, epochs = 100L,
                        early_stopping = TRUE, frac = 0.33, patience = 10L, batch_size = 1024L,
                        dropout = 0.1, num_workers = 1L)
    } else {
      stop("Model not found")
    }
    
    # Make predictions
    pred <- predict(model, newdata = test, type = "survival")
    dat <- data.frame(
      pred = c(pred), 
      time = rep(as.numeric(colnames(pred)), each = nrow(pred)),
      id = rep(1:nrow(pred), ncol(pred))
    )
    list(Survinng::extract_model(model, num_basehazard = num_basehazard), pred = dat)
  }, list(model, train, test, num_layers, num_hidden, cuts, num_basehazard), show = FALSE)
}


################################################################################
#                     Survinng --> Survex (converter)
################################################################################


# convert Survinng explainer to survex explainer
to_survex <- function(explainer, verbose = FALSE) {
  model_type <- explainer$model$.classes[1]
  
  # Get the model and freeze all random components
  model <- explainer$model
  model$eval()
  
  # Get the whole dataset
  dataset <- explainer$data[[1]]
  
  # Get the input data as an array
  data <- dataset[, !(colnames(dataset) %in% c("time", "status")), ]
  
  # Get the outcome
  y <- Surv(time = dataset$time, event = dataset$status)
  
  # Get target labels
  target <- switch(model_type, DeepHit = "pmf", CoxTime = "hazard", DeepSurv = "hazard")
  
  # Get the predict function for the risk score
  # Note: 'newdata' needs to be transformed to a torch tensor to be used for
  # the torch model. After that, the output needs to be transformed back to an
  # array.
  predict_function <- function(model, newdata) {
    input <- torch_tensor(as.matrix(newdata))
    input <- model$preprocess_fun(input)
    out <- model(input, target = target)[[1]]
    out <- out$squeeze(dim = seq_len(out$dim())[-c(1, out$dim())])
    as.array(out)
  }
  
  # Get time points
  if (model_type == "DeepHit") {
    times <- explainer$model$time_bins
  } else {
    times <- explainer$model$t_orig
  }
  
  # Get the predict survival function
  predict_survival_function <- function(model, newdata, times) {
    input <- torch_tensor(as.matrix(newdata))
    input <- model$preprocess_fun(input)
    out <- model(input, target = "survival")[[1]]
    out <- out$squeeze(dim = seq_len(out$dim())[-c(1, out$dim())])
    as.array(out)
  }
  
  # Get the predict cumulative hazard function
  target <- switch(model_type, DeepHit = "cif", CoxTime = "cum_hazard", DeepSurv = "cum_hazard")
  predict_cumulative_hazard_function <- function(model, newdata, times) {
    input <- torch_tensor(as.matrix(newdata))
    input <- model$preprocess_fun(input)
    out <- model(input, target = target)[[1]]
    out <- out$squeeze(dim = seq_len(out$dim())[-c(1, out$dim())])
    as.array(out)
  }
  
  # Create survex explainer
  explain_survival(
    model = model,
    data = data,
    verbose = verbose,
    y = y,
    predict_function = predict_function,
    label = model_type,
    times = times,
    predict_survival_function = predict_survival_function,
    predict_cumulative_hazard_function = predict_cumulative_hazard_function
  )
}