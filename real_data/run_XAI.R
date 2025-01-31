library(Survinng)
library(torch)
library(torchvision)

# Define multi modal model
MultiModalModel <- nn_module(
  "MultiModalModel",
  initialize = function(net_images, tabular_features, n_out = 1, n_img_out = 64,
                        include_time = FALSE, out_bias = FALSE) {
    self$net_images <- net_images

    input_dim <- n_img_out + length(tabular_features) + ifelse(include_time, 1, 0)
    self$fc1 <- nn_linear(input_dim, 256)
    self$fc2 <- nn_linear(256, 128)
    self$relu <- nn_relu()
    self$drop_0_3 <- nn_dropout(0.3)
    self$drop_0_5 <- nn_dropout(0.4)
    self$out <- nn_linear(128, n_out, bias = out_bias)
  },

  forward = function(input, time = NULL) {
    img <- input[[1]]
    tab <- input[[2]]
    img <- self$net_images(img)
    img <- self$drop_0_5(img)

    if (!is.null(time)) {
      x <- torch_cat(list(img, tab, time), dim = 2)
    } else {
      x <- torch_cat(list(img, tab), dim = 2)
    }

    x <- self$drop_0_3(x)
    x <- self$relu(self$fc1(x))
    x <- self$drop_0_3(x)
    x <- self$relu(self$fc2(x))
    x <- self$drop_0_3(x)
    x <- self$out(x)

    x
  }
)


# Load model--------------------------------------------------------------------

# Load model metadata
model_metadata <- read.csv("results/metadata.csv")
n_img_out <- model_metadata$n_img_out
n_out <- model_metadata$n_out
out_bias <- as.logical(model_metadata$out_bias)
n_tab_feat <- model_metadata$Number.of.tabular.features

# Load model state dict
model_state_dict <- load_state_dict("results/model.pt")

# Replicate model
net_image <- torchvision::model_resnet18(num_classes = n_img_out)
model <- MultiModalModel(net_image, tabular_features = rep(1, n_tab_feat),
                         n_out = n_out, n_img_out = n_img_out, out_bias = out_bias)

# Load model state dict
model <- model$load_state_dict(model_state_dict)
model$eval()

# Load data to be explained-----------------------------------------------------
data <- read.csv("data/tabular_data.csv")
test_images <- list.files("data/test", full.names = FALSE)

# Filter data
data <- data[data$full_path %in% test_images, ]
data <- data[1:5, ]

data_tab <- torch_tensor(as.matrix(data[, c(1, 2, 3)]))
data_img <- torch_stack(lapply(data$full_path, function(x) {
  base_loader(paste0("data/test/", x)) %>%
    (function(x) x[,,1:3]) %>%
    transform_to_tensor() %>%
    transform_center_crop(452) %>%
    transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
}), dim = 1)

# Explain model-----------------------------------------------------------------
exp_deephit <- explain(model, list(data_img, data_tab), model_type = "deephit",
                       time_bins = seq(0, 15, length.out = 50))

grad <- surv_grad(exp_deephit)













