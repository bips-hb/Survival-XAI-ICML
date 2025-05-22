################################################################################
#                   Setup script for the Survinng paper
#
#   * Check if all R packages are installed
#   * Install missing R packages
#   * Install Python environment including pycox
################################################################################

required_pks <- c(
  # Survival packages
  "simsurv", "survival", "survminer", "SurvMetrics", "Survinng", "survex",
  "survivalmodels",
  
  # Plotting and other useful packages
  "ggplot2", "cowplot", "viridis", "dplyr", "tidyr", "torch", "reticulate", "callr",
  "here", "data.table", "batchtools"
)

# Check if devtools is installed
if (!require("devtools")) {
  install.packages("devtools")
}

# Install all required packages
for (pk in required_pks) {
  if (!require(pk, character.only = TRUE)) {
    if (pk == "Survinng") {
      devtools::install_github("bips-hb/Survinng")
    } else if (pk == "SurvMetrics") {
      devtools::install_github("skyee1/SurvMetrics")
    } else {
      install.packages(pk)
    }
  }
}

# Install conda environment for Python and pycox
if (!reticulate::condaenv_exists("Survinng_paper")) {
  reticulate::conda_create("Survinng_paper", environment = "env_survinng_paper.yml",
                           pip = TRUE, pip_ignore_installed = TRUE,
                           additional_install_args = "--index-url https://download.pytorch.org/whl/cpu")
}

# Disable GPU usage
Sys.setenv(CUDA_VISIBLE_DEVICES = "")

