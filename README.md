# Gradient-based Explanations for Deep Learning Survival Models

This repository contains the code and material to reproduce the results of the 
manuscript "Gradient-based Explanations for Deep Learning Survival Models" 
accepted for publication in the proceedings of the *Forty-Second International 
Conference on Machine Learning (ICML) 2025*.

The reproduction material is based on the accompanying R package `survinng`
available on [github/bips-hb](https://github.com/bips-hb/survinng).

## 📁 Repository Structure

- `setup.R`: R environment setup script that installs required packages,
   the necessary conda environment `Survinng_paper`, and sets global options.
   It also installs the `survinng` package from the mentioned GitHub repository.
- `Sim_time_dependent.Rmd`: Simulation for time-dependent features. The results
   used in the paper are stored in the notebook `Sim_time_dependent.html` and
   figures are saved in the `figures_paper/` directory.
- `Sim_time_independent.Rmd`: Simulation for time-independent features. The results
   used in the paper are stored in the notebook `Sim_time_independent.html` and
   figures are saved in the `figures_paper/` directory.
-  `Sim_GradSHAP`: Simulation for comparing GradSHAP(t) and SurvSHAP(t) on 
   time-independent features regarding runtime, local accuarcy and feature ranking.
- `real_data/`: Scripts for reproducing the results on the real data example.
- `figures_paper/`: Directory for storing the figures used in the paper.

## 🚀 Reproducing the Results

* To reproduce the results, from Section 5.1.1 **TIME-INDEPENDENT EFFECTS**, run the 
  RMarkdown file `Sim_time_independent.Rmd` and the results will be stored 
  `Sim_time_independent.html` and the figures in the `figures_paper/` 
  directory.
  
* To reproduce the results, from Section 5.1.2 **TIME-DEPENDENT EFFECTS**, run the
  RMarkdown file `Sim_time_dependent.Rmd` and the results will be stored 
  `Sim_time_dependent.html` and the figures in the `figures_paper/` 
  directory.
  
* To reproduce the results, from Section 5.2 **GradSHAP(t) vs. SurvSHAP(t)**, run 
  corresponding scripts in the directory `Sim_GradSHAP/`, i.e.,
  - `sim_locacc.R`: for the local accuracy comparison
  - `sim_runtime.R`: for the runtime comparison
  - `sim_global_imp.R`: for the  global feature ranking comparison.
  The figures will be stored in the `figures_paper/` directory.
  **Note:** This simulation is computationally expensive and conducts a 
  simulation study using `batchtools`.
  
* To reproduce the results, from the Section 5.2 **Practical Feasibility** and
  Section 5.3 **Example on Real Multi-modal Medical Data**, we refer to
  the README file in the folder `real_data/`.

## 📚 Requirements

The script `setup.R` tries to install the necessary packages and the conda 
environment `Survinng_paper` (see file `env_survinng_paper.yml`). 
It installs the following R packages:

**Survival packages**

- `simsurv`
- `survival`
- `survminer`
- `SurvMetrics`
- `survinng` (from [github/bips-hb](https://github.com/bips-hb/survinng))
- `survex`
- `survivalmodels`
- `torch` (necessary for the `survinng` package)

**Plotting and other useful packages**

- `ggplot2`
- `cowplot`
- `viridis`
- `dplyr`
- `tidyr`
- `reticulate`
- `callr`
- `here`
- `data.table`
- `batchtools`
