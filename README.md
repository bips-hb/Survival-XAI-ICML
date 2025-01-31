# Gradient-based Explanations for Deep Learning Survival Models

This repository contains the code and material to reproduce the results of the 
manuscript "Gradient-based Explanations for Deep Learning Survival Models".
The paper is currently under review.

## 📁 Repository Structure

- `setup.R`: R environment setup script that installs required packages,
   the necessary conda environment `Survinng_paper`, and sets global options
- `Sim_time_dependent.Rmd`: Simulation for time-dependent features. The results
   used in the paper are stored in the notebook `Sim_time_dependent.html` and
   figures are saved in the `figures_paper/` directory.
- `Sim_time_independent.Rmd`: Simulation for time-independent features. The results
   used in the paper are stored in the notebook `Sim_time_independent.html` and
   figures are saved in the `figures_paper/` directory.
-  `Sim_GradSHAP`: Simulation for comparing GradSHAP(t) and SurvSHAP(t) on 
   time-independent features regarding runtime, local accuarcy and feature ranking.
- `real_data/`: Scripts for reproducing the results on the real data example.
- `Survinng.zip`: The corresponding R package for the paper.
- `figures_paper/`: Directory for storing the figures used in the paper.

## 🚀 Reproducing the Results

* To reproduce the results, from "TIME-INDEPENDENT EFFECTS" Section, run the 
  RMarkdown file `Sim_time_independent.Rmd` and the results will be stored 
  `Sim_time_independent.html` and the figures in the `figures_paper/` 
  directory.
  
* To reproduce the results, from "TIME-DEPENDENT EFFECTS" Section, run the
  RMarkdown file `Sim_time_dependent.Rmd` and the results will be stored 
  `Sim_time_dependent.html` and the figures in the `figures_paper/` 
  directory.
  
* To reproduce the results, from "GRADSHAP(T) VS. SURVSHAP(T)" Section, run the
  R file `Sim_GradSHAP/Sim_GradSHAP.R` and the figures will be stored in the 
  `figures_paper/` directory. Note: This simulation is computationally expensive
  and conducts a simulation study using `batchtools`.
  
* To reproduce the results, from "REAL DATA EXAMPLE" Section, we refer to
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
- `Survinng` (from the `Survinng.zip` file)
- `survex`
- `survivalmodels`
- `torch` (necessary for the `Survinng` package)

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
