
## Example on Real Multi-modal Medical Data

This folder contains the real-world case study from our paper, where we apply and
compare time-dependent explainability methods for survival neural survival models. We 
train a CNN-based DeepHit model on multimodal data from glioma patients 
(histology + clinical features), and apply SurvSHAP(t) and GradSHAP(t) to 
assess feature contributions over time and benchmark their runtime.

## 📁 Folder Structure

The following files are used for the main analysis:

- `data/`: Contains the preprocessed data (tabular + images) and the 
corresponding metadata. It includes the images resized to 256x256 (226x226 after cropping)
in the folder `data/full/` and the small images down-scaled to 40x40 (32x32 after cropping) 
in the folder `data/small/`. The data is split into training and test sets, respectively.
**Note:** The data originates from the 
[TCGA](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga) 
and was cleaned and preprocessed by 
[Mobadersany et al. (2018)](https://doi.org/10.1073/pnas.1717139115). 
It is publicly provided in the corresponding repository: 
[https://github.com/PathologyDataScience/SCNN](https://github.com/PathologyDataScience/SCNN)

- `run_XAI.R`: Applies GradSHAP(t) to the trained *full-sized* model and generates 
explanation plots used in the manuscript. The results are saved in the folder
`results/figures/`.

- `run_XAI_small.R`: Applies the GradSHAP(t) and SurvSHAP(t) to a *smaller* (i.e., image size of 32x32) 
model for runtime benchmarking purpose. The results are saved in the folder
`results/figures/`.

- `results/`: Stores the results, i.e., the trained models and the generated
explanations.

The following files are used for training the models and reproducing the data:

- `run_DeepHit.py`: Trains a multimodal DeepHit model that combines clinical 
features (age, sex, IDH mutation, 1p/19q codeletion) with CNN-derived images. 
The model outputs a discretized survival distribution for each patient. The model is
saved in the folder `results/deephit/`.

- `run_DeepHit_small.py`: Trains a smaller version of the DeepHit model for
runtime benchmarking, i.e., the original images are down-scaled to 32x32 pixels.  
The model is saved in the folder `results/deephit_small/`.

- `prepare_data.R`: Loads the raw data from Mobadersany et al. (especially the file `all_dataset.csv`), 
selects relevant features, and creates a new csv file with the selected features and
the corresponding image names. It's saved as `all_data_custom.csv`.

- `preprocess_data.py`: Preprocesses the data, i.e., it loads the file `all_data_custom.csv`
and downscales the images to 256x256 pixels (226x226 after cropping) and 40x40 pixels (32x32 after cropping).
The results are saved in the folders `data/full/` and `data/small/`, respectively.

- `environment.yml`: Conda setup file with all required Python and R dependencies 

## 🚀 Reproducing the Results

### Model training (not necessary!)
In order to train the models, a sufficient deep learning infrastructure is required,
i.e., a GPU and the corresponding CUDA drivers. We provided the necessary
dependencies in the `environment.yml` file. However, since the training is not
the main focus of this repository and the training results can vary due to
GPU-randomness, we provide the trained models in the `results/` folder and can be
loaded directly. Nevertheless, you can train the models from scratch by running the
scripts `run_DeepHit.py` and `run_DeepHit_small.py`. The models are saved in the
`results/deephit/` and `results/deephit_small/` folders, respectively.

### Model Explanation

To reproduce the results, you can run the scripts `run_XAI.R` and `run_XAI_small.R`, 
i.e., run the code:
```
Rscript run_XAI.R
Rscript run_XAI_small.R
```
The results are saved in the `results/figures/` folder.

## 📊 Dataset

We use the publicly available multi-modal dataset from 
[Mobadersany et al. (2018)](https://doi.org/10.1073/pnas.1717139115), 
which includes:

- 🖼️ **Histopathological whole-slide images** of regions of interest
of whole-slide image tissue sections
- 🧬 **Four tabular features**:
  - 👵 Age  
  - 🚻 Sex  
  - ⚛️ IDH mutation status (present/absent)  
  - 🧩 1p/19q codeletion status (present/absent)

The data were collected from the **TCGA LGG** and **GBM** cohorts. We built upon
the Docker-based release from the authors: 
[📦 PathologyDataScience/SCNN](https://github.com/PathologyDataScience/SCNN), 
applying minimal preprocessing to extract the necessary image and tabular inputs 
(see the script `prepare_data.R`).

**Note:** The original data is not included in this repository due to its size.
However, the preprocessed data is available in the `data/` folder. To reproduce our
preprocessed data, you can run the script `prepare_data.R` for the original csv file
`all_dataset.csv` and, then, `preprocess_data.py` to create the respective down-scaled
datasets in the folder `data/`.

## 📖 Reference

P. Mobadersany, S. Yousefi, M. Amgad, D.A. Gutman, J.S. Barnholtz-Sloan, J.E. 
Velázquez Vega, D.J. Brat, & L.A.D. Cooper, *Predicting cancer outcomes from 
histology and genomics using convolutional networks*, Proc. Natl. Acad. Sci. 
U.S.A. 115 (13) E2970-E2979, [https://doi.org/10.1073/pnas.1717139115](https://doi.org/10.1073/pnas.1717139115) (2018).

📦 GitHub: [https://github.com/PathologyDataScience/SCNN](https://github.com/PathologyDataScience/SCNN)

