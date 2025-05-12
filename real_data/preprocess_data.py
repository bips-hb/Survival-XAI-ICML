import os
import numpy as np
import torch
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
from skimage import io
import matplotlib.pyplot as plt
import random

# Import utility functions from utils.py
from utils import *

# Create data folder
if not os.path.exists('data'):
    os.makedirs('data')

# Create the full data (256x256 pixels) ----------------------------------------

print("------------------------------------")
print("Creating small data (256x256 pixels)")
print("------------------------------------")

# Set seeds for reproducibility
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)

# Set global variables
TAB_FEATURES = ['Age (at index)', 'gender', 'idh.mutation', 'codeletion'] #'ethnicity'
OUT_COLUMNS = ['censored', 'Survival.months']
IMAGE_SIZE = (256, 256)

# Resize images to the desired size (only needs to be done once)
image_dir_src = '/opt/example-data/TCGA-SCNN/' # Path to the input folder
image_dir = 'data/full/'   # Path to the output folder
resize_and_save_images(image_dir_src, image_dir, size=IMAGE_SIZE)

# Prepare the tabular data
tabular_file = '/opt/example-data/TCGA-SCNN/all_data_custom.csv'
tabular_data = load_tab_data(tabular_file, TAB_FEATURES, OUT_COLUMNS)

# Save as a CSV file
tabular_data.to_csv('data/deephit_tabular_data.csv', index=False)

print(tabular_data.head())

# Create the small data (40x40 pixels) -----------------------------------------

print("----------------------------------")
print("Creating small data (40x40 pixels)")
print("----------------------------------")

# Set seeds for reproducibility
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)

# Set global variables
TAB_FEATURES = ['Age (at index)', 'gender', 'idh.mutation', 'codeletion'] #'ethnicity'
OUT_COLUMNS = ['censored', 'Survival.months']
IMAGE_SIZE = (40, 40)

# Resize images to the desired size (only needs to be done once)
image_dir_src = '/opt/example-data/TCGA-SCNN/' # Path to the input folder
image_dir = 'data/small/'   # Path to the output folder
resize_and_save_images(image_dir_src, image_dir, size=IMAGE_SIZE)

# Prepare the tabular data
tabular_file = '/opt/example-data/TCGA-SCNN/all_data_custom.csv'
tabular_data = load_tab_data(tabular_file, TAB_FEATURES, OUT_COLUMNS)

# Save as a CSV file
tabular_data.to_csv('data/deephit_tabular_data_small.csv', index=False)

print(tabular_data.head())
