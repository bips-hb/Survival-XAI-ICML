##############################################################
# This file contains utility functions for the main scripts  #
##############################################################
import os
import torch
import torch.nn as nn
import torchtuples as tt
import numpy as np
from PIL import Image
from skimage import io
from tqdm import tqdm
import pandas as pd
import warnings
from pycox.evaluation import EvalSurv


# Preprocessing functions ------------------------------------
def resize_and_save_images(input_folder, output_folder, size=(256, 256)):
    """
    Loads images from a folder with subfolders, resizes them, and saves them to a new folder structure 
    that mirrors the original structure.
    
    :param input_folder: Path to the input folder containing subfolders with images.
    :param output_folder: Path to the output folder where the resized images will be saved.
    :param size: Tuple specifying the target size (width, height) of the images.
    """
    print("\n-------- IMAGE PREPROCESSING --------")
    print(f"Resizing images from '{input_folder}' to '{output_folder}' and rescaling them to size {size}.")

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        warnings.warn(f"Output folder '{output_folder}' already exists. Existing files may be overwritten.")

    # Walk through the input folder and process files
    for root, dirs, files in os.walk(input_folder):
        # Calculate the relative path of the current directory to the input folder
        relative_path = os.path.relpath(root, input_folder)
        # Create the corresponding directory in the output folder
        target_folder = os.path.join(output_folder, relative_path)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        if len(files) != len(dirs):
            print(f"Processing images in subfolder '{relative_path}'.")
            for file in tqdm(files):
                # Process only image files based on extensions
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                    input_file_path = os.path.join(root, file)
                    output_file_path = os.path.join(target_folder, file)

                    try:
                        # Open the image, resize it, and save it to the output folder
                        with Image.open(input_file_path) as img:
                            img_resized = img.resize(size)
                            img_resized.save(output_file_path)
                    except Exception as e:
                        print(f"Error processing {input_file_path}: {e}")
            print(f"Processed {len(files)} images in subfolder '{relative_path}'.")
    print("Image resizing and saving completed.")


def load_tab_data(tabular_file, tab_features, out_columns):
    """
    Load the tabular data from a CSV file.
    
    :param tabular_file: Path to the CSV file containing the tabular data.
    :return: DataFrame containing the tabular data.
    """
    print("\n-------- LOADING DATA --------")
    print(f"Loading tabular data from '{tabular_file}'.")
    
    # Read the tabular data
    tabular_data = pd.read_csv(tabular_file)

    # Use binary encoding for column 'gender'
    tabular_data['gender']  = (tabular_data['gender'] == 'male').astype(int)
    tabular_data['Survival.months'] = tabular_data['Survival.months'] / 365
    tabular_data['censored'] = tabular_data['censored'].astype(float)
    tabular_data['idh.mutation'] = tabular_data['idh.mutation'].fillna(-1) # Fill missing values with -1
    tabular_data['codeletion'] = tabular_data['codeletion'].fillna(-1) # Fill missing values with -1
    tabular_data['Age (at index)'] = tabular_data['Age (at index)'] / 100
    print(f"Loaded tabular data with {len(tabular_data)} rows and {len(tabular_data.columns)} columns.")
    
    # Remove unnecessary columns
    tabular_data = tabular_data[tab_features + out_columns + ['TCGA.ID', 'full_path', 'train']]
    print(f"Selected {len(tabular_data.columns)} columns: {', '.join(tabular_data.columns)}.")

    return tabular_data

# Collate function for the dataloader (from pycox)
def collate_fn(batch):
    """Stacks the entries of a nested tuple"""
    return tt.tuplefy(batch).stack()

# Create a torch dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tabular_data, image_dir, tab_features, transform=None, device = torch.device('cpu')):
        self.tabular_data = tabular_data
        self.image_dir = image_dir
        self.tab_features = tab_features
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.tabular_data)

    def __getitem__(self, idx):

        # Get the images
        img_name = os.path.join(self.image_dir, self.tabular_data['full_path'].iloc[idx])
        image = torch.tensor(io.imread(img_name)).to(self.device) / 255
        image = image[:,:,:3]

        # Make channels first
        image = image.permute(2, 0, 1)

        # Transform the images (if required)
        if self.transform:
            image = self.transform(image)

        # Get the tabular data
        tabular = torch.tensor(self.tabular_data[self.tab_features].iloc[idx].to_numpy(dtype=np.float32), device=self.device)

        # Get outcome (censored and survival time)
        event = np.array(self.tabular_data['censored'].iloc[idx], dtype=np.float32)
        time = np.array(self.tabular_data['Survival.months'].iloc[idx])


        return [image, tabular], tt.tuplefy(time, event).to_tensor()



# Neural network model ----------------------------------------

# Define multi modal model
class MultiModalModel(nn.Module):
    def __init__(self, net_images, tabular_features, n_out = 1, n_img_out = 64, include_time = False, out_bias = False):
        super(MultiModalModel, self).__init__()
        self.net_images = net_images
        if (include_time):
            self.fc1 = nn.Linear(n_img_out + len(tabular_features) + 1, 256)
        else:
            self.fc1 = nn.Linear(n_img_out + len(tabular_features), 256)
        self.fc2 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.drop_0_3 = nn.Dropout(0.3)
        self.drop_0_4 = nn.Dropout(0.3)
        self.out = nn.Linear(128, n_out, bias = out_bias)

    def forward(self, img, tab, time = None):
        img = self.net_images(img)
        img = self.drop_0_4(img)

        if time is not None:
            x = torch.cat((img, tab, time), dim=1)
        else:
            x = torch.cat((img, tab), dim=1)
        x = self.drop_0_3(x)
        x = self.relu(self.fc1(x))
        #x = self.drop_0_3(x)
        x = self.relu(self.fc2(x))
        x = self.out(x)

        return x


class Concordance(tt.cb.MonitorMetrics):
    def __init__(self, dl, per_epoch=1, verbose=True, nn_type = 'cox'):
        super().__init__(per_epoch)
        self.dl = dl
        self.verbose = verbose
        self.nn_type = nn_type
    
    def on_epoch_end(self):
        super().on_epoch_end()
        if self.epoch % self.per_epoch == 0:
            images = []
            tabulars = []
            times = []
            events = []
            for (img, tab), (time, event) in self.dl:
                images.append(img.cpu().detach())
                tabulars.append(tab.cpu().detach())
                times.append(time.cpu().detach())
                events.append(event.cpu().detach())
            images = torch.cat(images, dim = 0)
            tabulars = torch.cat(tabulars, dim = 0)
            times = torch.cat(times, dim = 0)
            events = torch.cat(events, dim = 0)
            if self.nn_type == 'cox':
                _ = self.model.compute_baseline_hazards(input = (images, tabulars), target = (times, events))
            surv = self.model.predict_surv_df(input = (images, tabulars))
            ev = EvalSurv(surv, times.numpy(), events.numpy())
            concordance = ev.concordance_td()
            self.append_score('concordance', concordance)
            if self.verbose:
                print('lr: {:,.6f}'.format(self.model.optimizer.param_groups[0]['lr']), end = '  |  ')
                print('c-Index (test): {:,.4f}'.format(concordance), end = '  |  ')
