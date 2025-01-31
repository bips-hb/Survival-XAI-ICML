import os
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torchtuples as tt
import torchvision
from torchvision.transforms import v2
import pycox
from pycox.models import DeepHitSingle, DeepHit
from pycox.utils import kaplan_meier
from pycox.evaluation import EvalSurv
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
from skimage import io
import matplotlib.pyplot as plt
import random


# Empty the CUDA cache
torch.cuda.empty_cache()

# Import utility functions from utils.py
from utils import *

# Set seeds for reproducibility
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)

# Set the device (CPU or GPU)
if torch.cuda.is_available(): 
    dev = "cuda:1" 
else: 
    dev = "cpu" 

# Set global variables
TAB_FEATURES = ['Age (at index)', 'gender', 'idh.mutation'] #'ethnicity'
OUT_COLUMNS = ['censored', 'Survival.months']
IMAGE_SIZE = (512, 512)
IMAGE_SIZE_CROPPED = (452, 452)
BATCH_SIZE = 64
DEVICE = torch.device(dev)

print("------------------------------------------")
print("|  CoxTime: Multi-modal Survival Model   |")
print("------------------------------------------")

# Load data and crate dataset ---------------------------------------------------

# Resize images to the desired size (only needs to be done once)
image_dir_src = '/opt/example-data/TCGA-SCNN/' # Path to the input folder
image_dir = 'data/'   # Path to the output folder
#resize_and_save_images(image_dir_src, image_dir, size=IMAGE_SIZE)

# Prepare the tabular data
tabular_file = '/opt/example-data/TCGA-SCNN/all_data_custom.csv'
tabular_data = load_tab_data(tabular_file)

# Deephit requires the time and events to be transformed
num_durations = 15
time = tabular_data['Survival.months'].values
events = -tabular_data['censored'].values + 1
labtrans = DeepHitSingle.label_transform(num_durations)
time, events  = labtrans.fit_transform(time, events)
tabular_data['censored'] = events
tabular_data['Survival.months'] = time

print(f"Number of total instances:      {len(tabular_data)}")
print(f"Number of training instances:   {len(tabular_data[tabular_data['train']])}")
print(f"Number of test instances:       {len(tabular_data[~tabular_data['train']])}")
print(f"Number of tabular features:     {len(TAB_FEATURES)}")
print(f"Image size:                     {IMAGE_SIZE_CROPPED}")
print(f"Imgae size (un-cropped):        {IMAGE_SIZE}")
print(f"Number of discrete timepoints:  {num_durations}")



# Set transformations for the images
transforms = v2.Compose([
    v2.ToDtype(torch.float32),
    v2.ColorJitter(brightness=.15, hue=.1),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomCrop(size=IMAGE_SIZE_CROPPED),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transforms_test = v2.Compose([
    v2.ToDtype(torch.float32),
    v2.CenterCrop(size=IMAGE_SIZE_CROPPED),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Create torch datasets
train_ds = CustomDataset(tabular_data[tabular_data['train']], image_dir + 'train/', TAB_FEATURES, transform = transforms, device = DEVICE)
test_ds = CustomDataset(tabular_data[~tabular_data['train']], image_dir + 'test/', TAB_FEATURES, transform = transforms_test, device = DEVICE)

# Create torch dataloaders
dl_train = torch.utils.data.DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
dl_test = torch.utils.data.DataLoader(test_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Define the model --------------------------------------------------------------
n_img_out = 128

# Use a ResNet18 model as the image encoder
net_images = torchvision.models.resnet34(num_classes = n_img_out).to(DEVICE)

# Create multi-modal model
model = MultiModalModel(net_images, TAB_FEATURES, n_out = num_durations, n_img_out = n_img_out).to(DEVICE)

# Print model information
print('\n-------- MODEL INFORMATION --------')
print('Image encoder architecture: ', net_images.__class__.__name__)
print('Number of parameters in the image encoder:     {:,.0f}'.format(sum(p.numel() for p in net_images.parameters())))
print('Number of parameters in the multi-modal model: {:,.0f}'.format(sum(p.numel() for p in model.parameters())))

# Create and train survival model ------------------------------------------------
epochs_warmup = 20
epochs_full = 10
lr_warmup = 0.005 #0.0005
lr_full = 5e-4 #5e-5

print('\n-------- TRAINING --------')
from pycox.models import loss as pycox_loss
from pycox.models.data import pair_rank_mat

# See https://github.com/havakv/pycox/issues/79
def deephit_loss(scores, labels, censors):
    rank_mat = pair_rank_mat(labels.cpu().numpy(), censors.cpu().numpy())
    rank_mat = torch.from_numpy(rank_mat)
    rank_mat = rank_mat.to(DEVICE)
    loss_single = pycox_loss.DeepHitSingleLoss(0.2, 0.1)
    loss = loss_single(scores, labels, censors, rank_mat)
    return loss

optimizer = tt.optim.Adam(lr_warmup, weight_decay=1e-5, amsgrad=True)
surv_model = DeepHitSingle(model, optimizer, device = DEVICE, duration_index=labtrans.cuts, loss = deephit_loss)
callbacks = [Concordance(dl_test, per_epoch = 5, nn_type = 'deephit')]

# Fit the model (warm-up rounds with a higher learning rate)
print('Warm-up rounds (', epochs_warmup, ' epochs and inital lr = ', lr_warmup,')', sep = '')
log = surv_model.fit_dataloader(dl_train, epochs=epochs_warmup // 2, callbacks = callbacks, verbose=True)
surv_model.optimizer = tt.optim.Adam(lr_warmup / 5, weight_decay=1e-5, amsgrad = True)
print('Reducing learning rate to ', lr_warmup / 5, sep = '')
log = surv_model.fit_dataloader(dl_train, epochs= epochs_warmup - epochs_warmup // 2, callbacks = callbacks, verbose=True)
print('Warm-up rounds complete. Starting full training (max. ', epochs_full, ' epochs)', sep = '')

# Fit the model (full training)
callbacks = [
    tt.cb.EarlyStopping(patience=20, file_path='checkpoints/deephit_model.pt'), 
    Concordance(dl_test, per_epoch=1, nn_type = 'deephit')
]
surv_model.optimizer = tt.optim.Adam(lr_full, weight_decay=1e-5, amsgrad = True)
log = surv_model.fit_dataloader(dl_train, epochs_full // 2, callbacks = callbacks, verbose = True, val_dataloader = dl_test)
print('Reducing learning rate to ', lr_full / 5, sep = '')
surv_model.optimizer = tt.optim.Adam(lr_full / 5, weight_decay=1e-5, amsgrad = True)
log = surv_model.fit_dataloader(dl_train, epochs_full - epochs_full // 2, callbacks = callbacks, verbose = True, val_dataloader = dl_test)
print('Training complete.')

# Save the model
surv_model.save_model_weights('results/DeepHit_model.pt')

# Show loss plot
plt.figure(figsize=(12, 6))
_ = log.plot()
plt.show()
plt.savefig('results/DeepHit_loss.png')


# Evaluation ---------------------------------------------------------------------
print('\n-------- EVALUATION --------')

# Calculate results on the test set
dl_test = torch.utils.data.DataLoader(test_ds, len(test_ds), shuffle=True, collate_fn=collate_fn)
inputs, target = next(iter(dl_test))
surv = surv_model.predict_surv_df(input = inputs)

# Show Survival curves
plt.figure(figsize=(12, 6))
surv.iloc[:, :5].plot()
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')
plt.show()
plt.savefig('results/DeepHit_survival.png')

ev = EvalSurv(surv, target[0].cpu().detach().numpy(), target[1].cpu().detach().numpy(), censor_surv='km')
print('Concordance index:', ev.concordance_td())

def simps(y, x=None, dx=1.0, axis=-1, even='avg'):
    return scipy.integrate.simpson(y, x=x, dx=dx, axis=axis)
scipy.integrate.simps = simps

plt.figure(figsize=(12, 6))
time_grid = np.linspace(0, target[0].max().cpu().detach().numpy(), 200)
_ = ev.brier_score(time_grid).plot()
plt.show()
plt.savefig('results/DeepHit_brier.png')

print('Integrated Brier score:', ev.integrated_brier_score(time_grid))