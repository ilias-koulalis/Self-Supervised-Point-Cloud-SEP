# %%
import os
os.environ['CUDA_PATH'] = '/media/gambino/students_workdir/ilias/miniconda3/envs/torch'

# %%
import torch
from torch.utils.data import DataLoader, TensorDataset
from models import PCAE, PointNetAE
from utils.helper_train import train_ae
from utils.helper_plotting import plot_latent_space
import matplotlib.pyplot as plt
import numpy as np
import gc
import json
import xarray
import pykeops

# %%
data_path = '/media/gambino/students_workdir/ilias/colonCancerPatient2_tiles/tile40_z3_dc_tacco_gene_expression_coords_masked.nc'
dataset = xarray.load_dataset(data_path)

# %%
np.random.seed(122)
shuffled_indices= np.random.permutation(dataset.index)
dataset_len = len(shuffled_indices)
train_indices = shuffled_indices[: int(0.8 * dataset_len)]
val_indices = shuffled_indices[int(0.8 * dataset_len) : int(0.9 * dataset_len)]
test_indices = shuffled_indices[int(0.9 * dataset_len) :]

# %%
train_dataset = dataset.isel(index=train_indices)
train = torch.Tensor(train_dataset.trans_coords.values)
val_dataset = dataset.isel(index=val_indices)
val = torch.Tensor(val_dataset.trans_coords.values)
test_dataset = dataset.isel(index=test_indices)
test = torch.Tensor(test_dataset.trans_coords.values)

# %%
train_loader = DataLoader(dataset=train,
                            batch_size=32,
                            num_workers=2,
                            shuffle=True,
                            drop_last=True)

# %%
val_loader = DataLoader(dataset=val,
                            batch_size=32,
                            num_workers=2,
                            shuffle=True,
                            drop_last=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#############################################################
model = PointNetAE(15,450, loss_fn='geom_sinkhorn')
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)  

num_epochs=15
log_dict = train_ae(num_epochs=num_epochs, model=model, 
                        optimizer = optimizer, device = device, 
                        train_loader = train_loader,
                        val_loader = val_loader,
                        checkpoint_interval=10,
                        dataset = data_path,
                        logging_interval=1000, save_model=True)