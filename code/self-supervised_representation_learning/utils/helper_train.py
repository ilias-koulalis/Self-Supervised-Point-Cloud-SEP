import torch
from torch import nn
from torch.optim.lr_scheduler import CyclicLR,ReduceLROnPlateau
import time
import numpy as np
import os
import json
import gc
from copy import deepcopy


def train_ae(num_epochs, model, optimizer, device, 
                 train_loader, val_loader, dataset: str,
                 logging_interval=100,
                 checkpoint_interval=10,
                 save_model=True):
    
    params = {'model':model._get_name(), 'layers':[module.__str__() for module in model.children()],
              'trainable_parameters':sum([param.flatten().size(0) for param in model.parameters()]),
              'num_epochs':num_epochs}
    
    log_dict = {'train_reconstruction_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'val_reconstruction_loss_per_epoch': [],
                'val_reconstruction_loss_per_batch': []}
        
    start_time = time.time()
    
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, cooldown=3, verbose=True)
    
    best_val_loss = float('inf')

    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.
        
        #TRAINING
        for train_batch_idx, features in enumerate(train_loader):
                           
            features = features.to(device)

            # FORWARD
            _, decoded = model(features)
            
            # BACKPROP
            optimizer.zero_grad(set_to_none=True)

            rec_loss = model.loss(decoded, features)                
            running_loss = running_loss + rec_loss.item()
            
            rec_loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING            
            log_dict['train_reconstruction_loss_per_batch'].append(rec_loss.item())
            
            if train_batch_idx % logging_interval == 0:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, train_batch_idx,
                          len(train_loader), rec_loss))
        
        average_loss  = running_loss / (train_batch_idx + 1)

        torch.cuda.empty_cache()
        gc.collect()
               
        #VALIDATION
        model.eval()
        with torch.no_grad(): 
            
            running_vloss = 0.
                  
            for val_batch_idx, features in enumerate(val_loader):
                            
                features = features.to(device)

                # FORWARD
                _, decoded = model(features)
                
                val_recloss = model.loss(decoded, features)
                running_vloss = running_vloss + val_recloss.item()
                
                # LOGGING
                log_dict['val_reconstruction_loss_per_batch'].append(val_recloss.item())
                
                if val_batch_idx % logging_interval == 0:
                    print('Epoch: %03d/%03d | Batch %04d/%04d | Validation Loss: %.4f'
                        % (epoch+1, num_epochs, val_batch_idx,
                            len(val_loader), val_recloss))       
        
            average_vloss = running_vloss / (val_batch_idx + 1)
        
        #Pass validation loss to learning rate scheduler 
        scheduler.step(average_vloss)
                
        #Keep best model in memory
        if average_vloss <= best_val_loss:
            best_val_loss = average_vloss
            state_dict = deepcopy(model.state_dict())
        
        # EPOCH STATS  
        print('***Epoch: %03d/%03d | Train Loss: %.3f' % (
                epoch+1, num_epochs, average_loss))
        log_dict['train_reconstruction_loss_per_epoch'].append(average_loss)

        print('***Epoch: %03d/%03d | Validation Loss: %.3f' % (
                epoch+1, num_epochs, average_vloss))
        log_dict['val_reconstruction_loss_per_epoch'].append(average_vloss)

        if epoch % checkpoint_interval==0:
            filename = f'checkpoints/checkpoint_{model._get_name()}_{model.k}_{model.loss_fn}'
            if not os.path.exists(filename):
                os.mkdir(filename)
            torch.save(state_dict, f'{filename}/state_dict.pth')
            
            model_params = { k:v for k,v in vars(model).items() if not k.startswith('_') }

            with open(f'{filename}/hyperparams.json', "w") as file:
                json.dump(model_params|params, file)
            
            with open(f'{filename}/log_dict.json', "w") as file:
                json.dump(log_dict, file)
        
        torch.cuda.empty_cache()
        gc.collect()

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    
    if save_model:
        path = f'{model._get_name()}_{model.k}k_{model.num_points}p_{model.loss_fn}_{num_epochs}e_{dataset.split("/")[-1]}'
        # PCAE_15_150p_cd-t_50
        path = os.path.join('trained_models',path)
        if not os.path.exists(path):
            os.mkdir(path)
        
        model_params = { k:v for k,v in vars(model).items() if not k.startswith('_') }
        
        with open(f'{path}/hyperparams.json', "w") as file:
            json.dump(model_params|params, file)
            
        with open(f'{path}/log_dict.json', "w") as file:
            json.dump(log_dict, file)
        
        torch.save(state_dict, f'{path}/state_dict.pth')
    
    return log_dict