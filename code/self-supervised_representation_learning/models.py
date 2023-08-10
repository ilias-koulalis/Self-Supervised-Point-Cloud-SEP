import torch
from torch import nn
from utils.helper_utils import ChamferDistance, DensityAwareChamferDistance,\
                                init_weights
from utils.helper_modules import STN2d
import numpy as np
import math
from geomloss import SamplesLoss
import gc


#Adapted from https://github.com/TropComplique/point-cloud-autoencoder/    
class PCAE(nn.Module):

    def __init__(self, k, num_points, loss_fn = 'cd-t'):
        """
        Arguments:
            k: an integer, dimension of the representation vector.
            num_points: an integer.
        """
        super().__init__()
        
        self.num_points = num_points
        self.loss_fn = loss_fn
        self.k = k

        # ENCODER
        pointwise_layers = []
        num_units = [2, 64, 128, k]

        for n, m in zip(num_units[:-1], num_units[1:]):
            pointwise_layers.extend([
                nn.Conv1d(n, m, kernel_size=1, bias=False),
                nn.BatchNorm1d(m),
                nn.ReLU(),
            ])
        self.pointwise_layers = nn.Sequential(*pointwise_layers)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        # DECODER
        self.decoder = nn.Sequential(
            nn.Conv1d(k, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, num_points * 2, kernel_size=1),
            nn.ReLU(),
        )

        #weight init
        self.pointwise_layers.apply(init_weights(nn.init.xavier_uniform_))
        self.maxpool.apply(init_weights(nn.init.xavier_uniform_))
        self.decoder.apply(init_weights(nn.init.xavier_uniform_))

    def encoding_fn(self, x):
        x, mask = torch.split(x, [2,1] ,dim=1)
        mask = torch.tile(mask, dims=(1,self.k,1))
        x = self.pointwise_layers(x)
        x = torch.where(mask == 1, x, -torch.inf)
        
        x = self.maxpool(x)
        encoded = x.squeeze()

        return encoded
        
    def forward(self, x):
        b = x.size(0)
        
        x, mask = torch.split(x, [2,1] ,dim=1)
        mask = torch.tile(mask, dims=(1,self.k,1))

        x = self.pointwise_layers(x)
        x = torch.where(mask == 1, x, -torch.inf)


        encoded = self.maxpool(x)
        x = self.decoder(encoded)
        decoded = x.view(b, 2, self.num_points)
        
        return encoded, decoded
    
    def loss(self, decoded, features):        
        
        match self.loss_fn:
            case 'cd-t':
                loss_fn = ChamferDistance(variant='cd-t')
                features, mask = torch.split(features, [2,1] ,dim=1)
                reconstruction_loss = torch.stack([loss_fn(single_decoded.unsqueeze(0),\
                                        single_feature[:,:torch.sum(single_mask, dtype=torch.int16)].unsqueeze(0))\
                                        for single_feature, single_decoded, single_mask in zip(features, decoded, mask)]).mean(0)
            case 'cd-p':
                loss_fn = ChamferDistance(variant='cd-p')
                features, mask = torch.split(features, [2,1] ,dim=1)
                reconstruction_loss = torch.stack([loss_fn(single_decoded.unsqueeze(0),\
                                        single_feature[:,:torch.sum(single_mask, dtype=torch.int16)].unsqueeze(0))\
                                        for single_feature, single_decoded, single_mask in zip(features, decoded, mask)]).mean(0)
            case 'density_aware_cd':
                #slow
                loss_fn = DensityAwareChamferDistance(alpha=1e-1,n_lambda=1)
                features, mask = torch.split(features, [2,1] ,dim=1)
                reconstruction_loss = torch.stack([loss_fn(single_decoded.unsqueeze(0),\
                                        single_feature[:,:torch.sum(single_mask, dtype=torch.int16)].unsqueeze(0))\
                                        for single_feature, single_decoded, single_mask in zip(features, decoded, mask)]).mean(0)

            case 'geom_sinkhorn':
                features, mask = torch.split(features.swapaxes(1,2), [2,1] ,dim=2)
                decoded = decoded.swapaxes(1,2).contiguous()
                features = features.contiguous()
                loss_fn = SamplesLoss(loss="sinkhorn", p=1, blur=0.5, diameter=50, scaling=0.2, debias=False, backend='online')  
                reconstruction_loss = torch.stack([loss_fn(single_decoded[:torch.argwhere(single_mask).size(0)], single_feature[:torch.argwhere(single_mask).size(0)])\
                                                    for single_feature, single_decoded, single_mask in zip(features, decoded, mask)]).mean(0) 
                
            case _:
                raise NameError('loss function undefined')
        
        return reconstruction_loss


##Adapted for pytorch from https://github.com/charlesq34/pointnet-autoencoder/
class PointNetAE(nn.Module):

    def __init__(self, k, num_points, loss_fn = 'cd-t'):
        """
        Arguments:
            k: an integer, dimension of the representation vector.
            num_points: an integer.
        """
        super().__init__()
        
        self.num_points = num_points
        self.loss_fn = loss_fn
        self.k = k

        # ENCODER
        self.stn = STN2d()
        
        self.global_feat = nn.Sequential(
            nn.Conv1d(2, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(1024, k, 1),
            nn.BatchNorm1d(k)
        )

        # DECODER

        self.decoder = nn.Sequential(
            nn.Linear(k, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, num_points*2)
        )

        #weight init
        self.global_feat.apply(init_weights(nn.init.xavier_normal_))
        self.decoder.apply(init_weights(nn.init.xavier_normal_))

    def encoding_fn(self, x):
        x, mask = torch.split(x, [2,1] ,dim=1)
        mask = torch.tile(mask, dims=(1,self.k,1))

        sp_trans = self.stn(x)
        x = x.transpose(2, 1) 
        x = torch.bmm(x, sp_trans)
        x = x.transpose(2, 1) 
        x = self.global_feat(x)

        x = torch.where(mask == 1, x, -torch.inf)
            
        x = torch.max(x, dim = 2, keepdim=True)[0]
        encoded = x.view(-1, self.k)
        
        return encoded
        
    def forward(self, x):
        b = x.size(0)
        n_pts = x.size(2)

        x, mask = torch.split(x, [2,1] ,dim=1)
        mask = torch.tile(mask, dims=(1,self.k,1))

        sp_trans = self.stn(x)
        x = x.transpose(2, 1) 
        x = torch.bmm(x, sp_trans)
        x = x.transpose(2, 1) 
        x = self.global_feat(x)

        x = torch.where(mask == 1, x, -torch.inf)
            
        x = torch.max(x, dim = 2, keepdim=True)[0]
        encoded = x.view(-1, self.k)

        x = self.decoder(encoded)
        decoded = x.view(b, 2, self.num_points)

        return encoded, decoded
    
    def loss(self, decoded, features):        
        
        match self.loss_fn:
            case 'cd-t':
                loss_fn = ChamferDistance(variant='cd-t')
                features, mask = torch.split(features, [2,1] ,dim=1)
                reconstruction_loss = torch.stack([loss_fn(single_decoded.unsqueeze(0),\
                                        single_feature[:,:torch.sum(single_mask, dtype=torch.int16)].unsqueeze(0))\
                                        for single_feature, single_decoded, single_mask in zip(features, decoded, mask)]).mean(0)
            case 'cd-p':
                loss_fn = ChamferDistance(variant='cd-p')
                features, mask = torch.split(features, [2,1] ,dim=1)
                reconstruction_loss = torch.stack([loss_fn(single_decoded.unsqueeze(0),\
                                        single_feature[:,:torch.sum(single_mask, dtype=torch.int16)].unsqueeze(0))\
                                        for single_feature, single_decoded, single_mask in zip(features, decoded, mask)]).mean(0)
            case 'density_aware_cd':
                #slow
                loss_fn = DensityAwareChamferDistance(alpha=1e-1,n_lambda=1)
                features, mask = torch.split(features, [2,1] ,dim=1)
                reconstruction_loss = torch.stack([loss_fn(single_decoded.unsqueeze(0),\
                                        single_feature[:,:torch.sum(single_mask, dtype=torch.int16)].unsqueeze(0))\
                                        for single_feature, single_decoded, single_mask in zip(features, decoded, mask)]).mean(0)

            case 'geom_sinkhorn':
                features, mask = torch.split(features.swapaxes(1,2), [2,1] ,dim=2)
                decoded = decoded.swapaxes(1,2).contiguous()
                features = features.contiguous()
                loss_fn = SamplesLoss(loss="sinkhorn", p=1, blur=0.5, diameter=50, scaling=0.2, debias=False, backend='online')  
                reconstruction_loss = torch.stack([loss_fn(single_decoded[:torch.argwhere(single_mask).size(0)], single_feature[:torch.argwhere(single_mask).size(0)])\
                                                    for single_feature, single_decoded, single_mask in zip(features, decoded, mask)]).mean(0)

            case _:
                raise NameError('loss function undefined')
        
        return reconstruction_loss