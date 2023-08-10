import tifffile
import xarray
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join


def imgs_to_xarray(simdata_dir: str, output_file: str) -> None:

    file_names = [f for f in listdir(simdata_dir) if isfile(join(simdata_dir, f))]

    images, desc = [],[]

    for file in file_names:
        images.append(tifffile.imread(join(simdata_dir, file)))
        desc.append(file.replace('.tif',''))


    patterns = ['_'.join(metadata.split('_')[:2]) if metadata.split('_')[1]=='edge' else metadata.split('_')[0] for metadata in desc]
    
    dataset = xarray.Dataset(data_vars = dict(images=(['image_index','row','col'],images)),
                            coords = dict(metadata = (['image_index'], desc),
                                          pattern = (['image_index'], patterns)))

    dataset.to_netcdf(output_file,engine='netcdf4')

def coords_to_xarray(simdata_dir: str, output_file: str, masked=False, pad_mode='repeat', num_points=200, normalize=False) -> None:
    
    file_names = [f for f in listdir(simdata_dir) if isfile(join(simdata_dir, f))]
    trans_coords_list, desc_list = [],[]

    for file in file_names:
        trans_coords = pd.read_csv(join(simdata_dir, file)).to_numpy().swapaxes(0,1)
        
        if pad_mode=='zeros':
            padded = np.pad(trans_coords, ((0,0),(0,num_points - len(trans_coords[-1]))), 'constant', constant_values=(0,))
        elif pad_mode=='repeat':
            padded = np.array([[coords[i % len(coords)] for i in range(num_points)] for coords in trans_coords])
        else:
            raise ValueError("`pad_mode` argument should be one of ('zeros','repeat')")
        
        if normalize:
            padded = (padded - np.min(padded))/np.ptp(padded)

        if masked:
            mask = np.concatenate((np.ones(trans_coords.shape),
                                   np.zeros((trans_coords.shape[0],num_points - trans_coords.shape[1]))),
                                   axis=1)
            padded = np.concatenate((padded,np.expand_dims(mask[0],0)),axis=0)

        trans_coords_list.append(padded)
        
        desc = file.replace('.csv','')
        desc_list.append(desc)
        
    trans_coords_array = np.array(trans_coords_list)
    
    patterns = ['_'.join(metadata.split('_')[:2]) if metadata.split('_')[1]=='edge' else metadata.split('_')[0] for metadata in desc_list]
        
    if masked:
        dataset = xarray.Dataset(data_vars = dict(trans_coords=(['index','coords','spots'], trans_coords_array)),
                                coords = dict(metadata = (['index'], desc_list),
                                              pattern = (['index'], patterns),
                                              coords = (['row','col','mask'])))
    else:
        dataset = xarray.Dataset(data_vars = dict(trans_coords=(['index','coords','spots'], trans_coords_array)),
                                coords = dict(metadata = (['index'], desc_list),
                                              pattern = (['index'], patterns),
                                              coords = (['row','col'])))
        
    dataset.to_netcdf(output_file,engine='netcdf4')