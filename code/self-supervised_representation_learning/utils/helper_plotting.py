import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import umap
import torch
import pandas as pd
import numpy as np
from scipy import spatial


def plot_latent_space(data: torch.Tensor, metadata: pd.DataFrame, metadata_df_key:str, model, device: torch.device,
                       plot_input=False, real_dataset=None, real_metadata_df_key=None, real_metadata_df_category=None):
    
    # RETRIEVE METADATA AND MAP TO COLORS
    color_dict = {}
    for i, key in enumerate(np.unique(metadata.loc[:,metadata_df_key])):
        color_dict[key] = i
    mapped = metadata.loc[:,metadata_df_key].map(color_dict)

    if real_dataset is not None:
        masks = real_dataset.get('cell_masks')

        real_metadata_df = pd.DataFrame({'gene':real_dataset.gene, 'cell_type':real_dataset.cell_type,
                                          'cell_id':[str(cell_id.values).split('_')[0] for cell_id in real_dataset.cell_id]})
        if real_metadata_df_key is not None:
            #PRINT ONLY DATA MATCHING CATEGORY
            if real_metadata_df_category is not None:
                real_selected_indices = np.where(real_metadata_df[real_metadata_df_key]==real_metadata_df_category)[0]
                real_dataset = real_dataset.isel(dict(index = real_selected_indices))
                real_metadata_df = real_metadata_df.iloc[real_selected_indices].reset_index()
                color_dict[f'real data - {real_metadata_df_category}'] = i+1
                mapped = pd.concat((mapped,pd.Series([i+1]*len(real_dataset.trans_coords))))
                alphas = np.concatenate(([0.2]*data.size(0),[0.9]*len(real_dataset.trans_coords)))

            else:
                real_color_dict = {}
                for j,category in enumerate(np.unique(real_metadata_df[real_metadata_df_key])):
                    real_color_dict[category] = i+j+1
                mapped.iloc[:] = 12
                mapped = pd.concat((mapped, real_metadata_df.loc[:,real_metadata_df_key].map(real_color_dict)))
                color_dict = {'simulated data':12}
                color_dict.update(real_color_dict)
                alphas = np.concatenate(([0.01]*data.size(0),[0.9]*len(real_dataset.trans_coords)))

        else:
            color_dict['real data'] = i+1
            mapped = pd.concat((mapped,pd.Series([i+1]*len(real_dataset.trans_coords))))
            alphas = np.concatenate(([0.2]*data.size(0),[1]*len(real_dataset.trans_coords)))
    
    else:
        alphas =[0.6]*data.size(0)

    #COMPUTE EMBEDDINGS
    data_loader = torch.utils.data.DataLoader(dataset=data,
                                            batch_size=256,
                                            num_workers=2,
                                            shuffle=False,
                                            drop_last=False)
    embeddings = torch.Tensor()
    model.eval()
    with torch.no_grad():
        for features in data_loader:
            encoded = model.encoding_fn(features.to(device))
            embeddings = torch.cat((embeddings, encoded.detach().cpu()))

        if real_dataset is not None:
            real_data = torch.Tensor(real_dataset.trans_coords.values)
            real_data_loader = torch.utils.data.DataLoader(dataset=real_data,
                                                batch_size=256,
                                                num_workers=2,
                                                shuffle=False,
                                                drop_last=False)
            for features in real_data_loader:
                encoded = model.encoding_fn(features.to(device))
                embeddings = torch.cat((embeddings, encoded.detach().cpu()))

    #UMAP TRANSFORM
    trans = umap.UMAP(n_neighbors=15, min_dist=0.8).fit(embeddings)
    # Make a tree of all the umap coordinates to make it easy to find the closest one to the coordinates you click
    tree = spatial.KDTree(trans.embedding_)

    if plot_input:
        fig,(ax1,ax2) = plt.subplots(1,2, figsize=(10,3))
        fig.subplots_adjust(wspace=1.1)
        fig.tight_layout()
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        
        ax1.set_title(model._get_name()+f' embeddings (k={model.k})')
        ax1.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s= 1, c=mapped[:], cmap='Spectral', alpha=alphas)
        
        handles = [mpatches.Patch(color=cm.Spectral(color), label=key)
                for key, color in zip(color_dict.keys(), plt.Normalize()([*color_dict.values()]))]
        ax1.legend(handles=handles, title=metadata_df_key, bbox_to_anchor=(1.1, 1))
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')

        def onclick(event):
            ax2.clear()
            pos = (event.xdata,event.ydata)
            # Find the closest point to the clicked location, and use it to index the input
            index = tree.query([pos])[1][0]
            
            ax1.plot(trans.embedding_[index,0], trans.embedding_[index,1], 'ko', fillstyle='none',markersize=8)
            if len(ax1.lines)>1:
                ax1.lines[-2].remove()

            # Plot original data
            if index > data.size(0):
                index = index - data.size(0)
                original_features = real_data

                if real_metadata_df is not None:
                    cell_id = int(real_metadata_df.loc[index,'cell_id'])
                    original_metadata = f"{real_metadata_df.loc[index,'gene']}"
                    mask_coords = np.where(masks == cell_id)
                    min_row = np.min(mask_coords[0])
                    max_row = np.max(mask_coords[0])
                    min_col = np.min(mask_coords[1])
                    max_col = np.max(mask_coords[1])
                    mask = masks[min_row:max_row,min_col:max_col].copy()
                    ax2.imshow(mask==cell_id, cmap='gray')
                else:
                    original_metadata = 'real data'
            else:
                original_features = data
                original_metadata = f'simulated_{metadata.loc[index,metadata_df_key]}'
            
            ax2.scatter(original_features[index, 1], original_features[index, 0], s=3)
            ax2.set_title(original_metadata)

        fig.canvas.mpl_connect('button_press_event', onclick)
    
    else:
        plt.figure(figsize=(10, 6))
        plt.subplots_adjust(right=0.8)      
        plt.title(model._get_name()+f' embeddings (k={model.k})')
        plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s= 2, c=mapped[:], cmap='Spectral', alpha=alphas)
        handles = [mpatches.Patch(color=cm.Spectral(color), label=key)
                for key, color in zip(color_dict.keys(), plt.Normalize()([*color_dict.values()]))]
        plt.legend(handles=handles, title=metadata_df_key, bbox_to_anchor=(1, 0.9), borderaxespad=0.)
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')

    plt.show()


def plot_latent_space_v2(embeddings: torch.Tensor, metadata: pd.DataFrame, metadata_df_key:str, model_name, device: torch.device):

    embeddings = embeddings
    
    trans = umap.UMAP(n_neighbors=15, min_dist=0.8, metric='euclidean').fit(embeddings)
    # Make a tree of all the umap coordinates to make it easy to find the closest one to the coordinates you click
    tree = spatial.KDTree(trans.embedding_)

    color_dict = {}
    for i, key in enumerate(np.unique(metadata.loc[:,metadata_df_key])):
        color_dict[key] = i
    mapped = metadata.loc[:,metadata_df_key].map(color_dict)

    fig,ax1 = plt.subplots(1,1, figsize=(8,6))
    fig.subplots_adjust(wspace=1.2)
    ax1.set_aspect('equal')
    
    ax1.set_title(model_name+f' embeddings (k={embeddings.shape[1]})')
    ax1.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s= 0.5, c=mapped[:], cmap='Spectral', alpha=0.4)
    handles = [mpatches.Patch(color=cm.Spectral(color), label=key)
            for key, color in zip(color_dict.keys(), plt.Normalize()([*color_dict.values()]))]
    ax1.legend(handles=handles, title=metadata_df_key, bbox_to_anchor=(1.25, 0.8))
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    plt.show()