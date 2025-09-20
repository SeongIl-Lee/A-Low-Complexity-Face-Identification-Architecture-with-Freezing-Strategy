import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from network import incremental_resnet34
from dataset import *
from test import *
import random

def herding_selection(img_arrays, m):
    mean_img = np.mean(img_arrays, axis=0)
    min_idx_list = []  
    comb = np.zeros_like(mean_img)  
    for num in range(m): 
        dist = np.zeros(img_arrays.shape[0]) 
        for i in range(img_arrays.shape[0]):
            p = (num + 1) * mean_img - comb 
            if i in min_idx_list:
                dist[i] = np.inf 
            else:
                dist[i] = np.round(np.linalg.norm(p - img_arrays[i]), decimals=3)  
        min_dist_idx = np.argmin(dist)
        min_dist = img_arrays[min_dist_idx]  
        comb += min_dist 
        min_idx_list.append(min_dist_idx) 
    return min_idx_list



def update_memory(df, net, iter, C_total, C_old,R,K,representative_memory_new_path,representative_memory_old_path=None): 
    transform= transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")  
    df_repre_new = pd.DataFrame(columns=['idx', 'path', 'class_name', 'class_id', 'cluster_idx']) 
    M = R // ( C_total * K ) 
    A = ( R // C_total ) - ( M * K ) 
    num_perclass = R // C_total 
    j = 0 
    if  R - (num_perclass * C_total ) > 0:
        random_class = random.sample(C_total, R - (num_perclass * C_total ) )
    else:
        random_class = []
        
    # Remove process
    if iter > 1: 
        
        repre_old_df = pd.read_csv(representative_memory_old_path)
        old_class_id = repre_old_df['class_id'].unique()
        for c in old_class_id:
            repre_old_perclass = repre_old_df.loc[repre_old_df['class_id']==c,:] 
            if c in random_class:
                random_cluster = random.sample(range(0, K), A+1)
            else:
                random_cluster = random.sample(range(0, K), A)
            for k in range(K): 
                repre_old_perclass_percluster = repre_old_perclass.loc[repre_old_perclass['cluster_idx']==k,:] 
                if k in random_cluster: 
                    remain_sample = repre_old_perclass_percluster[:M+1]
                else:
                    remain_sample = repre_old_perclass_percluster[:M]
                df_repre_new = pd.concat([df_repre_new, remain_sample], ignore_index=True)
    
    # Add process
    for c in range(C_old,C_total):
        df_perclass = df.loc[df['class_id']==c,:]
        dataset_perclass = CustomDataset(df_perclass,transform=transform)
        train_dataloader = DataLoader(dataset = dataset_perclass, batch_size=1,shuffle=False)
        for i,data in enumerate(train_dataloader):
            _,img,label = data
            img,label = img.to(device), label.to(device)
         
            feature = net.feature_extractor(img)
            feature_numpy = feature.cpu().detach().numpy()
            feature_numpy = feature_numpy.reshape(1,-1,1) 
            if i == 0:
                    features_numpy_array = feature_numpy  
            else:
                features_numpy_array = np.concatenate((features_numpy_array, feature_numpy), axis=0)
        
        criteria = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10,1.0 )
        flags = cv2.KMEANS_RANDOM_CENTERS  
        ret,labels,center = cv2.kmeans(features_numpy_array, K, None, criteria, 10, flags) 

        if c in random_class:
            random_cluster = random.sample(range(0, K), A+1)
        else:
            random_cluster = random.sample(range(0, K), A)
            
        for k in range(K): 
            index_per_cluster = np.where(labels == k)[0] 
            clusters = features_numpy_array[index_per_cluster]

            if k in random_cluster: 
                idx_list = herding_selection(clusters,M + 1)
            else:
                idx_list = herding_selection(clusters,M)
            
            index_per_cluster_herd = index_per_cluster[idx_list] 
            index_per_cluster_herd += j*600
            
            filtered_df = df.loc[df['idx'].isin(index_per_cluster_herd),:] 
            filtered_df = filtered_df.set_index('idx').reindex(index_per_cluster_herd).reset_index()
            
            filtered_df['cluster_idx'] = k 
            
            df_repre_new = pd.concat([df_repre_new, filtered_df], ignore_index=True)
        j += 1
    df_repre_new['idx'] = range(len(df_repre_new))
    df_repre_new = df_repre_new.drop(columns=['Unnamed: 0'])
    df_repre_new.to_csv(representative_memory_new_path, index=False)  
    

