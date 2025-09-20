import torch
from torch.utils.data import Dataset, DataLoader,Subset
from PIL import Image
import numpy as np
from torch.utils.data import Dataset



class CustomDataset(Dataset):
    def __init__(self, dataframe, loss='ce' , transform=None, random_seed=42):
        self.loss = loss
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.df.loc[:, "path"].values[index]
        img = Image.open(img_path)
        label = self.df.loc[:, "class_id"].values[index]
        if self.transform:
            img = self.transform(img)
        return index, img, label



class representative_dataset(Dataset): 
    def __init__(self,tensor_img,tensor_label,loss='ce',transform = None):
        self.loss = loss
        
        self.tensor_img = tensor_img
        self.tensor_label = tensor_label
        self.transform = transform
    def __getitem__(self, index):
        img = self.tensor_img[index,:,:,:]
        img = img.mul(255).byte().permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(img) # pil로 변환
        
        label = self.tensor_label[index,:].item()
        if self.loss == 'bce':
            
            label = 0.0
            label = np.float32(label)
        
        if self.transform:
            img = self.transform(img)
            
        return index+600, img,label
    
    def __len__(self):
        return len(self.tensor_label)





