import torch
import torch.optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from torchvision import transforms
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from network import incremental_resnet34
from dataset import *
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
from PIL import Image
import os
from test import *
import time


device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu") 
model_name = f'checkpoint/model_state_dict.pt'  
net = torch.load(model_name) 

transform_train= transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

#parameters
Episode = 1
lr= 1e-4
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
batchsize=64
old_class_num = 3
new_class_num = 1
total_class_num = old_class_num + new_class_num
old_class_list = list(range(0,old_class_num))

# add classifier 
net.add_classifier(new_class_num)
net =net.to(device)

# dataset construction
val_dataframe= pd.read_csv(f'dataframe/val.csv')
val_dataset = CustomDataset(val_dataframe, transform = transform_train)
val_dataloader = DataLoader(dataset = val_dataset, batch_size=batchsize,shuffle=True)

train_dataframe= pd.read_csv(f'dataframe/train.csv')
train_dataset = CustomDataset(train_dataframe, transform = transform_train)
train_dataloader = DataLoader(dataset = train_dataset, batch_size=batchsize,shuffle=True)      

# train
for epoch in range(Episode):
    net.train()
    for i,data in enumerate(train_dataloader): 
        _,img,label = data
        img,label = img.to(device), label.to(device) # image,label을 cuda로 전송
        output = net(img)
        loss = criterion(output,label) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
    save_path = 'checkpoint/' 
    torch.save(net, save_path + f'model_state_dict.pt')



