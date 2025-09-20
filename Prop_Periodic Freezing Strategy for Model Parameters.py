import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
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



columns = ['number_of_classes','minibatch_sum','acc','training_time']
df = pd.DataFrame(columns=columns)
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")  
transform_train= transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
end_point = False
model_name = f'checkpoint/model_state_dict.pt'
net = torch.load(model_name) 
old_num_class = [0,1,2]
total_num = 4

end_point = False
total_num_class = list(range(0,total_num))
new_num_class = list(set(total_num_class) - set(old_num_class))


# add classifier 
net.add_classifier(len(new_num_class)) 
net =net.to(device) 

# hyperparameters
Episode = 1
batchsize = 64
lr = 1e-4
optimizer= torch.optim.Adam(net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
Freeze = False
alternation_period = 5
step_index = 0
# Configure the dataloader

# 1) val dataloader
val_dataframe= pd.read_csv(f'dataframe/val.csv')
val_dataset = CustomDataset(val_dataframe, transform = transform_train)
val_dataloader = DataLoader(dataset = val_dataset, batch_size=batchsize,shuffle=True)

# 2)train dataloader (representative memory + samples for new class)
train_dataframe_newclass= pd.read_csv(f'dataframe/train.csv')
train_dataframe_repre= pd.read_csv(f'dataframe/repre.csv') 

train_dataframe_concat = pd.concat([train_dataframe_repre,train_dataframe_newclass], ignore_index=True)
train_dataframe_concat['idx'] = range(len(train_dataframe_concat))

train_dataset = CustomDataset(train_dataframe_concat, transform = transform_train) 
train_dataloader = DataLoader(dataset = train_dataset, batch_size=batchsize, shuffle=True)



# 3) train
for epoch in range(Episode):
    if end_point == True:
        break
    for i,data in enumerate(train_dataloader): 
        net =net.to(device) 
        _,img,label= data
        img,label = img.to(device), label.to(device)
        
        # alternation of freezing/unfreezing 
        if ( (step_index-1)// alternation_period< step_index // alternation_period) and (step_index != 0) :
            if Freeze == True:
                for param in net.parameters():
                    param.requires_grad = True
                Freeze = False
            else:
                for param in net.parameters():
                    param.requires_grad = False
        
                for param in net.classifiers[-1].parameters():
                    param.requires_grad = True
                Freeze = True
                
        # training       
        if Freeze == False:
            output = net(img)
            loss = criterion(output,label) 
        else:                                # phase2(홀수)
            true_label_idx = torch.where(label == new_num_class[0])
            false_label_idx = torch.where(label != new_num_class[0])
            label[true_label_idx] = 1.0
            label[false_label_idx] = 0.0
            label = label.type(torch.float32)
            label = label.reshape(-1,1)

            feature = net.feature_extractor(img)
            feature_flatten = torch.flatten(feature,1)
            output = net.classifiers[-1](feature_flatten)
            loss = F.binary_cross_entropy_with_logits(output,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        step_index += 1
save_path = 'checkpoint' 
torch.save(net, save_path + f'model_state_dict.pt') 
old_num_class = total_num_class



