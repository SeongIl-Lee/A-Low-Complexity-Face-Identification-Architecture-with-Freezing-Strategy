import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import time


class incremental_resnet34(nn.Module):
    def __init__(self,make_model = True):
        super().__init__()
        if make_model:
            self.make_model()
        
    def make_model(self):
        resnet = models.resnet34(weights='IMAGENET1K_V1')
     
        self.feature_extractor = nn.Sequential()
        self.classifiers = nn.ModuleList()
        for name,layer in resnet.named_children():
            if name != 'fc':
                self.feature_extractor.add_module(name,layer)
            
    def add_classifier(self,num_class):
        classifier = nn.Linear(512,num_class)
        self.classifiers.append(classifier)
    
    def clear_classifier(self):
        self.classifiers = nn.ModuleList()
        
    def forward(self,x):
        x = self.feature_extractor(x)
        x = torch.flatten(x,1)
        x = torch.cat([classifier(x) for classifier in self.classifiers],dim=1)
        return x