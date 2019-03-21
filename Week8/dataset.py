
# coding: utf-8

# In[2]:


import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
#import joblib

class Caltech101(Dataset):
    
    def __init__(self,inpath,transform=None):
        self.transform = transform
        self.path = inpath
        self.label_list = os.listdir(self.path)
        self.image_list = []
        for label in self.label_list:
            each_label = os.listdir(os.path.join(inpath,label))
            self.image_list.append([str(label)+i for i in each_label]) 
        self.image_list = [item for sublist in self.image_list for item in sublist]
        num_classes = len(self.label_list)
        #labels = torch.arange(num_classes)
        #labels = labels.reshape(num_classes,1)
        #self.one_hot_target = (labels == torch.arange(num_classes).reshape(1, num_classes)).float()
                                

    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self,idx):
        #print(self.image_list[idx])
        label = self.image_list[idx].split('image')[0]
        idx_ = 'image'+self.image_list[idx].split('image')[1]
        image_path = os.path.join(self.path,label)
        image = Image.open(os.path.join(image_path,idx_))
        image = image.convert('RGB')
        
        
        if self.transform:
            image = self.transform(image)
        #image = extra_transforms(image)
        image = image.reshape((3,256,256))
        target = self.label_list.index(label)
        target = torch.FloatTensor([target])
        
        return (image,target)
        
        
        

