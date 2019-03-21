
# coding: utf-8

# In[1]:


import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as data

def data_loader():
    #Parameters
    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225] )
        ])
    BATCH_SIZE = 4
    VALID_SPLIT = 0.15
    TEST_SPLIT = 0.15
    SHUFFLE_ = True
    SEED= 1230
    NUM_WORKERS=4
	
    # Data loader
	
    from dataset import Caltech101
    caltech = Caltech101('data',transform=TRANSFORM_IMG)

    import numpy as np
    from torch.utils.data.sampler import SubsetRandomSampler

    dataset_size = len(caltech)
    indices = list(range(dataset_size))
    split_1 = int(np.floor(VALID_SPLIT * dataset_size))
    split_2 = int(np.floor (TEST_SPLIT * dataset_size))

    if SHUFFLE_:
            np.random.seed(SEED)
            np.random.shuffle(indices)
    val_idx, test_idx, train_idx = indices[:split_1], indices[split_1:split_1+split_2],indices[split_2:] 

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(caltech, batch_size=BATCH_SIZE,
                             sampler = train_sampler, num_workers = NUM_WORKERS)
    test_loader = DataLoader(caltech, batch_size=BATCH_SIZE,
                            sampler = test_sampler, num_workers = NUM_WORKERS)
    val_loader = DataLoader(caltech, batch_size=BATCH_SIZE,
                           sampler = val_sampler, num_workers = NUM_WORKERS)
    return train_loader, test_loader, val_loader

