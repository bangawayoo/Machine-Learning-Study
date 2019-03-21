
# coding: utf-8

# In[3]:


from tensorboardlogger import Logger
from loader import data_loader
import torch


# In[4]:


train_loader, test_loader, val_loader = data_loader()


# In[5]:


import torch 
from models import resnet18

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = resnet18().cuda()


# In[6]:


import torch
from torch import nn
from tqdm import tqdm

#Pytorch Tensorboard 호환용 
#logger = Logger('./logs') 

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  

total_epoch = 100

for epoch in range(total_epoch):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        labels = labels.squeeze(1)
        inputs, labels = inputs.cuda(), labels.to(device, dtype=torch.int64)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, argmax = torch.max(outputs, 1)
        accuracy = (labels == argmax.squeeze()).float().mean()

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 1000 mini-batches
            correct = 0
            total = 0
            with torch.no_grad():
                valid_running_loss = 0 
                for data in val_loader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    labels = labels.to(device, dtype=torch.int64)
                    labels = labels.squeeze(1)
                    outputs = model(images)

                    valid_loss = criterion(outputs, labels)
                    valid_loss.cuda()
                    valid_running_loss += valid_loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('[%d, %5d] loss: %.3f ' %
                  (epoch + 1, i + 1, running_loss / 1000))
            print('        Validation Set, loss = %.3f, Acc.: %.2f' %(valid_running_loss/len(val_loader), correct/total))
            running_loss = 0.0
            valid_running_loss = 0


