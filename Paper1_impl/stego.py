
# coding: utf-8

# In[5]:


import os
import numpy as np
import torch
from torchvision import transforms, datasets
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[6]:


np.random.seed(4)


# In[7]:


def loadData(folderDir): 
    dataTransform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((512,512)),
        transforms.ToTensor()
    ])
    
    dataSet = datasets.ImageFolder(folderDir,dataTransform)
    dataLoader = torch.utils.data.DataLoader(dataSet, batch_size=4,shuffle=True, num_workers=2)

#     datasetsizes = {x: len(dataSet[x]) for x in ['train', 'val']}
    return dataSet,dataLoader


# In[8]:


folderDir = os.path.join("./bossbase_toy_dataset/train/")
trainData,trainDataLoader = loadData(folderDir)
# print(trainDataLoader.classes)

folderDir = os.path.join("./bossbase_toy_dataset/valid/")
validData,validDataLoader = loadData(folderDir)

folderDir = os.path.join("./bossbase_toy_dataset/test/")
testData,testDataLoader = loadData(folderDir)


# In[9]:


class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        
        #Group 1
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,bias=False,kernel_size=5,stride=1,padding=2)
        self.abs1 = torch.abs
        self.bn1 = nn.BatchNorm2d(num_features=8,momentum=0.9)
        self.tanh1 = nn.Tanh()
        self.avgPool1 = nn.AvgPool2d(kernel_size=5,stride=2,padding=2)
        
        #Group 2
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=5,stride=1,padding=2)
        self.bn2 = nn.BatchNorm2d(num_features=16,momentum=0.9)
        self.tanh2 = nn.Tanh()
        self.avgPool2 = nn.AvgPool2d(kernel_size=5,stride=2,padding=2)
        
        #Group3
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.bn3 = nn.BatchNorm2d(num_features=32,momentum=0.9)
        self.relu3 = nn.ReLU()
        self.avgPool3 = nn.AvgPool2d(kernel_size=5,stride=2,padding=2)
        
        #Group4
        self.conv4 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2)
        self.bn4 = nn.BatchNorm2d(num_features=64,momentum=0.9)
        self.relu4 = nn.ReLU()
        self.avgPool4 = nn.AvgPool2d(kernel_size=5,stride=2,padding=2)
        
        #Group5
        self.conv5 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5,stride=1,padding=2)
        self.bn5 = nn.BatchNorm2d(num_features=128,momentum=0.9)
        self.relu5 = nn.ReLU()
        self.avgPool5 = nn.AvgPool2d(kernel_size=32)
        
        self.flatten = torch.flatten
        self.fc = nn.Linear(in_features=128,out_features=2)
        self.softmax = nn.Softmax()
    
    def forward(self,x):
        #Group1
        out1 = self.conv1(x)
        out1 = self.abs1(out1)
        out1 = self.bn1(out1)
        out1 = self.tanh1(out1)
        out1 = self.avgPool1(out1)
        #Group2
        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        out1 = self.tanh2(out1)
        out1 = self.avgPool2(out1)
        
        #Group3
        out1 = self.conv3(out1)
        out1 = self.bn3(out1)
        out1 = self.relu3(out1)
        out1 = self.avgPool3(out1)
            
        #Group4
        out1 = self.conv4(out1)
        out1 = self.bn4(out1)
        out1 = self.relu4(out1)
        out1 = self.avgPool4(out1)
            
        
        #Group5
        out1 = self.conv5(out1)
        out1 = self.bn5(out1)
        out1 = self.relu5(out1)
        out1 = self.avgPool5(out1)
        
        out1 = self.flatten(out1,start_dim=1)
        out1 = self.fc(out1)
        out1 = self.softmax(out1)
        return out1
        


# In[10]:


model = Model()


# In[25]:


learningRate = 0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)


# In[26]:


iterations = 0
numEpocs = 50


# In[39]:


for epoch in range(numEpocs):
    for i,(images,labels) in enumerate(trainDataLoader):
        images = Variable(images/255.0)
        labels = Variable(labels)
#         print(labels)
        y_onehot = labels.numpy()
        y_onehot = (np.arange(2) == y_onehot[:,None]).astype(np.float32)
        y_onehot = torch.from_numpy(y_onehot)
#         print(y_onehot)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, y_onehot)
        loss.backward()
        optimizer.step()
        iterations += 1
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, numEpocs, i + 1, len(trainDataLoader), loss.item(),
                          (correct / total) * 100))
        

