import numpy as np 
import pandas as pd
import os
#import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler

from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset

from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings(action='ignore')


train_df = pd.read_csv('../datasets/heart_failure_clinical_records_dataset.csv')

train_df.info()

print(train_df.describe())

'''
fig, ax = plt.subplots()
ax.boxplot([train_df.platelets,train_df.creatinine_phosphokinase])
plt.xticks([1,2],['platelets','creatinine_phosphokinase'])
plt.legend()
plt.show()
'''

fix_features = pd.concat([train_df.platelets],axis=1)
scaler = MinMaxScaler(feature_range=(0,1))
fix_features = pd.DataFrame(scaler.fit_transform(fix_features))
df = pd.concat([train_df,fix_features],axis=1)
df.drop(['platelets'],axis=1,inplace=True)
df = df.rename(columns={0:'creatinine_phosphokinase',1:'platelets'})

print(df)

print(df.DEATH_EVENT.value_counts(normalize=True)*100)


# 准备数据
train, test = df[:-40], df[-40:]
print(train.shape,test.shape)

y_train = df.DEATH_EVENT
x_train = df.drop('DEATH_EVENT',axis=1)
print(x_train.shape,y_train.shape)

x_train, x_test, y_train, y_test = train_test_split(x_train,y_train,test_size=0.3,shuffle=True)

smote = SMOTE(random_state=0)
x_train_oversample , y_train_oversample = smote.fit_resample(x_train.values,y_train.values) 

print(x_train_oversample.shape , y_train_oversample.shape)
print(x_test.shape,y_test.shape)


# 数据集
class CustomDataset(Dataset):
    def __init__(self,data,label):
        self.data = data
        self.label = label
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        data = torch.tensor(self.data[idx],dtype=torch.float32)
        label = torch.tensor(self.label[idx],dtype=torch.float32)
        return data,label

train_set = CustomDataset(x_train_oversample,y_train_oversample)
test_set = CustomDataset(x_test.values,y_test.values)

test_loader = DataLoader(test_set,batch_size=len(y_test),num_workers=0)

print(y_test.value_counts())

# 模型
class GRU(nn.Module):
    def __init__(self):
        super(GRU,self).__init__()
        self.gru1 = nn.Linear(12,100)
        self.bn1 = nn.BatchNorm1d(100)
        self.gru2 = nn.Linear(100,50)
        self.bn2 = nn.BatchNorm1d(50)
        self.fc1 = nn.Linear(50,10)
        self.bn3 = nn.BatchNorm1d(10)
        self.fc2 = nn.Linear(10,2)
        self.relu = nn.ReLU()
        self.swish = nn.Hardswish()
        self.flatten = nn.Flatten()
    def forward(self,x):
        x = self.bn1(self.gru1(x))
        x = self.swish(x)
        x = self.bn2(self.gru2(x))
        x = self.swish(x)
        x = self.swish(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x

model = GRU()


n_epochs = 100
train_min_loss = 0
train_acc = torch.zeros(n_epochs)
train_loss = torch.zeros(n_epochs)
valid_acc = torch.zeros(n_epochs)
valid_loss = torch.zeros(n_epochs)
kf = KFold(n_splits=7,shuffle=True)
for fold , (train_idx,valid_idx) in enumerate(kf.split(train_set)):
    print(f'Fold:{fold+1}')
    train_sampler_kfold = SubsetRandomSampler(train_idx)
    valid_sampler_kfold = SubsetRandomSampler(valid_idx)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=16,
                                               num_workers=0,sampler=train_sampler_kfold,drop_last=True)
    valid_loader = torch.utils.data.DataLoader(train_set,batch_size=16,
                                num_workers=0,sampler=valid_sampler_kfold,drop_last=True)
    valid_max_acc = 0
    train_acc = torch.zeros(n_epochs)
    train_loss = torch.zeros(n_epochs)
    valid_acc = torch.zeros(n_epochs)
    valid_loss = torch.zeros(n_epochs)
    model = GRU()
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0003)
    criterion = torch.nn.CrossEntropyLoss()
    for e in range(0, n_epochs):
        model.train()
        for data, labels in tqdm(train_loader):
            data, labels = data.to(device).float(), labels.to(device).long()

            optimizer.zero_grad()
            logits = model(data)
            logits = logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss[e] += loss.item()

            softmax = logits.softmax(dim=1)
            argmax = softmax.argmax(1)
            equals = argmax == labels.reshape(argmax.shape)

            train_acc[e] += torch.mean(equals.type(torch.float)).detach().cpu()


        train_loss[e] /= len(train_loader)
        train_acc[e] /= len(train_loader)
        model.eval()
        with torch.no_grad():
            for data , labels in tqdm(valid_loader):
                data, labels = data.to(device).float(), labels.to(device).long()
                
                logits = model(data)
                loss = criterion(logits,labels)
                
                valid_loss[e] += loss.item()
                
                softmax = logits.softmax(dim=1)
                argmax = softmax.argmax(1)
                equals = argmax == labels.reshape(argmax.shape)
                
                valid_acc[e] += torch.mean(equals.type(torch.float)).detach().cpu()
                
            valid_loss[e] /= len(valid_loader)
            valid_acc[e] /= len(valid_loader)
        print('Epoch: {} \tTraining acc: {:.6f} \tTraining Loss: {:.6f}'.format(
            e+1, train_acc[e], train_loss[e]))
        
        print('Epoch: {} \tValidation acc: {:.6f} \tValidation Loss: {:.6f}'.format(
            e+1, valid_acc[e], valid_loss[e]))
        
        if valid_max_acc <= valid_acc[e]:
            torch.save(model,f'Linear_model_{fold}.pt')
            print('model save complete...')
            valid_max_acc = valid_acc[e]
            patience = 0
        else:
            patience += 1
            print(f'Patience :{patience}')
            if patience > 15:
                patience += 1
                print('We meet early Stopping so End Training...')
                print(f'Best Validation Accuracy : {valid_max_acc}')
                break