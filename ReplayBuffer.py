import os
import torch
import pickle
from autoattack import AutoAttack
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ReplayBuffer(Dataset):
    def __init__(self, file):
        self.init_buffer(file)
        
    def init_buffer(self, file):
        self.cache_time = 0
        data, label = pickle.load(open(file,'rb'))
        data = torch.Tensor(data).permute(0,3,1,2)
        data = self.transform(data)
        self.x = data
        self.pert = torch.zeros(data.shape)
        self.label = torch.LongTensor(label)
        self.loss = torch.ones(len(label))*float('inf')
        self.buffer_size = len(label)
    
    def transform(self, data):
        transform_func = transforms.Compose([
                transforms.RandomCrop(size = (3,3), padding = 2),
                transforms.RandomHorizontalFlip(0.5)]
            )
        augmented_data = transform_func(data)
        return augmented_data
    
    def buffer_update(self, idx, pert, loss):
        self.pert[idx] = pert
        self.loss[idx] = loss
        if self.cache_time<os.path.getmtime('./save/cache.pt'):
            self.cache_time = os.path.getmtime('./save/cache.pt')
            try:
                x, pert, label = torch.load('./save/cache.pt')
                self.x = torch.concat([x, x, self.x])
                self.pert = torch.concat([torch.zeros(pert.shape), pert, self.pert])
                self.label = torch.concat([label,label,self.label])
                self.loss = torch.concat([torch.ones(2*len(label))*float('inf'),self.loss])
            except:
                pass

        idx = torch.argsort(self.loss)
        self.x = self.x[idx][:self.buffer_size]
        self.pert = self.pert[idx][:self.buffer_size]
        self.label = self.label[idx][:self.buffer_size]
        self.loss = self.loss[idx][:self.buffer_size]
        
    def __getitem__(self, idx):
        return idx, self.x[idx], self.pert[idx] ,self.label[idx]
    
    def __len__(self):
        return self.buffer_size
