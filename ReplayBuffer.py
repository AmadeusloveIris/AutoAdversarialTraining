import os
import torch
import pickle
from autoattack import AutoAttack
from torch.utils.data import Dataset
import torch.nn.functional as f

class ReplayBuffer(Dataset):
    def __init__(self, file):
        self.init_buffer(file)
        
    def init_buffer(self, file):
        self.cache_time = 0
        data, label = pickle.load(open(file,'rb'))
        self.x = torch.Tensor(data).permute(0,3,1,2).contiguous().div(255)
        #self.pert = torch.zeros(self.x.shape)
        self.pert = 0.001*torch.randn((len(label),3,32,32))
        self.label = torch.LongTensor(label)
        self.loss = torch.ones(len(label))*float('inf')
        self.buffer_size = len(label)

    def transform(self, data):
        data = torch.Tensor(data).permute(0,3,1,2).contiguous().div(255)
        
        data = f.pad(data, [[2,2],[2,2]])
        h, w = data.shape[-2:]
        th, tw = 32, 32
        temp = []
        for i in range(0, h-th+1):
            for j in range(0, w-tw+1):
                temp.append(data[..., i:i+th,j:j+tw])
        data = torch.concat(temp)
        
        data = torch.concat([data,torch.flip(data,[-1])])

        return data
    
    def buffer_update(self, idx, pert, loss):
        self.pert[idx] = pert
        self.loss[idx] = loss
        if self.cache_time<os.path.getmtime('./save/cache.pt'):
            self.cache_time = os.path.getmtime('./save/cache.pt')
            try:
                x, pert, label = torch.load('./save/cache.pt')
                self.x = torch.concat([x, self.x])
                self.pert = torch.concat([pert, self.pert])
                self.label = torch.concat([label,self.label])
                self.loss = torch.concat([torch.ones(len(label))*float('inf'),self.loss])
            except:
                pass

        idx = torch.argsort(self.loss, descending=True)
        self.x = self.x[idx][:self.buffer_size]
        self.pert = self.pert[idx][:self.buffer_size]
        self.label = self.label[idx][:self.buffer_size]
        self.loss = self.loss[idx][:self.buffer_size]
        print((self.loss==float('inf')).float().mean().item())

    def __getitem__(self, idx):
        return idx, self.x[idx], self.pert[idx] ,self.label[idx]
    
    def __len__(self):
        return self.buffer_size
