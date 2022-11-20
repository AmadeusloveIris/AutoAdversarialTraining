import torch
import pickle
import numpy as np
from model import WideResNet
from autoattack import AutoAttack
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, models
class ImageDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        data, label = pickle.load(open(file,'rb'))
        self.data = self.transform(data)
        self.label = list(np.stack([label for i in range(25)]).T.reshape(-1))*2
    
    def transform(self, data):
        data = data.transpose(0,3,1,2)
        
        data = np.pad(data, [(0,0),(0,0),(2,2),(2,2)])
        h, w = data.shape[-2:]
        th, tw = 32, 32
        temp = []
        for i in range(0, h-th+1):
            for j in range(0, w-tw+1):
                temp.append(data[..., i:i+th,j:j+tw])
        data = np.concatenate(temp)
        
        data = np.concatenate([data,data[...,::-1]])

        return data

    def __getitem__(self, idx):
        return idx, torch.Tensor(self.data[idx]).div(255), self.label[idx]

    def __len__(self):
        return len(self.label)

device = 1
batch_size = 128
episilon = 8
model = models.__dict__['resnet50']().to(device)
ds = ImageDataset('./data/cifar_train.pt')
dl = DataLoader(ds,batch_size=batch_size,shuffle=True,pin_memory=True)

model.eval()
while True:
    for i,x,y in dl:
        while True:
            try:
                model.load_state_dict(torch.load('./save/model.pt'))
                break
            except: continue
        adversary = AutoAttack(model, norm='Linf', eps=episilon/255, attacks_to_run=['apgd-ce'], version='custom', device=device)
        x_prime = adversary.run_standard_evaluation(x,y,bs=batch_size)
        pert = x_prime - x
        idx = (pert==0).reshape(pert.shape[0],-1).all(-1)
        pert[idx] = 0.001*torch.randn((idx.sum(),3,32,32))
        torch.save((i, pert),'./save/cache.pt')
