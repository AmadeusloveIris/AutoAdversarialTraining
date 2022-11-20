import torch
import pickle
import torch.nn as nn
import numpy as np
from Sampler import PERSampler
from Prefetcher import DataPrefetcher
from ReplayBuffer import ReplayBuffer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, models

class ImageDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        data, label = pickle.load(open(file,'rb'))
        self.data = torch.Tensor(data).permute(0,3,1,2).contiguous().div(255)
        self.label = label

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.label)

device = 3
batch_size = 128
model = models.__dict__['resnet50']().to(device)
torch.save(model.state_dict(),'./save/model.pt')
ds = ReplayBuffer('./data/cifar_train.pt')
#sampler = PERSampler(len(ds), batch_size)
dl = DataLoader(ds, batch_size=batch_size, pin_memory=True)
dl = DataPrefetcher(dl, device)
criterion_cross = nn.CrossEntropyLoss(reduction='none')
criterion_kl = nn.KLDivLoss(reduction='none')
optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=0.0002)

ds_eval = ImageDataset('./data/cifar_test.pt')
dl_eval = DataLoader(ds_eval,batch_size=batch_size,pin_memory=True)

#priority_weight = torch.Tensor(1/(sampler.buffer_size*sampler.p)).to(device)
update_freq = 200
beta = 1
loss_total = 0
right_num = 0
epsilon = 8
pgd_step = 1
epsilon /= 255
pgd_step /= 255
while True:
    for i, (idx, (x, pert, y)) in enumerate(dl,start=1):
        model.train()
        x_prime = x + pert
        x_prime.requires_grad_()
        optimizer.zero_grad()
        pred_ori = model(x)
        pred_adv = model(x_prime)
        loss_kl = criterion_kl(pred_adv.log_softmax(-1),pred_ori.detach().softmax(-1)).mean()
        loss_kl.backward(retain_graph=True)
        pert = x_prime.grad.detach().sign()*pgd_step + pert.detach()
        pert = pert.clamp(-epsilon, epsilon).cpu()
        optimizer.zero_grad()
        #loss = (criterion_cross(pred_ori, y)+beta*loss_kl)*priority_weight[idx]
        loss = criterion_cross(pred_ori, y)+beta*loss_kl
        loss_update = loss.detach().cpu()
        loss.mean().backward()
        optimizer.step()
        ds.buffer_update(idx, pert, loss_update)
        loss_total += loss_update.sum().item()
        right_num += (pred_ori.detach().argmax(-1)==y).sum().item()
        if i%5==0: torch.save(model.state_dict(),'./save/model.pt')
        if i%update_freq==0:
            print(f'steps: {i}  accuracy: {right_num/(update_freq*batch_size)}  loss: {loss_total/(update_freq*batch_size)}')
            loss_total = 0
            right_num = 0
            eval_accuracy = 0
            model.eval()
            with torch.no_grad():
                for x_eval, y_eval in dl_eval:
                    x_eval = x_eval.to(device)
                    y_eval = y_eval.to(device)
                    pred_eval = model(x_eval)
                    eval_accuracy += (pred_eval.argmax(-1)==y_eval).sum()
            print(f'accuracy:{eval_accuracy/len(ds_eval)}')
