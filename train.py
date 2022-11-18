import torch
import torch.nn as nn
import numpy as np
from Sampler import PERSampler
from Prefetcher import DataPrefetcher
from ReplayBuffer import ReplayBuffer
from torch.utils.data import DataLoader
from model import WideResNet
from torchvision import datasets, transforms, models

device = 2
batch_size = 4096
model = models.__dict__['resnet50']().to(device)
torch.save(model.state_dict(),'./save/model.pt')
ds = ReplayBuffer('./data/cifar_train.pt')
sampler = PERSampler(len(ds), batch_size)
dl = DataLoader(ds, batch_sampler=sampler, pin_memory=True)
dl = DataPrefetcher(dl, device)
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), 1e-3)

priority_weight = torch.Tensor(1/(sampler.buffer_size*sampler.p)).to(device)
update_freq = 100
loss_total = 0
right_num = 0
epsilon = 0.031
pgd_step = 0.007
while True:
    for i, (idx, (x, pert, y)) in enumerate(dl,start=1):
        x_prime = x + pert
        x_prime.requires_grad_()
        optimizer.zero_grad()
        pred = model(x_prime)
        loss = criterion(pred, y)*priority_weight[idx]
        loss_update = loss.detach().cpu()
        loss.mean().backward()
        optimizer.step()
        pert = x_prime.grad.detach().sign()*pgd_step + pert.detach()
        pert = pert.clamp_(-epsilon, epsilon).cpu()
        ds.buffer_update(idx, pert, loss_update)
        loss_total += loss_update.sum().item()
        right_num += (pred.detach().argmax(-1)==y).sum().item()
        if i%5==0: torch.save(model.state_dict(),'./save/model.pt')
        #print(f'steps: {i}  accuracy: {(pred.detach().argmax(-1)==y).sum().item()/batch_size}  loss: {loss_update.mean().item()}')
        if i%update_freq==0:
            print(f'steps: {i}  accuracy: {right_num/(update_freq*batch_size)}  loss: {loss_total/(update_freq*batch_size)}')
            loss_total = 0
            right_num = 0