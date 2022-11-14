import torch
import torch.nn as nn
from Sampler import PERSampler
from Prefetcher import DataPrefetcher
from ReplayBuffer import ReplayBuffer
from torch.utils.data import DataLoader
from model import WideResNet

device = 2
model = WideResNet(32,10).to(device)
ds = ReplayBuffer('./data/cifar_train.pt')
sampler = PERSampler(len(ds), 16)
dl = DataLoader(ds, batch_sampler=sampler, pin_memory=True)
dl = DataPrefetcher(dl, device)
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=0.0001)

epsilon = 4.0
pgd_step = 4.0
for i, (idx, (x, pert, y)) in enumerate(dl,start=1):
    x_prime = x + pert
    x_prime.requires_grad_()
    optimizer.zero_grad()
    pred = model(x_prime)
    loss = criterion(pred, y)
    loss_update = loss.detach().cpu()
    loss.mean().backward()
    optimizer.step()
    pert = x_prime.grad.detach().sign()*pgd_step + pert.detach()
    pert = pert.clamp_(-epsilon, epsilon).cpu()
    ds.buffer_update(idx, pert, loss_update)
    if i%20==0: torch.save(model.state_dict(),'./save/model.pt')
