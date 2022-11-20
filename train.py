import os
import torch
import pickle
import torch.nn as nn
import numpy as np
from models.resnet import ResNet50
from models.wideresnet import WideResNet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, models

class ImageDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        data, label = pickle.load(open(file,'rb'))
        data = torch.Tensor(data).permute(0,3,1,2).contiguous().div(255)
        self.label = torch.LongTensor(label)
        self.transform = transforms.Compose([
                transforms.RandomCrop(size = 32, padding = 2),
                transforms.RandomHorizontalFlip()]
            )
        self.x = self.transform(data)
        self.pert = 0.001*torch.randn(len(label),3,32,32)

    def __getitem__(self, idx):
        return idx, self.x[idx], self.pert[idx], self.label[idx]

    def pert_update(self, idx, pert):
        self.pert[idx] = pert

    def ds_reinitialize(self):
        open('./end','w')
        self.x, self.pert, self.label = torch.load('./save/cache.pt')
        os.remove('./save/cache.pt')
        os.remove('./end')

    def __len__(self):
        return len(self.label)

device = 0
batch_size = 128
model = WideResNet().to(device)
torch.save(model.state_dict(),'./save/model.pt')
ds = ImageDataset('./data/cifar_train.pt')
dl = DataLoader(ds, batch_size=batch_size, pin_memory=True)
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=0.0001)

ds_eval = ImageDataset('./data/cifar_test.pt')
dl_eval = DataLoader(ds_eval,batch_size=batch_size,pin_memory=True)

update_freq = 200
loss_total = 0
right_num = 0
m = 8
epsilon = 8
pgd_step = epsilon/m
epsilon /= 255
pgd_step /= 255
epoch = 90

i=1
for i_epoch in range(epoch):
    for _ in range(m):
        for idx, x, pert, y in dl:
            x = x.to(device)
            pert = pert.to(device)
            y = y.to(device)
            
            x.requires_grad_()
            model.train()
            x_prime = x + pert
            x_prime = torch.clamp(x_prime, 0, 1)
            pred = model(x_prime)
            loss = criterion(pred, y)
            loss_update = loss.detach().cpu()
            loss.mean().backward()
            optimizer.step()
            pert = x.grad.detach().sign()*pgd_step + pert.detach()
            pert = pert.clamp(-epsilon, epsilon).cpu()
            ds.pert_update(idx, pert)
            loss_total += loss_update.sum().item()
            right_num += (pred.detach().argmax(-1)==y).sum().item()
            if i%5==0: torch.save(model.state_dict(),'./save/model_free_auto.pt')
            if i%update_freq==0:
                eval_accuracy = 0
                model.eval()
                with torch.no_grad():
                    for _, x_eval, y_eval in dl_eval:
                        x_eval = x_eval.to(device)
                        y_eval = y_eval.to(device)
                        pred_eval = model(x_eval)
                        eval_accuracy += (pred_eval.argmax(-1)==y_eval).sum()
                print(f'steps:{i} loss:{loss_total/(update_freq*batch_size)} train accuracy:{right_num/(update_freq*batch_size)} eval accuracy:{eval_accuracy/len(ds_eval)}')
                loss_total = 0
                right_num = 0
            i+=1
        ds.ds_reinitialize()
    if i_epoch==30:
        for param_group in optimizer.param_groups: param_group['lr'] /= 10
    if i_epoch==60:
        for param_group in optimizer.param_groups: param_group['lr'] /= 10
