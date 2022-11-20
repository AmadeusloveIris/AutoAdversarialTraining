import torch
from torch import nn
import pickle
from models.resnet import ResNet50
from models.wideresnet import WideResNet
from autoattack import AutoAttack
from Prefetcher import DataPrefetcher
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class ImageDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        data, label = pickle.load(open(file,'rb'))
        self.data = torch.Tensor(data).permute(0,3,1,2).contiguous().div(255)
        self.label = label
        self.transform = transforms.Compose([
                transforms.RandomCrop(size = 32, padding = 2),
                transforms.RandomHorizontalFlip()]
            )

    def __getitem__(self, idx):
        return self.transform(self.data[idx]), self.label[idx]
        
    def __len__(self):
        return len(self.label)

device = 1
batch_size = 128
ds_train = ImageDataset('./data/cifar_train.pt')
ds_eval = ImageDataset('./data/cifar_test.pt')

#model = ResNet50().to(device)
model = WideResNet().to(device)

dl_train = DataLoader(ds_train,batch_size=batch_size,shuffle=True,pin_memory=True, drop_last=True, num_workers=4)
dl_eval = DataLoader(ds_eval,batch_size=batch_size,pin_memory=True, num_workers=4)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=0.0001)

m = 8
epoch = 90
epsilon = 8
pgd_step = epsilon/m
epsilon /= 255
pgd_step /=255
update_freq = 200

loss_accu = 0
accuracy_accu = 0

i = 1
pert = torch.zeros((batch_size,3,32,32)).to(device)
for i_epoch in range(epoch):
    for x,y in dl_train:
        for _ in range(m):
            model.train()
            x = x.to(device)
            x.requires_grad_()
            y = y.to(device)
            optimizer.zero_grad()
            x_prime = x + pert
            x_prime = torch.clamp(x_prime, 0, 1)
            pred = model(x_prime)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            pert = x.grad.detach().sign()*pgd_step + pert.detach()
            pert = pert.clamp(-epsilon, epsilon)
            loss_accu += loss.item()
            accuracy_accu += (pred.argmax(-1)==y).float().mean().item()
        if i%5==0: torch.save(model.state_dict(),'./save/model_free.pt')
        if i%update_freq==0:
            model.eval()
            eval_accuracy = 0
            with torch.no_grad():
                for x_eval, y_eval in dl_eval:
                    x_eval = x_eval.to(device)
                    y_eval = y_eval.to(device)
                    pred_eval = model(x_eval)
                    eval_accuracy += (pred_eval.argmax(-1)==y_eval).sum()
            print(f'idx:{i} loss:{loss_accu/(update_freq*m)} train accuracy:{accuracy_accu/(update_freq*m)} eval accuracy:{eval_accuracy.item()/len(ds_eval)}')
            loss_accu = 0
            accuracy_accu = 0
            torch.save(model.state_dict(),'./save/model_free.pt')
        i+=1
    if i_epoch==30:
        for param_group in optimizer.param_groups: param_group['lr'] /= 10
    if i_epoch==60:
        for param_group in optimizer.param_groups: param_group['lr'] /= 10
