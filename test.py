import torch
from torch import nn
import pickle
from model import WideResNet
from autoattack import AutoAttack
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

class ImageDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        data, label = pickle.load(open(file,'rb'))
        self.data = data
        self.label = label
        self.transform_func = transforms.Compose([
                transforms.RandomCrop(size = self.data.shape[-2], padding = 2),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                     std=(0.247, 0.243, 0.261))]
            )

    def transform(self, data):
        data = torch.Tensor(data).permute(2,0,1).contiguous().div(255)
        augmented_data = self.transform_func(data)
        return augmented_data

    def __getitem__(self, idx):
        return self.transform(self.data[idx]), self.label[idx]

    def __len__(self):
        return len(self.label)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1024
# model = WideResNet(32,10).to(device)
model = models.__dict__['resnet50']().to(device)
ds = ImageDataset('./data/cifar_train.pt')

#model = models.__dict__['resnet50']().to(device)
if os.path.exists('./save/model.pt'):
    model.load_state_dict(torch.load('./save/model.pt'))
transform_func = transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomCrop(size = 32, padding = 2),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                          std=(0.247, 0.243, 0.261))])

#ds = datasets.CIFAR10('data',train=True,transform=transform_func)
dl = DataLoader(ds,batch_size=batch_size,shuffle=True,pin_memory=True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.001)

model.eval()
while True:
    for i, (x,y) in enumerate(dl,start=1):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        # print(pred.shape)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        print(i, ((pred.argmax(-1)==y).sum()/len(y)).item(), loss.item())
        if i%10==0: torch.save(model.state_dict(),'./save/model.pt')
        