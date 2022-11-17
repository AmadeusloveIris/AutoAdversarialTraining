import torch
import pickle
import torch.nn as nn
from model import WideResNet
from autoattack import AutoAttack
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, models

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
batch_size = 2048
#model = WideResNet(32,10).to(device)
model = models.__dict__['resnet50']().to(device)
ds = ImageDataset('./data/cifar_train.pt')
dl = DataLoader(ds,batch_size=batch_size,shuffle=True,pin_memory=True)
criterion = nn.CrossEntropyLoss()

model.eval()
while True:
    for x,y in dl:
        while True:
            try:
                model.load_state_dict(torch.load('./save/model.pt'))
                break
            except: continue
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad(): 
            pred = model(x)
            loss = criterion(pred, y)
        print(((pred.argmax(-1)==y).sum()/len(y)).item(), loss.item())
