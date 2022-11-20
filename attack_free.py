import os
import torch
import pickle
from models.resnet import ResNet50
from models.wideresnet import WideResNet
from autoattack import AutoAttack
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
class ImageDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        data, label = pickle.load(open(file,'rb'))
        self.data = torch.Tensor(data).permute(0,3,1,2).contiguous().div(255)
        self.label = torch.LongTensor(label)
        self.transform = transforms.Compose([
                transforms.RandomCrop(size = 32, padding = 2),
                transforms.RandomHorizontalFlip()]
            )

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def initialize_ds(self):
        self.x = self.transform(self.data)
        idx = torch.randperm(len(self.label))
        self.x = self.x[idx]
        self.y = self.label[idx]

    def __len__(self):
        return len(self.label)

device = 3
batch_size = 128
episilon = 8
#model = ResNet50().to(device)
model = WideResNet().to(device)
ds = ImageDataset('./data/cifar_train.pt')
dl = DataLoader(ds,batch_size=batch_size,shuffle=False,pin_memory=True)

model.eval()
while True:
    pert_all = []
    ds.initialize_ds()
    if os.path.exists('./end'): continue
    if os.path.exists('./save/cache.pt'): continue
    for x,y in dl:
        if os.path.exists('./end'): break
        while True:
            try:
                model.load_state_dict(torch.load('./save/model_free_auto.pt'))
                break
            except: continue
        adversary = AutoAttack(model, norm='Linf', eps=episilon/255, attacks_to_run=['apgd-ce','apgd-t'], version='custom', device=device)
        x_prime = adversary.run_standard_evaluation(x,y,bs=batch_size)
        pert = x_prime - x
        idx = (pert==0).reshape(batch_size,-1).all(-1)
        pert[idx] = torch.zeros(idx.sum(),3,32,32)
        pert_all.append(pert)
    pert_all = torch.concat(pert_all)
    pert_all = torch.concat([pert_all,torch.zeros(len(ds)-pert_all.shape[0],3,32,32)])
    torch.save((ds.x, pert_all, ds.y),'./save/cache.pt')
