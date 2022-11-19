import torch
import pickle
from model import WideResNet
from autoattack import AutoAttack
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, models

class ImageDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        data, label = pickle.load(open(file,'rb'))
        self.data = torch.Tensor(data).permute(0,3,1,2).contiguous().div(255)
        self.label = label
        self.transform = transforms.Compose([
                transforms.RandomCrop(size = self.data.shape[-1], padding = 2),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                     std=(0.247, 0.243, 0.261))]
            )

    def __getitem__(self, idx):
        return self.transform(self.data[idx]), self.label[idx]

    def __len__(self):
        return len(self.label)

device = 1
batch_size = 512
model = models.__dict__['resnet50']().to(device)
ds = ImageDataset('./data/cifar_train.pt')
dl = DataLoader(ds,batch_size=batch_size,shuffle=True,pin_memory=True)

model.eval()
while True:
    for x,y in dl:
        while True:
            try:
                model.load_state_dict(torch.load('./save/model.pt'))
                break
            except: continue
        adversary = AutoAttack(model, norm='Linf', eps=0.031, attacks_to_run=['apgd-ce','apgd-t'], version='custom', device=device)
        x = x.to(device)
        y = y.to(device)
        x_prime, label = adversary.run_standard_evaluation(x,y,bs=batch_size,return_labels=True)
        x, x_prime, label = x.cpu(), x_prime.cpu(), label.cpu()
        pert = x_prime - x
        torch.save((x, pert, label),'./save/cache.pt')