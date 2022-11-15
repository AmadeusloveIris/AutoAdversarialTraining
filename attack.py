import torch
import pickle
from model import WideResNet
from autoattack import AutoAttack
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        data, label = pickle.load(open(file,'rb'))
        self.data = self.transform(torch.Tensor(data).permute(0,3,1,2))
        self.label = label

    def transform(self, data):
        transform_func = transforms.Compose([
                transforms.RandomCrop(size = (3,3), padding = 2),
                transforms.RandomHorizontalFlip(0.5)]
            )
        augmented_data = transform_func(data)
        return augmented_data

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.label)

device = 1
batch_size = 64
model = WideResNet(32,10).to(device)
ds = ImageDataset('./data/cifar_train.pt')
dl = DataLoader(ds,batch_size=batch_size,shuffle=True,pin_memory=True)
model.eval()
for x,y in dl:
    try: model.load_state_dict(torch.load('./save/model.pt'))
    except: pass
    adversary = AutoAttack(model, norm='Linf', eps=2, version='standard',device=device)
    x = x.to(device)
    y = y.to(device)
    x_prime, label = adversary.run_standard_evaluation(x,y,bs=batch_size,return_labels=True)
    x, x_prime, label = x.cpu(), x_prime.cpu(), label.cpu()
    pert = x_prime - x
