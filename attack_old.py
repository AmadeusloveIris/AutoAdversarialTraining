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
        self.label = label
        self.transform = transforms.Compose([
                transforms.RandomCrop(size = 32, padding = 2),
                transforms.RandomHorizontalFlip()]
            )

    def __getitem__(self, idx):
        return self.transform(self.data[idx]), self.label[idx]

    def __len__(self):
        return len(self.label)

device = 0
batch_size = 128
episilon = 8
#model = ResNet50().to(device)
model = WideResNet().to(device)
ds = ImageDataset('./data/cifar_train.pt')
dl = DataLoader(ds,batch_size=batch_size,shuffle=True,pin_memory=True)

model.eval()
while True:
    for x,y in dl:
        while True:
            try:
                model.load_state_dict(torch.load('./save/model1.pt'))
                break
            except: continue
        adversary = AutoAttack(model, norm='Linf', eps=episilon/255, attacks_to_run=['apgd-ce','apgd-t'], version='custom', device=device)
        x_prime, label = adversary.run_standard_evaluation(x,y,bs=batch_size,return_labels=True)
        x, x_prime, label = x, x_prime, label
        pert = x_prime - x
        torch.save((x, pert, label),'./save/cache.pt')
