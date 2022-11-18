import torch
import pickle
from model import WideResNet
from autoattack import AutoAttack
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

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

device = 3
batch_size = 512
model = models.__dict__['resnet50']().to(device)
model.load_state_dict(torch.load('./save/model_test.pt'))
ds = ImageDataset('./data/cifar_train.pt')
dl = DataLoader(ds,batch_size=batch_size,pin_memory=True)

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=0.031,
                  num_steps=20,
                  step_size=0.007):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    #print('err pgd (white-box): ', 1-err_pgd/X.shape[0])
    return err_pgd.item()

pgd_right_accu = 0
natural_right_accu = 0

model.eval()
for x,y in dl:
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad(): pred = model(x)
    natural_right_accu += (pred.argmax(-1)==y).float().sum()
    pgd_right_accu += batch_size-_pgd_whitebox(model, x, y)
print(f'Natural accuracy: {natural_right_accu/len(ds)}, Robust accuracy: {pgd_right_accu/len(ds)}')
