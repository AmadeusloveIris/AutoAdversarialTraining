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
        # return self.transform(self.data[idx]), self.label[idx]
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.label)