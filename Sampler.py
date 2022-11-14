import numpy as np
from torch.utils.data import Sampler

class PERSampler(Sampler):
    def __init__(self, buffer_size, train_batch_size, gamma=0.6):
        self.buffer_size = buffer_size
        self.train_batch_size = train_batch_size
        self.p = (1/np.arange(1,buffer_size+1))**gamma
        self.p = self.p/self.p.sum()

    def __iter__(self):
        return self
    
    def __next__(self):
        persample_idx = np.random.choice(self.buffer_size, size=self.train_batch_size, p=self.p, replace=False)
        return persample_idx
