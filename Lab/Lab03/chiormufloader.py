from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import os
from numpy import asarray
import torch
from PIL import Image

class ChiorMufdataset(Dataset):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform
        self.genres = ['chihuahua', 'muffin']
        self.images = []
        self.labels = []
        for genre in self.genres:
            if genre == 'chihuahua':
                exten = '.jpg'
                lebel = 0
            else:
                exten = '.jpeg'
                lebel = 1
            for i in range(1,9):
                file_path = os.path.join('./Lab03/chihuahua-muffin', genre + '-' + str(i) + exten)
                image = Image.open(file_path)
                if self.transform:
                        image = self.transform(image)
                self.images.append(asarray(image))
                self.labels.append(lebel)
            
    def __len__(self):
        return len(self.images)

    def pop(self, idx):
        x = (self.images[idx], self.labels[idx])
        del self.images[idx]
        del self.labels[idx]
        return x

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]