#%%
import torch
from torch.utils.data import Dataset, DataLoader

class SDataset(Dataset):
    """S shape dataset."""

    def __init__(self, size, transform=None):

        pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
        
        self.transform = transform
        self.data = torch.zeros((size, 2))
        
        for i in range(size):
            theta = torch.FloatTensor(1).uniform_(0, 2*pi)
            r = torch.rand((1))
            x = (10+r)*torch.cos(theta)

            if .5*pi <= theta <= 1.5*pi:
                y = (10+r)*torch.sin(theta) + 10
            else:
                y = (10+r)*torch.sin(theta) - 10
        
            point = torch.cat((x, y), 0)
            self.data[i] = point

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data = SDataset(500)
    print(len(data.data))
    plt.scatter(data.data[2,0], data.data[2,1])
    # plt.scatter(data.data[:,0], data.data[:,1])

# %%
