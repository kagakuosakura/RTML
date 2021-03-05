#%%
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt

class LFWDataset(Dataset):
    """
    LFW - People (Face Recognition)
    The Labeled Faces in the Wild face recognition dataset.
    https://www.kaggle.com/atulanandjha/lfwpeople
    """

    def __init__(self,root , transform=None):
        
        self.transform = transform
        self.file_path = []
        subfolder = os.listdir(root)
        
        for folder in subfolder:
            try :
                filenames = os.listdir(os.path.join(root, folder))
                for filename in filenames:
                    self.file_path.append(os.path.join(root, folder, filename))
            except :
                pass

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = Image.open(self.file_path[idx])

        if self.transform:
            sample = self.transform(sample)

        return sample

#%%
if __name__ == "__main__":
    DATA_FOLDER = './archive/lfw_funneled'
    dataset = LFWDataset(DATA_FOLDER)

    fig = plt.figure()

    for i in range(len(dataset)):
        sample = dataset[i]

        print(i, sample)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.imshow(sample)

        if i == 3:
            plt.show()
            break
# %%

# %%
