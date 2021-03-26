#%%
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt

class A2CDataset(Dataset):
    """
    AIT 2 CelebA - People (Face Recognition)
    The Labeled Faces in the Wild face recognition dataset.
    https://www.kaggle.com/atulanandjha/lfwpeople
    """

    def __init__(self, root, except_folder=None, transform=None):
        
        self.transform = transform
        self.file_path = []
        subfolder = os.listdir(root)
        
        for folder in subfolder:
            if folder not in set(except_folder):
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
    DATA_FOLDER = './ait2celeb'
    dataset = A2CDataset(DATA_FOLDER, except_folder=['trainB', 'trainA'])
    print(len(dataset))

    fig = plt.figure()

    for i in range(len(dataset)):
        sample = dataset[i]

        print(i, sample)

        # ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.imshow(sample)

        if i == 3:
            plt.show()
            break

# %%
