import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import os
from copy import copy
from copy import deepcopy
import torch.nn.functional as F
from PIL import Image
import numpy as np
from numpy import asarray

from resnet import ResNet
from residualsebasicblock import ResidualSEBasicBlock
from basicblock import BasicBlock
from bottleneckblock import BottleneckBlock
from chiormufloader import ChiorMufdataset
from train_model import train_model

# Set device to GPU or CPU

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Allow augmentation transform for training set, no augementation for val/test set

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

full_dataset = ChiorMufdataset(transform=preprocess)

def KFlodloader(dataset, n_splits=8):
    val_n = len(dataset)/n_splits
    k_index = []

    for i in range(n_splits):
        indices = torch.randperm(len(dataset)).tolist()
        idx_val = 1
        while val_n ==2 and dataset[indices[0]][1] == dataset[indices[idx_val]][1]:
            idx_val += 1
        train = deepcopy(dataset)
        val = []
        val.append(train.pop(idx_val))
        val.append(train.pop(0))
        
        train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True, num_workers=4)
        val_dataloader = torch.utils.data.DataLoader(val, batch_size=2, shuffle=True, num_workers=4)
        yield {'train':train_dataloader, 'val':val_dataloader}, i

def train_Kflod_model(dataset, n_splits=8, num_epochs=25, weights_name='weight_save', is_inception=False):
    '''
    train_model: train a model on a dataset
    
            Parameters:
                    model: Pytorch model
                    dataloaders: dataset
                    criterion: loss function
                    optimizer: update weights function
                    num_epochs: number of epochs
                    weights_name: file name to save weights
                    is_inception: The model is inception net (Google LeNet) or not

            Returns:
                    model: Best model from evaluation result
                    val_acc_history: evaluation accuracy history
                    loss_acc_history: loss value history
    '''
    since = time.time()

    kflod_acc = []

    print('-' * 10)
    # KFlod loop
    for flodloader, i in KFlodloader(dataset, n_splits):
        
        epoch_start = time.time()
        
        print('Flod number: {}/{}'.format(i+1, n_splits))

        print('RESTART MODEL')
        ## load model
        model_this = ResSENet18().to(device)
        model_this.load_state_dict(torch.load('Lab03/ressenet18_bestsofar.pth'))
        model_this.linear = nn.Linear(512, 2).to(device)
        # print(model_this.eval())

        # Optimizer and loss function
        criterion_this = nn.CrossEntropyLoss()
        params_to_update = model_this.parameters()
        # Now we'll use Adam optimization
        optimizer_this = optim.Adam(params_to_update, lr=0.01)
        print('RESTART MODEL FINISHED')

        val_acc_history = []
        loss_acc_history = []
        best_acc = 0.0

        for epoch in range(num_epochs):
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model_this.train()  # Set model to training mode
                else:
                    model_this.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                # Iterate over data.
                for inputs, labels in flodloader[phase]:
                    # for process anything, device and dataset must put in the same place.
                    # If the model is in GPU, input and output must set to GPU
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    # it uses for update training weights
                    optimizer_this.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model_this(inputs)
                            # print('outputs', outputs)
                            loss1 = criterion_this(outputs, labels)
                            loss2 = criterion_this(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = model_this(inputs)
                            loss = criterion_this(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer_this.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(flodloader[phase].dataset)
                epoch_acc = running_corrects.double() / len(flodloader[phase].dataset)
                epoch_end = time.time()
                
                elapsed_epoch = epoch_end - epoch_start

                if epoch % 100 == 0:
                    print('...')
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                    print("Epoch time taken: ", elapsed_epoch)
                
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                if phase == 'val':
                    val_acc_history.append(epoch_acc)
                if phase == 'train':
                    loss_acc_history.append(epoch_loss)

        kflod_acc.append(best_acc.cpu().detach().numpy())
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        print('-' * 5)
        
        del model_this
        del criterion_this
        del optimizer_this
        
        if i+1 == 8:
            return kflod_acc, _, _

def ResNet18(num_classes = 10):
    '''
    First conv layer: 1
    4 residual blocks with two sets of two convolutions each: 2*2 + 2*2 + 2*2 + 2*2 = 16 conv layers
    last FC layer: 1
    Total layers: 1+16+1 = 18
    '''
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def ResSENet18(num_classes = 10):
    return ResNet(ResidualSEBasicBlock, [2, 2, 2, 2], num_classes)

def KFlod_CV(dataset):
    avg_acc, val_acc_history, loss_acc_history = train_Kflod_model(full_dataset, 8, 25, 'chiormuf_bestsofar')
    return avg_acc, val_acc_history, loss_acc_history

# avg_acc, val_acc_history, loss_acc_history = KFlod_CV(full_dataset)

# train model
ressenet = ResSENet18().to(device)
ressenet.load_state_dict(torch.load('Lab03/ressenet18_bestsofar.pth'))
ressenet.linear = nn.Linear(512, 2).to(device)
criterion = nn.CrossEntropyLoss()
params_to_update = ressenet.parameters()
optimizer = optim.Adam(params_to_update, lr=0.01)

if True:
    for loaddict, i in KFlodloader(full_dataset, n_splits=8):
        
        best_model, val_acc_history, loss_acc_history = train_model(ressenet, loaddict, criterion, optimizer, 5, 'chiormuf_bestsofar')
else:
    best_model = ResSENet18(2).to(device)
    best_model.load_state_dict(torch.load('Lab03/chiormuf_bestsofar.pth'))
# test
image1 = Image.open('Lab03/test/Screen Shot 2021-02-04 at 17.36.26.png').convert('RGB')
image1 = torch.tensor([asarray(preprocess(image1))]).to(device)
label1 = 1

image2 = Image.open('Lab03/test/Screen Shot 2021-02-04 at 17.37.44.png').convert('RGB')
image2 = torch.tensor([asarray(preprocess(image2))]).to(device)
label2 = 0

output1 = best_model(image1)
output2 = best_model(image2)

output1 = np.around(*output1.cpu().detach().numpy())
output2 = np.around(*output2.cpu().detach().numpy())

print('Muffin(1) is...', output1)
print('Chiwawa(0) is...', output2)