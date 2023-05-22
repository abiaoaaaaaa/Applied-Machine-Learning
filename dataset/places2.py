##########################
# Load the training set, verification set, and test set of the data set.
##########################

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

# load places train and valid set
# root: dataset root
# transform: data preprocessor
# train_size_len: train set size
# batch_size: batch size
# shuffle: whether to scramble the data at the beginning of each training cycle
# num_workers: number of child processes used for data loading
def places_train(root = None, 
        transform = None, 
        train_size_len = 0.8, 
        batch_size = 256, 
        shuffle = True, 
        num_workers = 0):
        data = torchvision.datasets.ImageFolder(root, transform=transform)
        train_len = int(len(data) * train_size_len)
        valid_len = len(data) - train_len
        train_data, valid_data = random_split(data, [train_len, valid_len])
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return train_loader, valid_loader

# load places test set
# root: dataset root
# transform: data preprocessor
# batch_size: batch size
# shuffle: whether to scramble the data at the beginning of each training cycle
# num_workers: number of child processes used for data loading
def places_test(root = None, 
        transform = None, 
        batch_size = 256, 
        shuffle = True, 
        num_workers = 0):
        data = torchvision.datasets.ImageFolder(root, transform=transform)
        test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return test_loader



