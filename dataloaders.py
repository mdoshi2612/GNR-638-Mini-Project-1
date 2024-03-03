import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def get_dataloaders(data_dir, train_batch_size, test_batch_size):

    # Define train and test transforms
    # Training transforms contain various augmentation methods
    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
    test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Create dataset with training transform
    all_data = datasets.ImageFolder(data_dir, transform = train_transform)
    train_data_len = int(len(all_data)*0.7)
    test_data_len = int(len(all_data) - train_data_len)
    
    # Get training data split and train dataloader
    train_data, _ = random_split(all_data, [train_data_len, test_data_len])
    train_data_loader = DataLoader(train_data, batch_size = train_batch_size, shuffle = True, num_workers = 4)

    # Testing and validation dataloaders
    all_data = datasets.ImageFolder(data_dir, transform = test_transform)
    _, test_data = random_split(all_data, [train_data_len, test_data_len])
    test_data_loader = DataLoader(test_data, batch_size = test_batch_size, shuffle = True, num_workers = 4)

    return (train_data_loader, train_data_len, test_data_loader, test_data_len)
