import argparse
import torch
from dataloaders import get_dataloaders
from model import EfficientNet
from tqdm import tqdm
from torchvision import datasets
import torch.nn as nn
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for Training')

    # Add arguments
    parser.add_argument('--model_weights', type=str, required=True, default='data/', help='Path to model weights')
    parser.add_argument('--data_dir', type=str, required=True, default='data/', help='Path to the directory containing the data')
    
    # Parse arguments
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_arguments()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_dataloader, test_data_size = \
        get_dataloaders(args.data_dir, 32, 32)
    print("Dataloader loaded successfully")

    test_loss = 0.0
    class_correct = list(0. for i in range(200))
    class_total = list(0. for i in range(200))

    # Load model and weights
    model_ft = EfficientNet(device, 200)
    model_ft.load_state_dict(torch.load(args.model_weights))
    model_ft.eval()

    # Datasets and loss function
    classes = datasets.ImageFolder(args.data_dir).classes
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

    # Loop through the dataset
    for inputs, labels in tqdm(test_dataloader):
        if torch.cuda.is_available(): 
            inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            output = model_ft(inputs)
            loss = criterion(output, labels)
        test_loss += loss.item()*inputs.size(0)
        _, pred = torch.max(output, 1)    
        correct_tensor = pred.eq(labels.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())
        
        # Extract number of correct labels and total labels
        for i in range(len(labels)):
            label = labels.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    test_loss = test_loss/len(test_dataloader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    # Print results
    for i in range(200):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))