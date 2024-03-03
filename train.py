import torch
from torch import nn, optim
from torch.nn import functional as F
from dataloaders import get_dataloaders
from model import EfficientNet
import argparse
from tqdm import tqdm
import os
import numpy as np
import logging
import time  # Import the time module

import warnings
warnings.filterwarnings("ignore")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for Training')

    # Add arguments
    parser.add_argument('--data_dir', type=str, required=True, default='CUB_200_2011/images', help='Path to the directory containing the training data')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    
    # Parse arguments
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # Get arguments from command line
    args = parse_arguments()

    # Create the output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'weights'), exist_ok=True)

    # Set up logging
    logging.basicConfig(filename=os.path.join(args.output_dir, 'logs.txt'), level=logging.INFO, format='%(message)s')

    # Start the training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataloader, train_data_size, test_dataloader, test_data_size = \
        get_dataloaders(args.data_dir, args.batch_size, 32)
    print("Dataloaders loaded successfully")

    model = EfficientNet(device, 200)
    print("Model loaded successfully")

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of trainable parameters", params)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.96)

    training_history = {'accuracy': [], 'loss': []}
    test_history = {'accuracy': [], 'loss': []}

    total_training_time = 0
    best_acc = 0.0
    iter_num = 0

    for epoch in range(1, args.num_epochs + 1):
        start_time_epoch = time.time()  # Record start time for the epoch

        logging.info('Starting Epoch {}/{}'.format(epoch, args.num_epochs))
        print('Epoch {}/{}'.format(epoch, args.num_epochs))
        print('-' * 10)

        # Start with the training phase
        model.train()

        train_loss = 0
        train_acc = 0

        # Loop through dataset
        for inputs, labels in tqdm(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Step of optimization
            optimizer.zero_grad()
            logits = model(inputs)
            _, predicted = torch.max(logits, 1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Store training logs
            train_loss += loss.item() * inputs.size(0)
            train_acc += torch.sum(predicted == labels.data)
            logging.info('Iteration %d : loss : %f' % (iter_num, loss.item()))
            iter_num += 1

        # LR Step after each epoch
        step_scheduler.step()

        # Calculate and store time taken for the epoch
        end_time_epoch = time.time()
        epoch_time = end_time_epoch - start_time_epoch
        print(f'Time taken for Epoch {epoch}: {epoch_time:.2f} seconds')
        logging.info(f'Time taken for Epoch {epoch}: {epoch_time:.2f} seconds')

        total_training_time += epoch_time

        # Store logs
        training_history['accuracy'].append(train_acc / train_data_size)
        training_history['loss'].append(train_loss / train_data_size)
        print('Train Accuracy: {:.4f} Train Loss: {:.4f}'.format(train_acc / train_data_size, train_loss / train_data_size))

        weights_path = os.path.join(args.output_dir, 'weights', f'epoch_{epoch}.pth')
        torch.save(model.state_dict(), weights_path)
        print(f'Model weights saved to {weights_path}')

        logging.info(f'Finished Epoch {epoch}/{args.num_epochs}, \
                      Train Accuracy: {train_acc / train_data_size}, \
                      Train Loss: {train_loss / train_data_size}')
        
        logging.info(50 * '-')

    # Calculate and print total time taken for training
    print(f'Total time taken for training: {total_training_time:.2f} seconds')
    logging.info(f'Total time taken for training: {total_training_time:.2f} seconds')
