import torch
from torch import nn
from torch import optim
import os
import numpy as np
from data import load_dataset
from model import CNNImageClassification
from config import Config

def one_epoch(model, dataloader, criterion, epoch, start_batch, optimizer, train):
    """
    Run the model through a single pass through the dataset defined by the dataloader
    model - The model being trained. Inherits torch.nn.Module
    dataloader - Encodes the dataset
    writer - SummaryWriter for Tensorboard
    loss_function - Pytorch loss function, like cross entropy
    epoch - Current epoch number, for printing status
    start_batch - (integer) Where to start the epoch (indexes the dataset)
    optimizer - Pytorch optimizer (like Adam)
    train - If set to True: will train on the dataset, update parameters, and save checkpoints.
            Otherwise, will run model over dataset without training and report loss
            (used for validation)
    """
    if train == True:
      model.train()
    else:
      model.eval()
    
    update = 0
    loss = 0
    running_loss = 0

    for index, data in enumerate(dataloader, 0):

        if train:
            print(f"[Training Epoch] {epoch}/{Config.NUM_EPOCHS - 1}, Batch Number: {index}/{len(dataloader)}")
        else:
            print(f"[Validation Epoch] {epoch}/{Config.NUM_EPOCHS - 1}, Validating: {index}/{len(dataloader)}")

        #get batch of features, labels
        features, labels = data
        features, labels = features.to(Config.DEVICE), labels.to(Config.DEVICE)
        
        output = model(features.float())
        loss = criterion(output, labels.long())
        

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        update += 1
        running_loss += loss.item()

        # update tensorboard and save model
        if update == 100:    # every 10 mini-batches
            running_avg = running_loss / 100
            print(f"[Loss] {running_avg}")

            save(train, epoch, index, model, optimizer, running_avg)
            
            running_loss = 0.0
            update = 0

def save(train, epoch, index, model, optimizer, running_avg):
    if train:
        checkpoint = {
            "epoch":epoch,
            "batch":index,
            "model_state":model.state_dict(),
            "optim_state":optimizer.state_dict()
        }
        
        torch.save(checkpoint, os.path.join(Config.DRIVE_PATH, Config.CHECKPOINT_PATH))      

        if os.path.exists(os.path.join(Config.DRIVE_PATH, 'train_loss.npy')):
            train_loss_values = np.load(os.path.join(Config.DRIVE_PATH, 'train_loss.npy'))
            train_loss_values = np.append(train_loss_values, running_avg)
            train_loss_values = np.save(os.path.join(Config.DRIVE_PATH, 'train_loss.npy'), train_loss_values)
        else:
            train_loss_values = np.array([running_avg])
            train_loss_values = np.save(os.path.join(Config.DRIVE_PATH, 'train_loss.npy'), train_loss_values)

    else:
        if os.path.exists(os.path.join(Config.DRIVE_PATH, 'val_loss.npy')):
            val_loss_values = np.load(os.path.join(Config.DRIVE_PATH, 'val_loss.npy'))
            val_loss_values = np.append(val_loss_values, running_avg)
            val_loss_values = np.save(os.path.join(Config.DRIVE_PATH, 'val_loss.npy'), val_loss_values)
        else:
            val_loss_values = np.array([running_avg])
            val_loss_values = np.save(os.path.join(Config.DRIVE_PATH, 'val_loss.npy'), val_loss_values)

def train():
    # Initialize out dir
    if not os.path.exists(Config.OUT_DIR):
        os.mkdir(Config.OUT_DIR)

    print('Device:', Config.DEVICE)

    model = CNNImageClassification().to(Config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    start_epoch = 0
    start_batch = 1

    if Config.LOAD_MODEL:
        checkpoint = torch.load(os.path.join(Config.DRIVE_PATH, Config.CHECKPOINT_PATH),
                                map_location=Config.DEVICE)

        start_batch = checkpoint["batch"]
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number Parameters:", pytorch_total_params)
    
    trainloader, testloader = load_dataset()

    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        print(f"[Epoch] {epoch}/{Config.NUM_EPOCHS - 1}")

        one_epoch(model, trainloader, loss_function, epoch, start_batch, optimizer, train=True)
        one_epoch(model, testloader, loss_function, epoch, start_batch, optimizer, train=False)
    
if __name__ == '__main__':
    train()