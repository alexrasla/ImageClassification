import timeit
import torch
from torch import nn
from torch import optim
import os
import numpy as np
from data import load_dataset
from model import CNNImageClassification
from config import Config

def one_epoch(model, dataloader, criterion, epoch, optimizer, train):
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
    start = timeit.default_timer()
    if train == True:
      model.train()
    else:
      model.eval()
    
    running_loss = 0

    for index, data in enumerate(dataloader, 0):

        if index % 100 == 0:
            if train:
                print(f"[Training] Epoch: {epoch}/{Config.NUM_EPOCHS - 1}, Batch Number: {index}/{len(dataloader)}")
            else:
                print(f"[Validating] Epoch: {epoch}/{Config.NUM_EPOCHS - 1}, Validating: {index}/{len(dataloader)}")

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

        running_loss += loss.item()
    end = timeit.default_timer()
    runtime = start - end
    return running_loss / len(dataloader), runtime

def save(train_loss, val_loss, train_runtime, val_runtime, epoch, model, optimizer):
    checkpoint = {
        "epoch":epoch,
        "model_state":model.state_dict(),
        "optim_state":optimizer.state_dict(),
        "train_loss":train_loss,
        "val_loss":val_loss,
        "train_runtime":train_runtime,
        "val_runtime":val_runtime
    }
    
    torch.save(checkpoint, os.path.join(Config.DRIVE_PATH, Config.CHECKPOINT_PATH))
    
def train():
    # Initialize out dir
    if not os.path.exists(Config.OUT_DIR):
        os.mkdir(Config.OUT_DIR)

    print('Device:', Config.DEVICE)

    model = CNNImageClassification().to(Config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    start_epoch = 0

    if Config.LOAD_MODEL:
        checkpoint = torch.load(os.path.join(Config.DRIVE_PATH, Config.CHECKPOINT_PATH),
                                map_location=Config.DEVICE)

        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number Parameters:", pytorch_total_params)
    
    trainloader, testloader = load_dataset()

    train_loss = []
    val_loss = []
    train_runtime = []
    val_runtime = []
    
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        print("-------------------------------------")
        print(f"[Epoch] {epoch}/{Config.NUM_EPOCHS - 1}")

        train_loss_val, train_time = one_epoch(model, trainloader, loss_function, epoch, optimizer, train=True) 
        val_loss_val, val_time = one_epoch(model, testloader, loss_function, epoch, optimizer, train=False)

        train_loss.append(train_loss_val)
        train_runtime.append(train_time)
        val_loss.append(val_loss_val)
        val_runtime.append(val_time)
        
        save(train_loss, val_loss, train_runtime, val_runtime, epoch, model, optimizer)
        print(f"Train Loss:", train_loss, ", Runtime:", train_time)
        print(f"Val Loss:  ", val_loss, ", Runtime:", val_time)

    
if __name__ == '__main__':
    train()