from sklearn.metrics import confusion_matrix
import torch
import numpy as np
from torch import nn
import os
from data import load_dataset
from model import BaselineImageClassification, LargeImageClassification, BestImageClassification
from config import Config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model")
args = parser.parse_args()

def eval(model_path):
    # Initialize out dir
    print('Device:', Config.DEVICE)

    model = Config.MODEL.to(Config.DEVICE)
    model.eval()
   
    checkpoint = torch.load(model_path,
                            map_location=Config.DEVICE)

    model.load_state_dict(checkpoint["model_state"])
    
    trainloader, testloader = load_dataset()
    
    y_pred = np.empty((Config.BATCH_SIZE) * len(testloader))
    y_true = np.empty((Config.BATCH_SIZE) * len(testloader))
    
    for index, data in enumerate(testloader, 0):
        with torch.no_grad():

            #get batch of features, labels
            features, labels = data
            features, labels = features.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            output = model(features.float())
            
            predicted = torch.nn.functional.softmax(output, dim=-1)
            predicted = np.argmax(predicted.cpu().numpy(), axis=-1)
            
            truth = labels.cpu().numpy()
            
            y_pred[index*Config.BATCH_SIZE : (index+1)*Config.BATCH_SIZE] = predicted
            y_true[index*Config.BATCH_SIZE : (index+1)*Config.BATCH_SIZE] = truth

    if not os.path.exists(Config.OUT_DIR):
        os.mkdir(Config.OUT_DIR)
    
    res = confusion_matrix(y_true, y_pred)
    np.save(os.path.join(Config.OUT_DIR, "confusion_matrix"), res)
    
    num_correct = np.trace(res)
    num_images = np.sum(res)
    
    print("Overall Accuracy:", num_correct/num_images)
   
    
if __name__ == '__main__':
    model_path = args.model
    eval(model_path)