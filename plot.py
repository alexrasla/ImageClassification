from matplotlib import pyplot as plt, ticker
from sklearn.metrics import plot_confusion_matrix
import torch
from config import Config
import os
import numpy as np
from data import get_label_names
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir")
args = parser.parse_args()

def plot_runtime(dir_path, save_path):
    model_path = os.path.join(dir_path, 'checkpoint.pth')
    
    checkpoint = torch.load(model_path,
                        map_location=Config.DEVICE)

    train_loss = checkpoint["train_loss"]
    val_loss = checkpoint["val_loss"]
    train_runtime = checkpoint["train_runtime"]
    val_runtime = checkpoint["val_runtime"]
    
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.5)


    ax.set_title("Loss Plot")
    ax.set_ylabel("Loss Value")
    ax.set_xlabel("Epochs")
    ax.plot(train_loss, label="training")
    ax.plot(val_loss, label="validation")
    ax.legend()

    # ax[1].set_title("Runtime Plots")
    # ax[1].set_ylabel("Runtime (Seconds)"
    plt.savefig(os.path.join(save_path, 'plots.png'))
    
    print('Training:', np.mean(train_runtime))
    print('Validation:', np.mean(val_runtime))

def plot_commission_omission(dir_path, save_path):
    
    confusion_matrix = np.load(os.path.join(dir_path, 'confusion_matrix.npy'))
    train_classes, test_classes = get_label_names()
    num_correct = np.trace(confusion_matrix)
    num_images = np.sum(confusion_matrix)
        
    print("Overall Accuracy:", num_correct/num_images)
    
    # errors of commission: 
    # when a classification procedure assigns pixels to a certain class that in fact don’t belong to it

    # errors of omission:
    # when pixels that in fact belong to one class, are included into other classes. 
    commision_error = []
    ommision_error = []

    for idx in range(len(confusion_matrix)):
        commission_sum = np.sum(confusion_matrix[:, idx])
        omission_sum = np.sum(confusion_matrix[idx])
        
        commision_val = (commission_sum - confusion_matrix[idx, idx])/ commission_sum
        omission_val = (omission_sum - confusion_matrix[idx, idx])/ omission_sum
        
        commision_error.append(commision_val)
        ommision_error.append(omission_val)
        
        # print(train_classes[idx], (commission_sum - confusion_matrix[idx, idx])/ commission_sum)
        # print(train_classes[idx], (omission_sum - confusion_matrix[idx, idx])/ omission_sum)
        
    fig, axs = plt.subplots(2)
    fig.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.5)


    axs[0].set_title("Commision Error")
    axs[0].set_ylabel("Percentage")
    axs[0].bar(train_classes, commision_error)
    plt.setp(axs[0].get_xticklabels(), rotation=30, horizontalalignment='right')


    axs[1].set_title("Omission Error")
    axs[1].set_ylabel("Percentage")
    axs[1].bar(train_classes, ommision_error)
    plt.setp(axs[1].get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.savefig(os.path.join(save_path, 'commission_omision.png'))

def plot_confusion_matrix(dir_path, save_path):
    confusion_matrix = np.load(os.path.join(dir_path, 'confusion_matrix.npy'))
    row_sums = confusion_matrix.sum(axis=1)
    confusion_matrix = np.round(confusion_matrix / row_sums[:, np.newaxis], 2)
    
    train_classes, test_classes = get_label_names()
    
    fig, ax = plt.subplots()
    
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('Model Label')
    ax.set_xlabel('True Label')
    ax.set_xticklabels(train_classes)
    ax.set_yticklabels(train_classes)
        
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues)

    for i in range(10):
        for j in range(10):
            c = confusion_matrix[j,i]
            ax.text(i, j, str(c), va='center', ha='center')
    
    # plt.matshow(confusion_matrix)
    plt.xticks(np.arange(0, 10, 1))
    plt.yticks(np.arange(0, 10, 1))
    plt.setp(ax.get_xticklabels(), rotation=30, visible=True)
    plt.setp(ax.get_yticklabels(), rotation=30, visible=True)
    plt.savefig(os.path.join(save_path, 'confusion.png'))
    
if __name__ == '__main__':
    dir_path = args.dir
    
    if not os.path.exists(os.path.join(dir_path, 'output')):
        os.mkdir(os.path.join(dir_path, 'output'))
    
    save_path = os.path.join(dir_path, 'output')
    
    plot_runtime(dir_path, save_path)
    plot_commission_omission(dir_path, save_path)
    plot_confusion_matrix(dir_path, save_path)