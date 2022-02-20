import pickle
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from config import Config

def load_dataset():
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=Config.BATCH_SIZE,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=Config.BATCH_SIZE,
                                            shuffle=False, num_workers=2)
    
    return trainloader, testloader
    
    # features = np.zeros((50000, 3072))
    # for i in range(1, 6):
    #     dic = unpickle(f'./data/data_batch_{i}')
           
    #     data = dic[b'data']
    #     features[(i-1)*data.shape[0]:i*data.shape[0]] = data
    
    # features = torch.from_numpy(features)#[torch.from_numpy(img.reshape(32, 32, 3)) for img in dic[b'data']]
    # labels = torch.tensor(dic[b'labels'], dtype=torch.int)
    # return features, labels 

def unpickle(file):  
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict    