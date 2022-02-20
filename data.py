import pickle
import torch
import numpy as np

def load_dataset():
    
    features = np.zeros((50000, 3072))
    for i in range(1, 6):
        dic = unpickle(f'./data/data_batch_{i}')
           
        data = dic[b'data']
        features[(i-1)*data.shape[0]:i*data.shape[0]] = data
    
    features = torch.from_numpy(features)#[torch.from_numpy(img.reshape(32, 32, 3)) for img in dic[b'data']]
    labels = torch.tensor(dic[b'labels'], dtype=torch.int)
    return features, labels 

def unpickle(file):  
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict    