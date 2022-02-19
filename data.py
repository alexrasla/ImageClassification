import pickle
import torch

def load_dataset(current_batch):
    dic = unpickle(f'./data/data_batch_{current_batch}')
    
    features = torch.from_numpy(dic[b'data'])
    labels = torch.tensor(dic[b'labels'], dtype=torch.int)
    return features, labels 

def unpickle(file):  
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict    