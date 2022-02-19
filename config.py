import torch

class Config():
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.01
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    OUT_DIR = 'output'
    DATA_DIR = 'data'
    DRIVE_PATH = 'drive/MyDrive'
    CHECKPOINT_PATH = 'checkpoint.pth'
    LOAD_MODEL = False
    
    # EMBEDDING_SIZE = 512
    # HIDDEN_SIZE = 512
    # D_MODEL = 512
    # NUM_HEADS = 8
    # FEED_FORWARD_DIM = 1024
    # NUM_LAYERS = 3
    # DROPOUT = 0.1