import torch

class Config():
    BATCH_SIZE = 50
    IMG_DIM = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.01
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    OUT_DIR = 'output'
    DATA_DIR = 'data'
    DRIVE_PATH = 'drive/MyDrive'
    CHECKPOINT_PATH = 'checkpoint.pth'
    LOAD_MODEL = False