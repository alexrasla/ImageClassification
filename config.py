import torch
from model import LargeImageClassification, BaselineImageClassification

class Config():
    MODEL = BaselineImageClassification()
    BATCH_SIZE = 100
    IMG_DIM = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    OUT_DIR = 'output'
    DATA_DIR = 'data'
    DRIVE_PATH = 'drive/MyDrive'
    CHECKPOINT_PATH = 'checkpoint.pth'
    LOAD_MODEL = False