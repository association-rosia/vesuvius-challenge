import torch

FRAGMENTS_PATH = 'data/raw/train'
TRAIN_FRAGMENTS = ['1', '2']
VAL_FRAGMENTS = ['3']

Z_START = 27
Z_DIM = 8
TILE_SIZE = 256

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')