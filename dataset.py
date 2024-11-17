from torch.utils.data import Dataset
import torch

class OasisDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        # C H W
        return torch.rand((3, 360, 640))

    def __len__(self):
        return 200
