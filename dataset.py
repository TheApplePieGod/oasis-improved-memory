from torch.utils.data import Dataset
import torch

class OasisDataset(Dataset):
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len

    def __getitem__(self, idx):
        # T C H W
        return torch.rand((self.max_seq_len, 3, 360, 640))

    def __len__(self):
        return 200
