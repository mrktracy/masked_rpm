import numpy as np
import torch
from torch.utils.data import Dataset

# Dataset for training and evaluation, corresponding to model v17
class RPMFullSentencesRaw_v2(Dataset):
    def __init__(self, files, device):
        self.files = files
        self.device = device

    def __getitem__(self, idx):

        filename = self.files[idx]
        data = np.load(filename)
        image = data['image']
        imagetensor = torch.from_numpy(image).float() / 255  # convert context panels to tensor
        imagetensor = imagetensor.unsqueeze(1).to(self.device) # shape (16, 1, 160, 160)

        target_num = data['target'].item()

        context = imagetensor[0:8, :, :, :] # size (8, 1, 160, 160)
        candidates = imagetensor[8:, :, :, :] # size (8, 1, 160, 160)

        context_expanded = context.unsqueeze(0).expand(8,-1,-1,-1,-1)

        # Concatenate context and candidates along the second dimension

        sentences = torch.cat([context_expanded, candidates.unsqueeze(1)], dim = 1)

        return sentences, target_num

    def __len__(self):
        length = len(self.files)
        return length