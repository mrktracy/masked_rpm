import numpy as np
import torch
from torch.utils.data import Dataset

# Dataset for training and evaluation, corresponding to model v14
class RPMFullSentencesRaw_v1(Dataset):
    def __init__(self, files, embed_dim, device):
        self.files = files
        self.embed_dim = embed_dim
        self.device = device

    def __getitem__(self, idx):
        mask_exp = torch.ones(self.embed_dim).to(self.device)  # create mask token for tensor output

        filename = self.files[idx]
        data = np.load(filename)
        image = data['image']
        imagetensor = torch.from_numpy(image).float() / 255  # convert context panels to tensor
        imagetensor = imagetensor.unsqueeze(1).to(self.device) # shape (16, 1, 160, 160)

        target_num = data['target'].item()
        cands = imagetensor[8:, :, :, :] # extract candidate panel embeddings
        target = imagetensor[8+target_num, :, :, :]  # extract target panel embeddings

        sentence = imagetensor[0:8, :, :, :] # size (8, 1, 160, 160)
        mask = torch.ones([1,1,160,160]).to(self.device)  # create masking token
        masked_sentence = torch.cat([sentence, mask], 0)  # create masked sentence

        return masked_sentence, cands, target_num, target

    def __len__(self):
        length = len(self.files)
        return length