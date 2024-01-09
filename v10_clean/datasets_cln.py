import numpy as np
import torch
# from torch import nn
from torch.utils.data import Dataset
# import random
# from collections import defaultdict
# from transformers import ViTImageProcessor, ViTModel, ViTConfig

# 1. Dataset
class RPMSentencesSupervisedRaw(Dataset):
    def __init__(self, files, embed_dim, device):
        self.files = files
        self.embed_dim = embed_dim
        self.device = device

    def __getitem__(self, idx):
        fileidx = idx // (9*4)
        panelidx = idx % 9

        filename = self.files[fileidx]
        data = np.load(filename)
        image = data['image']
        indices = list(range(8)) + [8 + data['target']]
        imagetensor = torch.from_numpy(image[indices,:,:]).float() / 255 # convert context panels to tensor
        imagetensor = imagetensor.unsqueeze(1).to(self.device) # (9, 1, 160, 160)
        # imagetensor = 1-imagetensor # experiment with inverted colors

        # # experiment with normalization
        # pix_mean = 0.9031295340401794
        # pix_std = 0.263461851960206
        # imagetensor = (imagetensor - pix_mean)/pix_std

        target = imagetensor[panelidx, :, :, :].clone()  # extract target image
        imagetensor[panelidx, :, :, :] = torch.ones_like(target)  # replace with mask

        # create mask tensor for selecting output
        mask_tensor = torch.zeros(9, self.embed_dim)
        mask = torch.ones(self.embed_dim).to(self.device)  # create mask for tensor output
        mask_tensor[panelidx, :] = mask  # ones where the mask is, 0s elsewhere

        # rotate grid
        masked_sen_grid = imagetensor.reshape([3, 3, 1, 160, 160])
        masked_sen_grid_rotated = torch.rot90(masked_sen_grid, k=idx%4, dims=[0,1])
        final_sentence = masked_sen_grid_rotated.reshape([9, 1, 160, 160])

        # rotate mask tensor
        mask_grid = mask_tensor.reshape([3, 3, self.embed_dim])
        mask_grid_rotated = torch.rot90(mask_grid, k=idx%4, dims=[0,1])
        final_mask_tensor = mask_grid_rotated.reshape([9, self.embed_dim])

        return final_sentence, target, final_mask_tensor

    def __len__(self):
        length = len(self.files)*(9*4)
        return length