import numpy as np
import torch
from torch.utils.data import Dataset

# Dataset for training and evaluation, corresponding to model v17
class RPMFullSentences_evalByType(Dataset):
    def __init__(self, df, device):
        assert "file" in df.columns, "Dataframe must have column 'file'"
        assert "folder" in df.columns, "Dataframe must have column 'folder'"

        self.files = df["file"].reset_index(drop=True)
        self.folders = df["folder"].reset_index(drop=True)
        self.device = device

    def __getitem__(self, idx):

        fileidx = idx // 16
        rot_idx = idx % 16 # index of rotation for data augmentation
        inner_rot = idx % 4 # inner rotation
        outer_rot = rot_idx // 4

        filename = self.files[fileidx]
        data = np.load(filename)
        image = data['image']
        imagetensor = torch.from_numpy(image).float() / 255  # convert context panels to tensor
        imagetensor = imagetensor.unsqueeze(1).to(self.device) # shape (16, 1, 160, 160)

        imagetensor = torch.rot90(imagetensor, k=inner_rot, dims=[-2, -1]) # rotate inner
        target_num = data['target'].item()

        context = imagetensor[0:8, :, :, :] # size (8, 1, 160, 160)
        candidates = imagetensor[8:, :, :, :] # size (8, 1, 160, 160)

        context_expanded = context.unsqueeze(0).expand(8,-1,-1,-1,-1)

        # Concatenate context and candidates along the second dimension
        sentences = torch.cat([context_expanded, candidates.unsqueeze(1)], dim = 1)

        # form grids for rotation
        sentences_grid = sentences.reshape(8, 3, 3, 1, 160, 160) # sentences is (8, 9, 1, 160, 160)
        sentences_grid = torch.rot90(sentences_grid, k=outer_rot, dims=[1,2]) # rotate
        sentences = sentences_grid.reshape(8, 9, 1, 160, 160) # reshape

        folder = self.folders.loc[fileidx]

        return sentences, target_num, folder, filename

    def __len__(self):
        length = len(self.files) * 16
        return length

# Dataset for training and evaluation, corresponding to model v17
class RPMFullSentencesRaw_dataAug(Dataset):
    def __init__(self, files, device):
        self.files = files
        self.device = device

    def __getitem__(self, idx):

        fileidx = idx // 16
        rot_idx = idx % 16 # index of rotation for data augmentation
        inner_rot = idx % 4 # inner rotation
        outer_rot = rot_idx // 4

        filename = self.files[fileidx]
        data = np.load(filename)
        image = data['image']
        imagetensor = torch.from_numpy(image).float() / 255  # convert context panels to tensor
        imagetensor = imagetensor.unsqueeze(1).to(self.device) # shape (16, 1, 160, 160)

        imagetensor = torch.rot90(imagetensor, k=inner_rot, dims=[-2, -1]) # rotate inner
        target_num = data['target'].item()

        context = imagetensor[0:8, :, :, :] # size (8, 1, 160, 160)
        candidates = imagetensor[8:, :, :, :] # size (8, 1, 160, 160)

        context_expanded = context.unsqueeze(0).expand(8,-1,-1,-1,-1)

        # Concatenate context and candidates along the second dimension
        sentences = torch.cat([context_expanded, candidates.unsqueeze(1)], dim = 1)

        # form grids for rotation
        sentences_grid = sentences.reshape(8, 3, 3, 1, 160, 160) # sentences is (8, 9, 1, 160, 160)
        sentences_grid = torch.rot90(sentences_grid, k=outer_rot, dims=[1,2]) # rotate
        sentences = sentences_grid.reshape(8, 9, 1, 160, 160) # reshape

        return sentences, target_num, None, None

    def __len__(self):
        length = len(self.files) * 16
        return length

# Dataset for training and evaluation, corresponding to model v17
class RPMFullSentencesRaw_base(Dataset):
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

        return sentences, target_num, None, None

    def __len__(self):
        length = len(self.files)
        return length