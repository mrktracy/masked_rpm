import numpy as np
import torch
from torch.utils.data import Dataset

class RPMSentencesNew(Dataset):
    def __init__(self, files, ResNetAutoencoder, device):
        self.files = files
        self.autoencoder = ResNetAutoencoder
        self.device = device

    def __getitem__(self, idx):

        filename = self.files[idx]
        data = np.load(filename)
        image = data['image'].reshape(16,160,160)
        imagetensor = torch.from_numpy(image).float() / 255 # convert context panels to tensor
        imagetensor = imagetensor.unsqueeze(1).to(self.device)

        embeddings = self.autoencoder.get_embedding(imagetensor) # get panel embeddings

        target = data['target'].item()
        target_onehot = torch.zeros(8)
        target_onehot[target] = 1

        return embeddings, target_onehot, target

    def __len__(self):
        length = len(self.files)
        return length

# 1. Dataset
class RPMSentences(Dataset):
    def __init__(self, files, ResNetAutoencoder, embed_dim, device):
        self.files = files
        self.autoencoder = ResNetAutoencoder
        self.embed_dim = embed_dim
        self.device = device

    def __getitem__(self, idx):
        mask = torch.ones(self.embed_dim).to(self.device) # create masking token
        mask_exp = torch.ones(self.embed_dim*2).to(self.device) # create mask for tensor output
        pad = torch.zeros([1,self.embed_dim]).to(self.device) # create padding token

        fileidx = idx // (8*4)
        panelidx = idx % 8

        filename = self.files[fileidx]
        data = np.load(filename)
        image = data['image']
        imagetensor = torch.from_numpy(image[0:8,:,:]).float() / 255 # convert context panels to tensor
        imagetensor = imagetensor.unsqueeze(1).to(self.device)

        embeddings = self.autoencoder.get_embedding(imagetensor) # get panel embeddings
        maskedsentence = embeddings.clone() # create masked sentence
        maskedsentence[panelidx, :] = mask # replace one panel with mask token
        paddedmaskedsentence = torch.cat([maskedsentence, pad], 0) # (9, 256)

        # rotate grid
        paddedmaskedgrid = paddedmaskedsentence.reshape([3, 3, self.embed_dim])
        paddedmaskedgrid_rotated = torch.rot90(paddedmaskedgrid, k=idx%4, dims=[0,1])
        final_sentence = paddedmaskedgrid_rotated.reshape([9, self.embed_dim])

        mask_tensor = torch.zeros(9, self.embed_dim*2)
        mask_tensor[panelidx, :] = mask_exp  # ones where the mask is, 0s elsewhere

        # rotate mask tensor
        maskgrid = mask_tensor.reshape([3, 3, self.embed_dim*2])
        maskgrid_rotated = torch.rot90(maskgrid, k=idx%4, dims=[0,1])
        final_mask_tensor = maskgrid_rotated.reshape([9, self.embed_dim*2])

        target = embeddings[panelidx, :] # extract target panel embedding

        return final_sentence, target, final_mask_tensor

    def __len__(self):
        length = len(self.files)*8
        return length

# Dataset for evaluation
class RPMFullSentences(Dataset):
    def __init__(self, files, ResNetAutoencoder, embed_dim, device):
        self.files = files
        self.autoencoder = ResNetAutoencoder
        self.embed_dim = embed_dim
        self.device = device

    def __getitem__(self, idx):
        mask = torch.ones([1,self.embed_dim]).to(self.device)  # create masking token
        mask_exp = torch.ones(self.embed_dim*2).to(self.device)  # create mask token for tensor output

        filename = self.files[idx]
        data = np.load(filename)
        image = data['image']
        target_num = data['target'].item()
        imagetensor = torch.from_numpy(image).float() / 255  # convert context panels to tensor
        imagetensor = imagetensor.unsqueeze(1).to(self.device)

        embeddings = self.autoencoder.get_embedding(imagetensor)  # get panel embeddings
        sentence = embeddings[0:8, :]
        maskedsentence = torch.cat([sentence, mask], 0)  # create masked sentence

        target_embed = embeddings[target_num+8,:]  # extract target panel embedding

        mask_tensor = torch.zeros(9, self.embed_dim*2)
        mask_tensor[8, :] = mask_exp  # ones where the mask is, 0s elsewhere

        return maskedsentence, target_embed, imagetensor, target_num, embeddings, mask_tensor

    def __len__(self):
        length = len(self.files)
        return length