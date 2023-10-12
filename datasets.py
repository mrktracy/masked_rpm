import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import random
from collections import defaultdict
from transformers import ViTImageProcessor, ViTModel, ViTConfig

class RPMSentencesViT(Dataset):
    def __init__(self, files, ViT_model_name, device, num_gpus):
        self.files = files
        self.device = device

        # set separately calculated mean and std of pixel values
        mean = 0.9031295340401794 * 255
        std = 0.263461851960206 * 255

        # Initialize feature extractor and ViT model
        configuration = ViTConfig.from_pretrained(ViT_model_name, num_channels=1)
        self.feature_extractor = ViTImageProcessor.from_pretrained(ViT_model_name, \
                                                                   do_rescale=False, \
                                                                   image_mean = mean, \
                                                                   image_std = std)
        encoder = ViTModel(configuration).to(device)
        if num_gpus > 1:  # use multiple GPUs
            encoder = nn.DataParallel(encoder)
        self.encoder = encoder

        # Ensure encoder is in eval mode and gradients are not computed
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def __getitem__(self, idx):
        filename = self.files[idx]
        data = np.load(filename)
        images = data['image'].reshape(16, 1, 160, 160)

        # Preprocessing for ViT
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        inputs = {key:val.to(self.device) for key,val in inputs.items()}

        # Get embeddings using Vision Transformer
        with torch.no_grad():
            vit_outputs = self.encoder(**inputs)

        # Extract embedding of the 'CLS' token
        embeddings = vit_outputs.last_hidden_state[:, 0, :]

        target = data['target'].item()
        return embeddings, target

    def __len__(self):
        return len(self.files)

class CustomMNIST(Dataset):
    def __init__(self, mnist_data, num_samples):
        self.mnist_data = mnist_data
        self.num_samples = num_samples

        self.label_to_images = defaultdict(list)
        for img, label in mnist_data:
            self.label_to_images[label].append(img)

        self.random_nums = [random.randint(1,8) for _ in range(num_samples)]
        self.random_order = [random.randint(0, 1) for _ in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        random_num = self.random_nums[idx]
        random_order = self.random_order[idx]

        low_num = random_num - 1
        high_num = random_num + 1

        low_imgs = random.sample(self.label_to_images[low_num],8)
        high_imgs = random.sample(self.label_to_images[high_num],8)

        if random_order == 0:
            question_imgs = low_imgs[0:8] + high_imgs[0:8]
        else:
            question_imgs = high_imgs[0:8] + low_imgs[0:8]

        question_tensor = torch.stack(question_imgs)
        target = random_num - 1

        return question_tensor, target

class RPMSentencesRaw(Dataset):
    def __init__(self, files):
        self.files = files

    def __getitem__(self, idx):

        filename = self.files[idx]
        data = np.load(filename)
        image = data['image'].reshape(16,160,160)
        imagetensor = torch.from_numpy(image).float() / 255 # convert context panels to tensor
        imagetensor = imagetensor.unsqueeze(1)

        target = data['target'].item()

        return imagetensor, target

    def __len__(self):
        length = len(self.files)
        return length

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

        # get panel embeddings
        # num_gpus = torch.cuda.device_count()
        # embeddings = self.autoencoder.module.get_embedding(imagetensor) if num_gpus > 1 else self.autoencoder.get_embedding(imagetensor)
        embeddings = self.autoencoder.get_embedding(imagetensor)

        target = data['target'].item()

        return embeddings, target

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