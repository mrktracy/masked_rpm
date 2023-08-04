## Use transformer output directly, without subsequent MLP layers
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
# import torch.nn.functional as F
from main_ae import ResNetAutoencoder, gather_files
from timm.models.vision_transformer import Block
import time
import os
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

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
        mask_tensor[9, :] = mask_exp  # ones where the mask is, 0s elsewhere

        return maskedsentence, target_embed, imagetensor, target_num, embeddings, mask_tensor

    def __len__(self):
        length = len(self.files)
        return length

# 2 Positional encodings: from FAIR's 'Masked Autoencoders...'
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

# 3 Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, embed_dim=256, grid_size = 3, num_heads=16, mlp_ratio=4.,norm_layer=nn.LayerNorm, depth = 4):
        super(TransformerModel, self).__init__()

        # initialize and retrieve positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros([grid_size**2, embed_dim]), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(embed_dim=embed_dim, grid_size=grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.blocks = nn.ModuleList([
            Block(embed_dim*2, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim*2)

        # self.flatten = nn.Flatten()
        #
        # self.fc1 = nn.Linear(256*9, 256*7)
        #
        # self.fc2 = nn.Linear(256*7, 256*5)
        #
        # self.fc3 = nn.Linear(256*5, 256*3)
        #
        # self.fc4 = nn.Linear(256*3, 256)

    def forward(self, x):
        batch_size = x.size(0)  # Get the batch size from the first dimension of x
        x = torch.cat([x, self.pos_embed.unsqueeze(0).expand(batch_size, -1, -1)], dim=1).unsqueeze(0)  # add positional embeddings

        for blk in self.blocks: # multi-headed self-attention layer
            x = blk(x)
        x = self.norm(x)
        # x = self.flatten(x) # flatten
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)

        return x

def pick_answers(outputs, candidates):
    mse = torch.mean((outputs.unsqueeze(1) - candidates)**2)
    min_indices = torch.argmin(mse, dim=-1)

    return min_indices

# 4. Evaluation Module
def evaluate_model(model, dataloader, autoencoder, save_path, device):
    os.makedirs(save_path, exist_ok=True)  # make file path if it doesn't exist, do nothing otherwise

    model.eval()
    with torch.no_grad():

        num_correct = 0
        imgnum = 0
        for idx, (inputs, targets, imagetensors, target_nums, embeddings, mask_tensors) in enumerate(dataloader):

            batch_size = len(inputs)
            offset_target_nums = target_nums + 8 # offset by 8

            # move images to the device
            inputs = inputs.to(device)
            mask_tensors = mask_tensors.to(device)

            # forward pass
            outputs = model(inputs) # (batch_size,9,256)
            guesses = (outputs * mask_tensors).sum(dim=1)
            guesses_cut = guesses[:,:,0:256]

            candidates = embeddings[:,8:,:].to(device) # embeddings is shape (batch_size, 16, 256)

            min_indices = pick_answers(guesses_cut, candidates).cpu()
            num_correct += torch.sum(min_indices == target_nums)

            guess_images = autoencoder.decode(guesses_cut) # get image form of guesses
            target_images = imagetensors[torch.arange(batch_size), offset_target_nums] # get image form of target
            decoded_target_images = autoencoder.decode(targets)

            # print(f"guess_images shape: {guess_images.shape}")
            # print(f"target_images shape: {target_images.shape}")
            # print(f"decoded_target_images shape: {decoded_target_images.shape}")

            idx = 0
            for guess, target, decoded_target in zip(guess_images, target_images, decoded_target_images):
                if idx >= 1:  # only save first 1 images from each mini-batch
                    break
                guess = guess.cpu().numpy()
                target = target.cpu().numpy()
                decoded_target = decoded_target.cpu().numpy()

                filename = f"eval_{imgnum}"
                np.savez(os.path.join(save_path, filename), guess=guess, target=target, decoded_target=decoded_target)
                imgnum += 1
                idx += 1

    return num_correct / len(dataloader.dataset)

def main():
    # Define Hyperparameters
    EPOCHS = 15
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01

    # Initialize device, data loader, model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    root_dir = '../RAVEN-10000'
    all_files = gather_files(root_dir)
    num_files = len(all_files)
    train_proportion = 0.7
    val_proportion = 0.15
    # test proportion is 1 - train_proportion - val_proportion
    train_files = all_files[:int(num_files * train_proportion)]
    val_files = all_files[int(num_files * train_proportion):int(num_files * (train_proportion + val_proportion))]
    # test_files = all_files[int(num_files * (train_proportion + val_proportion)):]

    # initialize autoencoder
    autoencoder = ResNetAutoencoder().to(device)
    state_dict = torch.load('../modelsaves/autoencoder_v0.pth')
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()

    train_dataset = RPMSentences(train_files, autoencoder, embed_dim=256, device=device)
    val_dataset = RPMFullSentences(val_files, autoencoder, embed_dim=256, device=device)
    # test_dataset = RPMPanels(test_files)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    transformer_model = TransformerModel(depth=12).to(device) # instantiate model

    # for name,param in transformer_model.named_parameters(): # initialize model
    #     if 'weight' in name:
    #         torch.nn.init.xavier_normal_(param)

    if num_gpus > 1: # use multiple GPUs
        transformer_model = nn.DataParallel(transformer_model)

    # # comment out this block if training
    # state_dict_tr = torch.load('../modelsaves/transformer_v0_ep14.pth')
    # transformer_model.load_state_dict(state_dict_tr)
    # transformer_model.eval()

    optimizer = torch.optim.Adam(list(transformer_model.parameters()),
                                 lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(EPOCHS):
        for idx, (inputs, targets, mask_tensors) in enumerate(train_dataloader):

            if idx%100 == 0:
                start_time = time.time()

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = transformer_model.forward(inputs) # (B,9,512)
            guesses = (outputs * mask_tensors).sum(dim=1) # (B, 1, 512)
            guesses_cut = guesses[:,:,0:256]

            loss = criterion(guesses_cut,targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if idx%100 == 99:
                end_time = time.time()
                batch_time = end_time - start_time
                print(f"100 mini-batches processed in {batch_time} seconds")
                print(f"Most recent batch total loss: {loss.item()}\n")

        torch.save(transformer_model.state_dict(), f"../modelsaves/transformer_v1_ep{epoch+1}.pth")
        print(f"Epoch {epoch+1}/{EPOCHS} completed: loss = {loss.item()}\n")

    # Evaluate the model
    proportion_correct = evaluate_model(transformer_model, val_dataloader, autoencoder, save_path='../tr_results/v2/', device=device)
    print(f"Proportion of answers correct: {proportion_correct}")

    output_file_path = "../tr_results/v2/proportion_correct.txt"
    with open(output_file_path, "w") as file:
        file.write(f"Proportion of answers correct: {proportion_correct}.")

if __name__ == "__main__":
    main()