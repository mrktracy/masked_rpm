import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from main_ae import ResNetAutoencoder, gather_files
from timm.models.vision_transformer import Block
import time

# 1. Dataset
class RPMSentences(Dataset):
    def __init__(self, files, ResNetAutoencoder, embed_dim):
        self.files = files
        self.autoencoder = ResNetAutoencoder
        self.embed_dim = embed_dim

    def __getitem__(self, idx):
        mask = torch.ones(self.embed_dim) # create masking token
        pad = torch.zeros([1,self.embed_dim]) # create padding token

        fileidx = idx // 8
        panelidx = idx % 8

        filename = self.files[fileidx]
        data = np.load(filename)
        image = data['image']
        imagetensor = torch.from_numpy(image[0:8,:,:]).float() / 255 # convert context panels to tensor

        embeddings = self.autoencoder.get_embedding(imagetensor) # get panel embeddings
        maskedsentence = embeddings.clone() # create masked sentence
        maskedsentence[panelidx, :] = mask # replace one panel with mask token
        paddedmaskedsentence = torch.cat([maskedsentence, pad], 0)

        target = embeddings[panelidx, :] # extract target panel embedding

        return paddedmaskedsentence, target

    # Dataset for evaluation
    class RPMFullSentences(Dataset):
        def __init__(self, files, ResNetAutoencoder, embed_dim):
            self.files = files
            self.autoencoder = ResNetAutoencoder
            self.embed_dim = embed_dim

        def __getitem__(self, idx):
            mask = torch.ones([1,self.embed_dim])  # create masking token

            filename = self.files[idx]
            data = np.load(filename)
            image = data['image']
            target_num = data['target'].item()
            imagetensor = torch.from_numpy(image).float() / 255  # convert context panels to tensor

            embeddings = self.autoencoder.get_embedding(imagetensor)  # get panel embeddings
            sentence = embeddings[0:8, :]
            maskedsentence = torch.cat([sentence, mask], 0)  # create masked sentence

            target_embed = embeddings[target_num+8,:]  # extract target panel embedding

            return maskedsentence, target_embed, imagetensor, target_num

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
    def __init__(self, embed_dim=256, grid_size = 3, num_heads=8, mlp_ratio=4.,norm_layer=nn.LayerNorm, depth = 4):
        super(TransformerModel, self).__init__()

        # initialize and retrieve positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros([grid_size**2, embed_dim]), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(embed_dim=embed_dim, grid_size=grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        self.pooling = nn.AdaptiveAvgPool1d(output_size=1)

        self.fc1 = nn.Linear(256, 128)

        self.fc2 = nn.Linear(128, 256)

    def forward(self, x):
        x = x + self.pos_embed # add positional embeddings
        for blk in self.blocks: # multiheaded self-attention layer
            x = blk(x)
        x = self.norm(x)
        x = x.permute(0,2,1) # permute to pool
        x = self.pooling(x) # average pooling to get embed_dim-dimensional summary
        x = x.squeeze(-1) # eliminate last (pooled-across) dimension
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# 4. Evaluation Module
def evaluate_model(model, dataloader, autoencoder, save_path):
    os.makedirs(save_path, exist_ok=True)  # make file path if it doesn't exist, do nothing otherwise

    model.eval()
    with torch.no_grad():
        criterion = nn.MSELoss()
        total_loss = 0
        imgnum = 0
        for idx, (inputs, targets, imagetensors, target_nums) in enumerate(dataloader):

            # move images to the device
            inputs = inputs.to(device)

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            total_loss += loss.item()

            # get image form of guesses
            guess_images = autoencoder.decode(outputs)
            # get image form of target
            target_images = imagetensors[:,8+target_nums,:,:]

            idx = 0
            for guess, target in zip(guess_images, target_images):
                if idx >= 5:  # only save first 5 images from each mini-batch
                    break
                guess = guess.cpu().numpy()
                target = target.cpu().numpy()
                filename = f"eval_{imgnum}"
                np.savez(os.path.join(save_path, filename), guess=guess, target=target)
                imgnum += 1
                idx += 1

    return total_loss / len(dataloader.dataset)

def main():
    # Define Hyperparameters
    EPOCHS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    # Initialize device, data loader, model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    autoencoder = ResNetAutoencoder()
    state_dict = torch.load('../modelsaves/autoencoder_v0.pth')
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()

    train_dataset = RPMSentences(train_files, autoencoder, embed_dim=256)
    val_dataset = RPMFullSentences(val_files, autoencoder, embed_dim=256)
    # test_dataset = RPMPanels(test_files)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    transformer_model = TransformerModel().to(device)

    optimizer = torch.optim.Adam(list(transformer_model.parameters()),
                                 lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(EPOCHS):
        for idx, (inputs, targets) in enumerate(train_dataloader):

            if idx%2000 == 0:
                start_time = time.time()

            outputs = transformer_model.forward(inputs)

            loss = criterion(outputs,targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if idx%2000 == 1999:
                end_time = time.time()
                batch_time = end_time - start_time
                print(f"2000 batches processed in {batch_time} seconds")
                print(f"Most recent batch total loss: {loss.item()}\n")

    # Evaluate the model
    avg_val_loss = evaluate_model(transformer_model, val_dataloader, val_files, autoencoder, save_path='../tr_results/v0/')
    print(f"Average validation loss: {avg_val_loss}")

    torch.save(transformer_model.state_dict(), "../modelsaves/transformer_v0.pth")

if __name__ == "__main__":
    main()