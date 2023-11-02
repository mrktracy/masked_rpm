# 1. Dataset
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random
import time
import re
import logging

VERSION = 'ae-v2-itr0'
logfile = f"../ae_results/{VERSION}/runlog.log"
os.makedirs(os.path.dirname(logfile), exist_ok=True)
logging.basicConfig(filename=logfile,level=logging.INFO)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def gather_files(root_dir):
    all_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.npz'):
                all_files.append(os.path.join(dirpath, filename))
    random.shuffle(all_files)
    return all_files

def gather_files_pgm(root_dir):
    all_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.npz'):
                all_files.append(os.path.join(dirpath, filename))
    random.shuffle(all_files)

    train_pattern = "train"
    val_pattern = "val"
    test_pattern = "test"

    train_files = [filename for filename in all_files if re.search(train_pattern, filename)]
    val_files = [filename for filename in all_files if re.search(val_pattern, filename)]
    test_files = [filename for filename in all_files if re.search(test_pattern, filename)]

    return train_files, val_files, test_files

class RPMPanels(Dataset):
    def __init__(self, files):
        self.files = files

    def __getitem__(self, idx):
        fileidx = idx // 16
        panelidx = idx % 16
        filename = self.files[fileidx]
        data = np.load(filename)
        image = data['image'].reshape([16,160,160])
        panel = torch.from_numpy(image[panelidx,:,:]).float() / 255
        label = panel.clone()

        return (panel.unsqueeze(0), label.unsqueeze(0))

    def __len__(self):
        length = len(self.files)*16
        return length

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNetAutoencoder(nn.Module):
    def __init__(self, embed_dim=768):
        super(ResNetAutoencoder, self).__init__()

        self.embed_dim = embed_dim

        self.encoder = nn.Sequential(
            ResidualBlock(1, 16), # N, 16, 160, 160
            ResidualBlock(16, 32, 2), # N, 32, 80, 80
            ResidualBlock(32, 64, 2), # N, 64, 40, 40
            ResidualBlock(64, 128, 2), # N, 128, 20, 20
            ResidualBlock(128, 256, 2), # N, 256, 10, 10
            nn.Flatten(), # N, 256*10*10
            nn.Linear(256*10*10, self.embed_dim), # N, 512
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, 256*10*10),
            nn.Unflatten(1, (256,10,10)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # N, 128, 20, 20
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # N, 64, 40, 40
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # N, 32, 80, 80
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # N, 16, 160, 160
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1), # N, 1, 160, 160
            nn.Sigmoid()  # to ensure the output is in [0, 1] as image pixel intensities
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_embedding(self, x):
        x = self.encoder(x)
        return x  # reshaping to (batch_size, embed_dim)

    def decode(self,x):
        x = self.decoder(x)
        return x

def evaluate_model(model, dataloader, device, save_path):

    os.makedirs(save_path, exist_ok=True) # make file path if it doesn't exist, do nothing otherwise

    model.eval()
    with torch.no_grad():
        criterion = nn.MSELoss()
        total_loss = 0
        imgnum = 0
        for batch in dataloader:
            # assuming that the data loader returns images and labels, but we don't need labels here
            images, _ = batch
            # move images to the device, reshape them and ensure channel dimension is present
            images = images.to(device)
            # forward pass
            outputs = model(images)
            loss = criterion(outputs, images)
            total_loss += loss.item()
            idx = 0
            for image,output in zip(images,outputs):
                if imgnum % 50 == 0:
                    start_time = time.time()
                if idx >= 1: # only save first image from each mini-batch
                    break
                image = image.cpu().numpy()
                output = output.cpu().numpy()
                filename = f"eval_{imgnum}"
                np.savez(os.path.join(save_path,filename), image=image, output=output)
                imgnum += 1
                idx += 1
                if imgnum % 50 == 49:
                    end_time = time.time()
                    run_time = end_time - start_time
                    print(f"50 images processed in {run_time} seconds\n")

    return total_loss/len(dataloader.dataset)

def main():
    # Define Hyperparameters
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    LOGS_PER_EPOCH = 100
    BATCHES_PER_PRINT = 20
    EPOCHS_PER_SAVE = 1

    # Initialize device, data loader, model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    root_dir = '../i_raven_data/'
    train_files, val_files, test_files = gather_files_pgm(root_dir)

    # # Uncomment if using RAVEN data
    # root_dir = '../RAVEN-10000/'
    # all_files = gather_files(root_dir)
    # num_files = len(all_files)
    # train_proportion = 0.7
    # val_proportion = 0.15
    # # test proportion is 1 - train_proportion - val_proportion
    # train_files = all_files[:int(num_files*train_proportion)]
    # val_files = all_files[int(num_files * train_proportion):int(num_files * (train_proportion+val_proportion))]
    # test_files = all_files[int(num_files * (train_proportion+val_proportion)):]

    train_dataset = RPMPanels(train_files)
    val_dataset = RPMPanels(val_files)
    # test_dataset = RPMPanels(test_files)

    print("Training files: {}, validation files: {}, testing files: {}".format(len(train_files), len(val_files),\
                                                                               len(test_files)))

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    train_length = len(train_dataloader)
    batches_per_log = train_length // LOGS_PER_EPOCH

    autoencoder = ResNetAutoencoder().to(device)

    if num_gpus > 1: # use multiple GPUs
        autoencoder = nn.DataParallel(autoencoder)

    # Comment out if training
    # state_dict = torch.load('../modelsaves/autoencoder_v1_ep1.pth')
    # autoencoder.load_state_dict(state_dict)
    # autoencoder.eval()

    optimizer = torch.optim.Adam(list(autoencoder.parameters()),
                                 lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(EPOCHS):
        tot_loss = 0
        count = 0
        for idx, (images,_) in enumerate(train_dataloader):

            if idx%BATCHES_PER_PRINT == 0:
                start_time = time.time()

            # move images to the device, reshape them and ensure channel dimension is present
            images = images.to(device)

            # forward pass
            outputs = autoencoder(images)
            loss = criterion(outputs, images)

            # backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            tot_loss += loss.item()
            count += 1

            if (idx + 1) % BATCHES_PER_PRINT == 0:
                end_time = time.time()
                batch_time = end_time - start_time
                print(
                    f"{BATCHES_PER_PRINT} batches processed in {batch_time:.2f} seconds. Training loss: {tot_loss / count}")

            if (idx + 1) % batches_per_log == 0:
                output = f"Epoch {epoch + 1} - {idx + 1}/{train_length}. loss: {tot_loss / count:.4f}."
                print(output)
                logging.info(output)
                tot_loss = 0
                count = 0

        if (epoch + 1) % EPOCHS_PER_SAVE == 0:
            save_file = f"../modelsaves/{VERSION}/{VERSION}_ep{epoch + 1}.pth"
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save(autoencoder.state_dict(), save_file)

    # Evaluate the model
    avg_val_loss = evaluate_model(autoencoder, val_dataloader, device, save_path=f"../ae_results/{VERSION}")

    output_file_path = f"../ae_results/{VERSION}/avg_val_loss.txt"
    with open(output_file_path, "w") as file:
        file.write(f"Average validation loss: {avg_val_loss}")

if __name__ == "__main__":
    main()