## Use transformer output directly, without subsequent MLP layers
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from main_ae import ResNetAutoencoder, gather_files, gather_files_pgm
import time
import random
from evaluate import evaluate_model
from datasets import RPMSentencesNew, RPMSentencesRaw, CustomMNIST
from models import TransformerModelv7, TransformerModelv4, TransformerModelv5, TransformerModelMNIST
import os
import logging

logfile = "../tr_results/v6-itr1/runlog.log"
os.makedirs(os.path.dirname(logfile), exist_ok=True)
logging.basicConfig(filename=logfile,level=logging.INFO)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def initialize_weights_he(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def main():

    # Initialize device, model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    # transformer_model = TransformerModelv5(embed_dim=512, num_heads=64, abstr_depth=20, reas_depth=20, \
    #                                         cat=False).to(device)
    # transformer_model = TransformerModelMNIST(embed_dim=256, num_heads=16).to(device)
    # transformer_model = TransformerModelv3(embed_dim=256, num_heads=16, con_depth=20, can_depth=20, \
    #                                        guess_depth=20, cat=True).to(device)
    transformer_model = TransformerModelv6(embed_dim=512, con_depth=10, can_depth=10, \
                                           guess_depth=10, cat=False).to(device)

    # initialize weights
    transformer_model.apply(initialize_weights_he)

    # # initialize autoencoder
    # autoencoder = ResNetAutoencoder(embed_dim=256).to(device)

    if num_gpus > 1:  # use multiple GPUs
        transformer_model = nn.DataParallel(transformer_model, device_ids=["cuda:0", "cuda:3"])
        # autoencoder = nn.DataParallel(autoencoder) # uncomment if using PGM

    # state_dict = torch.load('../modelsaves/autoencoder_v1_ep1.pth') # for PGM
    # state_dict = torch.load('../modelsaves/autoencoder_v0.pth') # for RAVEN
    # autoencoder.load_state_dict(state_dict)
    # autoencoder.eval()

    ''' Load saved model '''
    # state_dict_tr = torch.load('../modelsaves/transformer_v2_ep14.pth')
    # transformer_model.load_state_dict(state_dict_tr)
    # transformer_model.eval()

    ''' Use PGM dataset '''
    # root_dir = '../pgm/neutral/'
    # train_files, val_files, test_files = gather_files_pgm(root_dir)
    # train_files = train_files[0:32] # delete this after test
    # val_files = train_files[0:32] # delete this after test

    ''' Use RAVEN dataset '''
    root_dir = '../i_raven_data'
    all_files = gather_files(root_dir)
    num_files = len(all_files)
    train_proportion = 0.7
    val_proportion = 0.15
    # test proportion is 1 - train_proportion - val_proportion
    train_files = all_files[:int(num_files * train_proportion)]
    val_files = all_files[int(num_files * train_proportion):int(num_files * (train_proportion + val_proportion))]
    # test_files = all_files[int(num_files * (train_proportion + val_proportion)):]

    ''' Use MNIST dataset '''
    # train_proportion = 0.85
    # val_proportion = 0.15
    # mnist_data = MNIST(root='../MNIST/', train=True, download=True, \
    #                    transform=transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()]))
    # mnist_len = len(mnist_data)
    # train_len = int(mnist_len*train_proportion)
    # val_len = int(mnist_len*val_proportion)
    #
    # mnist_train, mnist_val = random_split(mnist_data, [train_len, val_len])

    ''' Transformer model v2 to v4, v7 '''
    # train_dataset = RPMSentencesNew(train_files, autoencoder, device=device)
    # val_dataset = RPMSentencesNew(val_files, autoencoder, device=device)

    ''' Transformer model v5, v6 '''
    train_dataset = RPMSentencesRaw(train_files)
    val_dataset = RPMSentencesRaw(val_files)

    ''' MNIST transformer model '''
    # train_dataset = CustomMNIST(mnist_train, num_samples=100000)
    # val_dataset = CustomMNIST(mnist_val, num_samples=10000)

    ''' Define Hyperparameters '''
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.00001
    LOGS_PER_EPOCH = 4
    BATCHES_PER_PRINT = 35
    EPOCHS_PER_SAVE = 5
    VERSION = "v6-itr1"
    VERSION_SUBFOLDER = "" # e.g. "MNIST/" or ""

    ''' Instantiate data loaders, optimizer, criterion '''
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_length = len(train_dataloader)
    batches_per_log = train_length // LOGS_PER_EPOCH

    optimizer = torch.optim.Adam(list(transformer_model.parameters()),
                                 lr=LEARNING_RATE)

    scheduler = ExponentialLR(optimizer, gamma=0.98)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(EPOCHS):
        for idx, (inputs, targets) in enumerate(train_dataloader):

            if idx % BATCHES_PER_PRINT == 0:
                start_time = time.time()

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = transformer_model(inputs) # (B,8)
            loss = criterion(outputs,targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (idx+1) % BATCHES_PER_PRINT == 0:
                end_time = time.time()
                batch_time = end_time - start_time
                print(f"{BATCHES_PER_PRINT} batches processed in {batch_time:.2f} seconds. Training loss: {loss.item()}")

            if (idx+1) % batches_per_log == 0:
                val_loss = evaluate_model(transformer_model, val_dataloader, device, max_batches=150)
                output = f"Epoch {epoch+1} - {idx+1}/{train_length}. loss: {loss.item():.4f}. lr: {scheduler.get_last_lr()[0]:.6f}. val: {val_loss:.2f}"
                print(output)
                logging.info(output)

        if (epoch+1) % EPOCHS_PER_SAVE == 0:
            save_file = f"../modelsaves/{VERSION}/{VERSION_SUBFOLDER}tf_{VERSION}_ep{epoch + 1}.pth"
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save(transformer_model.state_dict(), save_file)

        scheduler.step()

if __name__ == "__main__":
    main()