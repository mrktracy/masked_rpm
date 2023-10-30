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
from evaluate_masked import evaluate_model_masked
from datasets import RPMSentencesNew, RPMSentencesRaw, CustomMNIST, RPMSentencesViT_Masked, RPMFullSentencesViT_Masked
from models import TransformerModelv8, TransformerModelv7, TransformerModelMNISTv6, TransformerModelv5
import os
import logging

logfile = "../tr_results/v8-itr2/runlog.log"
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

    transformer_model = TransformerModelv8(depth=10, num_heads=16).to(device)

    # initialize weights
    transformer_model.apply(initialize_weights_he)

    if num_gpus > 1:  # use multiple GPUs
        transformer_model = nn.DataParallel(transformer_model)
        # transformer_model = nn.DataParallel(transformer_model, device_ids=["cuda:0", "cuda:3"])

    ''' Load saved model '''
    # state_dict_tr = torch.load('../modelsaves/transformer_v2_ep14.pth')
    # transformer_model.load_state_dict(state_dict_tr)
    # transformer_model.eval()

    ''' Use for PGM or I-RAVEN dataset '''
    # root_dir = '../pgm/neutral/'
    root_dir = '../i_raven_data/'
    train_files, val_files, test_files = gather_files_pgm(root_dir)

    ''' Use RAVEN dataset '''
    # root_dir = '../RAVEN-10000'
    # all_files = gather_files(root_dir)
    # num_files = len(all_files)
    # train_proportion = 0.7
    # val_proportion = 0.15
    # # test proportion is 1 - train_proportion - val_proportion
    # train_files = all_files[:int(num_files * train_proportion)]
    # val_files = all_files[int(num_files * train_proportion):int(num_files * (train_proportion + val_proportion))]
    # # test_files = all_files[int(num_files * (train_proportion + val_proportion)):]

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

    ''' Transformer model v8 '''
    train_dataset = RPMSentencesViT_Masked(train_files, \
                                    ViT_model_name="google/vit-base-patch16-224-in21k", \
                                    device = device, num_gpus = num_gpus)
    val_dataset = RPMFullSentencesViT_Masked(val_files, \
                                  ViT_model_name="google/vit-base-patch16-224-in21k", \
                                  device = device, num_gpus = num_gpus)

    ''' MNIST transformer model '''
    # train_dataset = CustomMNIST(mnist_train, num_samples=100000)
    # val_dataset = CustomMNIST(mnist_val, num_samples=10000)

    ''' Define Hyperparameters '''
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-1
    MOMENTUM = 0.90
    LOGS_PER_EPOCH = 20
    BATCHES_PER_PRINT = 20
    EPOCHS_PER_SAVE = 1
    VERSION = "v8-itr2"
    VERSION_SUBFOLDER = "" # e.g. "MNIST/" or ""

    ''' Instantiate data loaders, optimizer, criterion '''
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_length = len(train_dataloader)
    batches_per_log = train_length // LOGS_PER_EPOCH

    optimizer = torch.optim.SGD(list(transformer_model.parameters()),
                                 lr=LEARNING_RATE, momentum = MOMENTUM)

    scheduler = ExponentialLR(optimizer, gamma=0.98)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(EPOCHS):
        count = 0
        avg_loss = 0
        count_log = 0
        avg_loss_log = 0
        for idx, (inputs, targets) in enumerate(train_dataloader):

            if idx % BATCHES_PER_PRINT == 0:
                start_time = time.time()

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = transformer_model(inputs) # (B,embed_dim)
            loss = criterion(outputs,targets)

            avg_loss += loss.item() # update running averages
            avg_loss_log += loss.item()
            count += 1
            count_log += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (idx+1) % BATCHES_PER_PRINT == 0:
                end_time = time.time()
                batch_time = end_time - start_time
                print(f"{BATCHES_PER_PRINT} batches processed in {batch_time:.2f} seconds. Training loss: {avg_loss/count}")
                avg_loss = 0
                count = 0

            if (idx+1) % batches_per_log == 0:
                val_loss = evaluate_model_masked(transformer_model, val_dataloader, device, max_batches=150)
                output = f"Epoch {epoch+1} - {idx+1}/{train_length}. loss: {avg_loss_log/count_log:.4f}. lr: {scheduler.get_last_lr()[0]:.6f}. val: {val_loss:.2f}"
                print(output)
                logging.info(output)
                avg_loss_log = 0
                count_log = 0

        if (epoch+1) % EPOCHS_PER_SAVE == 0:
            save_file = f"../modelsaves/{VERSION}/{VERSION_SUBFOLDER}tf_{VERSION}_ep{epoch + 1}.pth"
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save(transformer_model.state_dict(), save_file)

        scheduler.step()

if __name__ == "__main__":
    main()