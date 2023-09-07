## Use transformer output directly, without subsequent MLP layers
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from main_ae import ResNetAutoencoder, gather_files, gather_files_pgm
import time
import random
from evaluate import evaluate_model
from datasets import RPMSentencesNew, RPMSentencesRaw, CustomMNIST
from models import TransformerModelv6, TransformerModelv4, TransformerModelv5, TransformerModelMNIST
import os

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
    transformer_model = TransformerModelv6(embed_dim=256, cat=True).to(device)

    # initialize weights
    transformer_model.apply(initialize_weights_he)

    # # initialize autoencoder
    # autoencoder = ResNetAutoencoder(embed_dim=256).to(device)

    if num_gpus > 1:  # use multiple GPUs
        transformer_model = nn.DataParallel(transformer_model)
        # autoencoder = nn.DataParallel(autoencoder) # uncomment if using PGM

    # state_dict = torch.load('../modelsaves/autoencoder_v1_ep1.pth')
    # state_dict = torch.load('../modelsaves/autoencoder_v0.pth')
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
    root_dir = '../RAVEN-10000'
    all_files = gather_files(root_dir)
    num_files = len(all_files)
    train_proportion = 0.7
    val_proportion = 0.15
    # test proportion is 1 - train_proportion - val_proportion
    train_files = all_files[:int(num_files * train_proportion)]
    val_files = all_files[int(num_files * train_proportion):int(num_files * (train_proportion + val_proportion))]
    # test_files = all_files[int(num_files * (train_proportion + val_proportion)):]

    train_files = train_files[0:150]
    val_files = train_files[0:150]

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

    ''' Transformer model v2 to v4 '''
    # train_dataset = RPMSentencesNew(train_files, autoencoder, device=device)
    # val_dataset = RPMSentencesNew(val_files, autoencoder, device=device)

    ''' Transformer model v5, v6 '''
    train_dataset = RPMSentencesRaw(train_files)
    val_dataset = RPMSentencesRaw(val_files)

    ''' MNIST transformer model '''
    # train_dataset = CustomMNIST(mnist_train, num_samples=100000)
    # val_dataset = CustomMNIST(mnist_val, num_samples=10000)

    ''' Define Hyperparameters '''
    EPOCHS = 40
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    TOTAL_DATA = len(train_dataset)  # training dataset size
    SAVES_PER_EPOCH = 2
    BATCHES_PER_SAVE = TOTAL_DATA // BATCH_SIZE // SAVES_PER_EPOCH
    VERSION = "v6-itr0"
    VERSION_SUBFOLDER = "" # e.g. "MNIST/" or ""

    ''' Instantiate data loaders, optimizer, criterion '''
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(list(transformer_model.parameters()),
                                 lr=LEARNING_RATE)

    scheduler = StepLR(optimizer, step_size=2, gamma=0.9)
    criterion = nn.CrossEntropyLoss()

    log_file_path = f"../tr_results/{VERSION}/{VERSION_SUBFOLDER}runlog.txt"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Training loop
    for epoch in range(EPOCHS):
        for idx, (inputs, targets) in enumerate(train_dataloader):

            if idx%10 == 0:
                start_time = time.time()

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = transformer_model(inputs) # (B,8)
            loss = criterion(outputs,targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if idx%10 == 9:
                end_time = time.time()
                batch_time = end_time - start_time
                print(f"10 mini-batches processed in {batch_time} seconds")
                print(f"Most recent batch total loss: {loss.item()}\n")

            # save multiple times per epoch
            if idx%BATCHES_PER_SAVE == BATCHES_PER_SAVE - 1:
                model_path = f"../modelsaves/{VERSION}/{VERSION_SUBFOLDER}transformer_{VERSION}_ep{epoch + 1}_sv{idx//BATCHES_PER_SAVE+1}.pth"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(transformer_model.state_dict(), model_path)

                proportion_correct = evaluate_model(transformer_model, val_dataloader, device)
                with open(log_file_path, "a") as file:
                    file.write(f"Epoch {epoch+1}, save point {idx//BATCHES_PER_SAVE+1}:\n")
                    file.write(f"Most recent training loss: {loss.item()}.\n")
                    file.write(f"Validation accuracy: {proportion_correct}.\n\n")

        print(f"Epoch {epoch+1}/{EPOCHS} completed: loss = {loss.item()}\n")

    # # Evaluate the model
    # proportion_correct = evaluate_model(transformer_model, val_dataloader, device=device)
    # print(f"Proportion of answers correct: {proportion_correct}")
    #
    # output_file_path = f"../tr_results/{VERSION}/{VERSION_SUBFOLDER}final_proportion_correct.txt"
    # os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    # with open(output_file_path, "w") as file:
    #     file.write(f"Proportion of answers correct: {proportion_correct}.")

if __name__ == "__main__":
    main()