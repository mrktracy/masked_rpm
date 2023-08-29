## Use transformer output directly, without subsequent MLP layers
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from main_ae import ResNetAutoencoder, gather_files, gather_files_pgm
import time
import random
from evaluate import evaluate_model
from datasets import RPMSentencesNew
from models import TransformerModelNew, TransformerModelNew16
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
    # Define Hyperparameters
    EPOCHS = 25
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    TOTAL_DATA = 49000 # data set size
    SAVES_PER_EPOCH = 1
    BATCHES_PER_SAVE = TOTAL_DATA // BATCH_SIZE // SAVES_PER_EPOCH

    # VERSION = 'v2'

    # Initialize device, data loader, model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    # initialize both stages of model
    # instantiate model
    # transformer_model = TransformerModelNew(embed_dim=256, num_heads=32, con_depth=10, can_depth=10,\
    #                                         guess_depth=10, cat=False).to(device)

    transformer_model = TransformerModelv5(embed_dim=256, num_heads=32, abstr_depth=14, reas_depth=10, \
                                            cat=True).to(device)

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

    # # comment out this block if training
    # state_dict_tr = torch.load('../modelsaves/transformer_v2_ep14.pth')
    # transformer_model.load_state_dict(state_dict_tr)
    # transformer_model.eval()

    # root_dir = '../pgm/neutral/'
    # train_files, val_files, test_files = gather_files_pgm(root_dir)
    # train_files = train_files[0:32] # delete this after test
    # val_files = train_files[0:32] # delete this after test

    # Uncomment if using RAVEN dataset
    root_dir = '../RAVEN-10000'
    all_files = gather_files(root_dir)
    num_files = len(all_files)
    train_proportion = 0.7
    val_proportion = 0.15
    # test proportion is 1 - train_proportion - val_proportion
    train_files = all_files[:int(num_files * train_proportion)]
    val_files = all_files[int(num_files * train_proportion):int(num_files * (train_proportion + val_proportion))]
    # test_files = all_files[int(num_files * (train_proportion + val_proportion)):]

    train_files = train_files[0:10]
    val_files = train_files[0:10]

    # train_dataset = RPMSentencesNew(train_files, autoencoder, device=device)
    # val_dataset = RPMSentencesNew(val_files, autoencoder, device=device)

    train_dataset = RPMSentencesRaw(train_files, device=device)
    val_dataset = RPMSentencesRaw(val_files, device=device)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(list(transformer_model.parameters()),
                                 lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(EPOCHS):
        for idx, (inputs, _, targets) in enumerate(train_dataloader):

            if idx%10 == 0:
                start_time = time.time()

            inputs = inputs.to(device)
            # targets_onehot = targets_onehot.to(device)
            targets = targets.to(device)

            outputs = transformer_model(inputs) # (B,8)
            loss = criterion(outputs,targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if idx%10 == 9:
                end_time = time.time()
                batch_time = end_time - start_time
                print(f"10 mini-batches processed in {batch_time} seconds")
                print(f"Most recent batch total loss: {loss.item()}\n")

            # save four times per epoch
            if idx%BATCHES_PER_SAVE == BATCHES_PER_SAVE - 1:
                model_path = f"../modelsaves/v4-itr0/transformer_v4-itr0_ep{epoch + 1}_sv{idx//BATCHES_PER_SAVE+1}.pth"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(transformer_model.state_dict(), model_path)

        # if epoch%10 == 9: # comment out after test
        print(f"Epoch {epoch+1}/{EPOCHS} completed: loss = {loss.item()}\n")

    # Evaluate the model
    proportion_correct = evaluate_model(transformer_model, val_dataloader, device=device)
    print(f"Proportion of answers correct: {proportion_correct}")

    output_file_path = "../tr_results/v4-itr0/proportion_correct_test.txt"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as file:
        file.write(f"Proportion of answers correct: {proportion_correct}.")

if __name__ == "__main__":
    main()