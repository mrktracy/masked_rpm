## Use transformer output directly, without subsequent MLP layers
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from main_ae import ResNetAutoencoder, gather_files, gather_files_pgm
import time
import random
from evaluate import evaluate_model
from datasets import RPMSentences, RPMFullSentences
from models import TransformerModel

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def main():
    # Define Hyperparameters
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    # VERSION = 'v2'

    # Initialize device, data loader, model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    root_dir = '../pgm/neutral/'
    train_files, val_files, test_files = gather_files_pgm(root_dir)

    # # Uncomment if using RAVEN dataset
    # root_dir = '../RAVEN-10000'
    # all_files = gather_files(root_dir)
    # num_files = len(all_files)
    # train_proportion = 0.7
    # val_proportion = 0.15
    # # test proportion is 1 - train_proportion - val_proportion
    # train_files = all_files[:int(num_files * train_proportion)]
    # val_files = all_files[int(num_files * train_proportion):int(num_files * (train_proportion + val_proportion))]
    # # test_files = all_files[int(num_files * (train_proportion + val_proportion)):]

    # initialize autoencoder
    autoencoder = ResNetAutoencoder().to(device)
    state_dict = torch.load('../modelsaves/autoencoder_v0.pth')
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()

    train_dataset = RPMSentences(train_files, autoencoder, embed_dim=256, device=device)
    val_dataset = RPMFullSentences(val_files, autoencoder, embed_dim=256, device=device)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    transformer_model = TransformerModel(depth=12).to(device) # instantiate model

    # for name,param in transformer_model.named_parameters(): # initialize model
    #     if 'weight' in name:
    #         torch.nn.init.xavier_normal_(param)

    if num_gpus > 1: # use multiple GPUs
        transformer_model = nn.DataParallel(transformer_model)

    # # comment out this block if training
    # state_dict_tr = torch.load('../modelsaves/transformer_v2_ep14.pth')
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
            mask_tensors = mask_tensors.to(device)

            outputs = transformer_model.forward(inputs) # (B,9,512)
            guesses = (outputs * mask_tensors).sum(dim=1) # (B,512)
            guesses_cut = guesses[:,0:256] # (B, 1, 256)

            loss = criterion(guesses_cut,targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if idx%100 == 99:
                end_time = time.time()
                batch_time = end_time - start_time
                print(f"100 mini-batches processed in {batch_time} seconds")
                print(f"Most recent batch total loss: {loss.item()}\n")

        torch.save(transformer_model.state_dict(), f"../modelsaves/transformer_v4_ep{epoch+1}.pth")
        print(f"Epoch {epoch+1}/{EPOCHS} completed: loss = {loss.item()}\n")

    # Evaluate the model
    proportion_correct = evaluate_model(transformer_model, val_dataloader, autoencoder, save_path='../tr_results/v2/', device=device)
    print(f"Proportion of answers correct: {proportion_correct}")

    output_file_path = "../tr_results/v2/proportion_correct.txt"
    with open(output_file_path, "w") as file:
        file.write(f"Proportion of answers correct: {proportion_correct}.")

if __name__ == "__main__":
    main()