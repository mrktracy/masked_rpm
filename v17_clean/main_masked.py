import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from funs import gather_files_pgm
import time
import random
from evaluate_masked import evaluate_model_masked_BERT_dist as evaluation_function
from datasets import RPMFullSentencesRaw_v2
from models import TransformerModelv17
import os
import logging

logfile = "../../tr_results/v17-itr0/runlog.txt"

os.makedirs(os.path.dirname(logfile), exist_ok=True)
# logging.basicConfig(filename=logfile,level=logging.INFO, filemode='w')
# logging.info("Test initializing logger.")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def initialize_weights_he(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def main_BERT():

    # Initialize device, model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    # print(num_gpus)

    transformer_model = TransformerModelv17(symbol_factor=1, depth=5, num_heads=64, cat_pos=True, \
                                            cat_output=True, use_backbone=True).to(device)

    # initialize weights
    transformer_model.apply(initialize_weights_he)

    if num_gpus > 1:  # use multiple GPUs
        transformer_model = nn.DataParallel(transformer_model)
        # transformer_model = nn.DataParallel(transformer_model, device_ids=["cuda:0", "cuda:3"])

    ''' Load saved model '''
    # state_dict_tr = torch.load('../../modelsaves/v16-itr2/tf_v16-itr2_ep15.pth')
    # transformer_model.load_state_dict(state_dict_tr)
    # transformer_model.eval()

    if isinstance(transformer_model, nn.DataParallel):
        original_model = transformer_model.module
    else:
        original_model = transformer_model

    ''' Use for PGM or I-RAVEN dataset '''
    # root_dir = '../pgm/neutral/'
    root_dir = '../../i_raven_data_cnst/'
    # root_dir = '../../i_raven_data_full/'
    train_files, val_files, test_files = gather_files_pgm(root_dir)
    # train_files = train_files[:5]
    # val_files = val_files[:5]

    ''' Transformer model v9 '''
    train_dataset = RPMFullSentencesRaw_v2(train_files, \
                                           device=device)
    val_dataset = RPMFullSentencesRaw_v2(val_files, \
                                            device=device)

    ''' Define Hyperparameters '''
    EPOCHS = 15
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    # MOMENTUM = 0.90
    LOGS_PER_EPOCH = 10
    BATCHES_PER_PRINT = 20
    EPOCHS_PER_SAVE = 15
    VERSION = "v17-itr0"
    VERSION_SUBFOLDER = "" # e.g. "MNIST/" or ""
    ALPHA = 0.75 # for relative importance of guess vs. autoencoder accuracy

    ''' Instantiate data loaders, optimizer, criterion '''
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    train_length = len(train_dataloader)
    batches_per_log = train_length // LOGS_PER_EPOCH

    # optimizer = torch.optim.SGD(list(transformer_model.parameters()),
    #                              lr=LEARNING_RATE, momentum = MOMENTUM)
    optimizer = torch.optim.Adam(list(transformer_model.parameters()), lr=LEARNING_RATE)

    scheduler = ExponentialLR(optimizer, gamma=0.99)

    criterion_1 = nn.CrossEntropyLoss()
    criterion_2 = nn.MSELoss()

    # Training loop
    for epoch in range(EPOCHS):
        count = 0
        tot_loss = 0
        times = 0
        for idx, (sentences, target_nums) in enumerate(train_dataloader):

            if idx % BATCHES_PER_PRINT == 0:
                start_time = time.time()

            batch_size = sentences.size(0)

            sentences = sentences.to(device) # passed to model to get output and recreation of inputs
            target_nums = target_nums.to(device)  # used to select from among candidates

            dist, recreation = transformer_model(sentences)

            loss = ALPHA*criterion_1(dist, target_nums) + (1-ALPHA)*criterion_2(sentences, recreation)
            # loss = ALPHA * criterion_1(dists, target_nums) + (1 - ALPHA) * criterion_2(inputs, recreation)

            tot_loss += loss.item() # update running averages
            count += 1

            loss.backward()
            optimizer.step()

            if (idx+1) % BATCHES_PER_PRINT == 0:
                end_time = time.time()
                batch_time = end_time - start_time
                print(f"{BATCHES_PER_PRINT} batches processed in {batch_time:.2f} seconds. Training loss: {tot_loss/count}")
                # print(f"Output all zeros: {torch.equal(outputs, torch.zeros_like(outputs))}")

            if (idx+1) % batches_per_log == 0:
                val_loss = evaluation_function(transformer_model, val_dataloader, device, max_batches=150)
                output = f"Epoch {epoch+1} - {idx+1}/{train_length}. loss: {tot_loss/count:.4f}. lr: {scheduler.get_last_lr()[0]:.6f}. val: {val_loss:.2f}\n"
                # output = f"Epoch {epoch + 1} - {idx + 1}/{train_length}. loss: {tot_loss / count:.4f}."
                print(output)
                # logging.info(output)
                with open(logfile, 'a') as file:
                    file.write(output)

                tot_loss = 0
                count = 0

            optimizer.zero_grad()

        if (epoch+1) % EPOCHS_PER_SAVE == 0:
            save_file = f"../../modelsaves/{VERSION}/{VERSION_SUBFOLDER}tf_{VERSION}_ep{epoch + 1}.pth"
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save(transformer_model.state_dict(), save_file)

        scheduler.step()

if __name__ == "__main__":
    main_BERT()