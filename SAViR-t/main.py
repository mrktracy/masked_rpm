import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from funs import gather_files_pgm, gather_files_by_type
import time
import random
from evaluate import evaluate_model_dist as evaluation_function
from datasets import RPMFullSentencesRaw_dataAug as rpm_dataset
from models import SAViRt, DynamicWeighting, DynamicWeightingRNN
import os
import logging

version = "SAViRt_v0-itr1_full"

logfile = f"../../tr_results/{version}/runlog_{version}.txt"
results_folder = os.path.dirname(logfile)

os.makedirs(results_folder, exist_ok=True)
logging.basicConfig(filename=logfile,level=logging.INFO, filemode='w')

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def initialize_weights_he(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def main_BERT(VERSION, RESULTS_FOLDER):

    MLP_DW = True
    HISTORY_SIZE = 12
    AUTO_REG = False

    if AUTO_REG:
        max_history_length = HISTORY_SIZE*4
    else:
        max_history_length = HISTORY_SIZE*2

    # Initialize device, model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    # print(num_gpus)

    transformer_model = SAViRt(embed_dim=512,
                               grid_dim=5,
                               bb_depth=1,
                               bb_num_heads=4,
                               per_mlp_drop=0).to(device)

    if MLP_DW:
        dynamic_weights = DynamicWeighting(embed_dim=max_history_length,
                                           mlp_ratio=2,
                                           mlp_drop=0.1,
                                           output_dim=2).to(device)
    else:
        if AUTO_REG:
            dynamic_weights = DynamicWeightingRNN(input_dim=4).to(device)
        else:
            dynamic_weights = DynamicWeightingRNN(input_dim=2).to(device)

    # initialize weights
    transformer_model.apply(initialize_weights_he)
    dynamic_weights.apply(initialize_weights_he)

    if num_gpus > 1:  # use multiple GPUs
        transformer_model = nn.DataParallel(transformer_model)
        dynamic_weights = nn.DataParallel(dynamic_weights)
        # transformer_model = nn.DataParallel(transformer_model, device_ids=["cuda:0", "cuda:3"])

    ''' Load saved model '''
    # state_dict_tr = torch.load('../../modelsaves/v22-itr0_full/tf_v22-itr0_full_ep10.pth')
    # transformer_model.load_state_dict(state_dict_tr)
    # transformer_model.eval()

    # if isinstance(transformer_model, nn.DataParallel):
    #     original_model = transformer_model.module
    # else:
    #     original_model = transformer_model

    ''' Use for PGM or I-RAVEN dataset '''
    # root_dir = '../../pgm_data/neutral/'
    # root_dir = '../../i_raven_data_cnst/'
    root_dir = '../../i_raven_data_full/'
    train_files, val_files, test_files = gather_files_pgm(root_dir)
    # train_files, val_files, test_files = gather_files_by_type(root_dir)

    ''' Transformer model v9 '''
    train_dataset = rpm_dataset(train_files, device=device)
    val_dataset = rpm_dataset(val_files, device=device)

    ''' Define Hyperparameters '''
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    # MOMENTUM = 0.90
    LOGS_PER_EPOCH = 5
    BATCHES_PER_PRINT = 60
    EPOCHS_PER_SAVE = 5
    VERSION_SUBFOLDER = ""  # e.g. "MNIST/" or ""
    # ALPHA = 0.5 # for relative importance of guess vs. autoencoder accuracy
    BETA = 2

    ''' Instantiate data loaders, optimizer, criterion '''
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ''' Evaluate model on different types of problems '''
    # record = evaluation_function(transformer_model, val_dataloader, device, max_batches=None)
    # record.to_csv(os.path.join(RESULTS_FOLDER, f"record_{VERSION}.csv"))
    #
    # record_by_type = record.groupby("folder")["correct"].mean()
    # record_by_type.to_csv(os.path.join(RESULTS_FOLDER, f"record_by_type_{VERSION}.csv"))

    ''' Train model '''
    train_length = len(train_dataloader)
    batches_per_log = train_length // LOGS_PER_EPOCH

    # optimizer = torch.optim.SGD(list(transformer_model.parameters()),
    #                              lr=LEARNING_RATE, momentum = MOMENTUM)
    optimizer_1 = torch.optim.Adam(list(transformer_model.parameters()),
                                   lr=LEARNING_RATE,
                                   weight_decay=1e-4)

    optimizer_2 = torch.optim.Adam(list(dynamic_weights.parameters()),
                                   lr=LEARNING_RATE,
                                   weight_decay=1e-4)

    scheduler_1 = ExponentialLR(optimizer_1, gamma=0.95)
    scheduler_2 = ExponentialLR(optimizer_2, gamma=0.95)

    criterion_1 = nn.CrossEntropyLoss()
    criterion_2 = nn.MSELoss()

    if MLP_DW:
        err_history = torch.zeros(max_history_length).to(device)
    else:
        err_history = torch.zeros(HISTORY_SIZE, 4).to(device) if AUTO_REG else torch.zeros(HISTORY_SIZE, 2).to(
            device)

    weights = torch.zeros(2).to(device)

    # Training loop
    for epoch in range(EPOCHS):
        count = 0
        tot_loss = 0
        times = 0
        for idx, (sentences, target_nums, _, _) in enumerate(train_dataloader):

            if idx % BATCHES_PER_PRINT == 0:
                start_time = time.time()

            batch_size = sentences.size(0)

            optimizer_1.zero_grad()
            optimizer_2.zero_grad()

            sentences = sentences.to(device)  # passed to model to get output and recreation of inputs
            target_nums = target_nums.to(device)  # used to select from among candidates

            dist, recreation = transformer_model(sentences)

            task_err = criterion_1(dist, target_nums)
            rec_err = criterion_2(sentences, recreation)

            # logging.info(f"task_err: {task_err.shape}, rec_err: {rec_err.shape}")

            if MLP_DW:
                if AUTO_REG:
                    err_history = torch.cat([err_history[4:], task_err.unsqueeze(0), rec_err.unsqueeze(0), \
                                             weights], dim=-1).detach()

                else:
                    err_history = torch.cat([err_history[2:], task_err.unsqueeze(0), \
                                             rec_err.unsqueeze(0)], dim=-1).detach()

            else:
                if AUTO_REG:
                    # Concatenate the current task error and reconstruction error to the history
                    err_history = torch.cat(
                        [err_history, torch.cat([torch.stack([task_err, rec_err], dim=-1).unsqueeze(0), \
                                                 weights.unsqueeze(0)], dim=-1)], dim=0).detach()

                else:
                    # Concatenate the current task error and reconstruction error to the history
                    err_history = torch.cat([err_history, \
                                             torch.stack([task_err, rec_err], dim=-1).unsqueeze(0)], dim=0).detach()

                # Remove the oldest entry if the history length exceeds the desired length
                if err_history.size(1) > HISTORY_SIZE:
                    err_history = err_history[:, -HISTORY_SIZE:, :]

            # logging.info(f"err_history: {err_history.shape}")

            weights = dynamic_weights(err_history.unsqueeze(0))  # unsqueeze to create "batch" dimension expected

            loss = weights[0] * task_err + weights[1] * rec_err + BETA * torch.var(weights)

            tot_loss += loss.item()  # update running averages
            count += 1

            loss.backward()

            optimizer_1.step()
            optimizer_2.step()

            if (idx + 1) % BATCHES_PER_PRINT == 0:
                end_time = time.time()
                batch_time = end_time - start_time
                output = f"{BATCHES_PER_PRINT} batches processed in {batch_time:.2f} seconds. Training loss: {tot_loss / count}"
                logging.info(output)
                logging.info(f"Weights: {weights}")

            if (idx + 1) % batches_per_log == 0:
                val_loss = evaluation_function(transformer_model, val_dataloader, device, max_batches=150)
                output = f"Epoch {epoch + 1} - {idx + 1}/{train_length}. loss: {tot_loss / count:.4f}. lr: {scheduler_1.get_last_lr()[0]:.6f}. val: {val_loss:.2f}\n"
                logging.info(output)
                # with open(logfile, 'a') as file:
                #     file.write(output)

                tot_loss = 0
                count = 0

        if (epoch + 1) % EPOCHS_PER_SAVE == 0:
            save_file = f"../../modelsaves/{VERSION}/{VERSION_SUBFOLDER}tf_{VERSION}_ep{epoch + 1}.pth"
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                'transformer_model_state_dict': transformer_model.state_dict(),
                'dynamic_weights_state_dict': dynamic_weights.state_dict(),
                'optimizer_1_state_dict': optimizer_1.state_dict(),
                'optimizer_2_state_dict': optimizer_2.state_dict(),
                'scheduler_1_state_dict': scheduler_1.state_dict(),
                'scheduler_2_state_dict': scheduler_2.state_dict()
            }, save_file)

        scheduler_1.step()
        scheduler_2.step()

if __name__ == "__main__":
    main_BERT(version, results_folder)