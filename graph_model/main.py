import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from funs import gather_files_pgm, gather_files_by_type
import time
import random
from evaluate_masked import evaluate_model_dist as evaluation_function
from datasets import RPMFullSentencesRaw_dataAug as rpm_dataset
# from datasets import RPMFullSentencesRaw_base as rpm_dataset
from models import AsymmetricGraphModel
import os
import logging
import math

version = "AGM_v0_itr0"

logfile = f"../../tr_results/{version}/runlog_{version}.txt"
results_folder = os.path.dirname(logfile)

os.makedirs(results_folder, exist_ok=True)
logging.basicConfig(filename=logfile,level=logging.INFO, filemode='w')

logging.info("Begin log.\n")

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

    # Initialize device, model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    transformer_model = AsymmetricGraphModel(
        embed_dim=512,
        grid_size=3,
        num_candidates=8,
        n_msg_passing_steps=5,
        bb_depth=2,
        bb_num_heads=8,
        neuron_dim=1024,
        n_neurons=10).to(device)

    # initialize weights
    transformer_model.apply(initialize_weights_he)

    if num_gpus > 1:  # use multiple GPUs
        transformer_model = nn.DataParallel(transformer_model)
        # transformer_model = nn.DataParallel(transformer_model, device_ids=["cuda:0", "cuda:3"])

    # logging.info("Models declared and initialized.\n")

    ''' Use for PGM or I-RAVEN dataset '''
    # root_dir = '../../pgm_data/neutral/'
    # root_dir = '../../pgm_data/extrapolation/'
    # root_dir = '../../i_raven_data_cnst/'
    root_dir = '../../i_raven_data_full/'
    train_files, val_files, test_files = gather_files_pgm(root_dir)
    # train_files, val_files, test_files = gather_files_by_type(root_dir)

    ''' Transformer model v9 '''
    train_dataset = rpm_dataset(train_files, device=device)
    val_dataset = rpm_dataset(val_files, device=device)
    test_dataset = rpm_dataset(test_files, device=device)

    ''' Define Hyperparameters '''
    EPOCHS = 30
    FIRST_EPOCH = 0
    BATCH_SIZE = 32
    LEARNING_RATE = 0.00005
    # MOMENTUM = 0.90
    LOGS_PER_EPOCH = 15
    BATCHES_PER_PRINT = 40
    EPOCHS_PER_SAVE = 5
    VERSION_SUBFOLDER = "" # e.g. "MNIST/" or ""
    ALPHA = 0.5

    ''' Instantiate data loaders, optimizer, criterion '''
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # logging.info("Data loaded.\n")

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

    scheduler_1 = ExponentialLR(optimizer_1, gamma=0.95)

    criterion_1 = nn.CrossEntropyLoss()
    criterion_2 = nn.MSELoss()

    # Training loop
    for epoch in range(FIRST_EPOCH, FIRST_EPOCH + EPOCHS):
        count = 0
        tot_loss = 0

        for idx, (sentences, target_nums, _, _) in enumerate(train_dataloader):

            if idx % BATCHES_PER_PRINT == 0:
                start_time = time.time()

            optimizer_1.zero_grad()

            sentences = sentences.to(device) # passed to model to get output and recreation of inputs
            target_nums = target_nums.to(device)  # used to select from among candidates

            embeddings, recreation, dist = transformer_model(sentences)

            task_err = criterion_1(dist, target_nums)
            rec_err = criterion_2(sentences, recreation)

           # logging.info("Calculating loss...\n")

            loss = ALPHA*task_err + (1 - ALPHA)*rec_err

            tot_loss += loss.item() # update running averages
            count += 1

            loss.backward()

            optimizer_1.step()

            if (idx+1) % BATCHES_PER_PRINT == 0:
                end_time = time.time()
                batch_time = end_time - start_time
                output = f"{BATCHES_PER_PRINT} batches processed in {batch_time:.2f} seconds. Average training loss: {tot_loss/count}"
                logging.info(output)

            if (idx+1) % batches_per_log == 0:

                # Note: resets feedback to None
                val_loss, _ = evaluation_function(transformer_model, val_dataloader, device, max_batches=150)
                output = f"Epoch {epoch+1} - {idx+1}/{train_length}. Avg loss: {tot_loss/count:.4f}. lr: {scheduler_1.get_last_lr()[0]:.6f}. val: {val_loss:.2f}\n"
                logging.info(output)

        if (epoch+1) % EPOCHS_PER_SAVE == 0:
            save_file = f"../../modelsaves/{VERSION}/{VERSION_SUBFOLDER}agm_{VERSION}_ep{epoch + 1}.pth"
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                'model_state_dict': transformer_model.state_dict(),
                'optimizer_1_state_dict': optimizer_1.state_dict(),
                'scheduler_1_state_dict': scheduler_1.state_dict()
            }, save_file)

        scheduler_1.step()

    # To evaluate model, uncomment this part
    transformer_model.eval()

    val_loss, _ = evaluation_function(transformer_model, val_dataloader, device, feedback=None)
    output = f"Final evaluation: {val_loss:.2f}\n"
    logging.info(output)

if __name__ == "__main__":
    main_BERT(version, results_folder)