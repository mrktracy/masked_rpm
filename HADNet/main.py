import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from funs import gather_files_pgm
import time
import random
from evaluate_masked import evaluate_model_dist as evaluation_function
from datasets import RPMFullSentencesRaw_base as rpm_dataset
from models import HADNet
import os
import logging

# Versioning
version = "HADNet_v0_itr0"
logfile = f"../../tr_results/{version}/runlog_{version}.txt"
results_folder = os.path.dirname(logfile)

os.makedirs(results_folder, exist_ok=True)
logging.basicConfig(filename=logfile, level=logging.INFO, filemode='w')

logging.info("Begin log.\n")

# Set seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def initialize_weights_he(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def main_HADNet(VERSION, RESULTS_FOLDER):

    # Initialize device, model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    hadnet_model = HADNet(
        embed_dim=512,
        grid_size=3,
        num_candidates=8,
        n_levels=3,
        bb_depth=2,
        bb_num_heads=8
    ).to(device)

    # Initialize weights
    hadnet_model.apply(initialize_weights_he)

    if num_gpus > 1:  # Use multiple GPUs
        hadnet_model = nn.DataParallel(hadnet_model)

    ''' Dataset setup '''
    root_dir = '../../i_raven_data_full/'
    train_files, val_files, test_files = gather_files_pgm(root_dir)

    train_dataset = rpm_dataset(train_files, device=device)
    val_dataset = rpm_dataset(val_files, device=device)
    test_dataset = rpm_dataset(test_files, device=device)

    ''' Hyperparameters '''
    EPOCHS = 100
    FIRST_EPOCH = 0
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    LOGS_PER_EPOCH = 15
    BATCHES_PER_PRINT = 30
    EPOCHS_PER_SAVE = 20
    ALPHA = 0.5  # Balancing factor between task and reconstruction losses

    ''' Data loaders, optimizer, criterion '''
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(hadnet_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    criterion_task = nn.CrossEntropyLoss()
    criterion_reconstruction = nn.MSELoss()

    # Training loop
    for epoch in range(FIRST_EPOCH, FIRST_EPOCH + EPOCHS):
        count = 0
        tot_loss = 0

        for idx, (sentences, target_nums, _, _) in enumerate(train_dataloader):

            if idx % BATCHES_PER_PRINT == 0:
                start_time = time.time()

            optimizer.zero_grad()

            sentences = sentences.to(device)
            target_nums = target_nums.to(device)

            embeddings, recreation, scores = hadnet_model(sentences)

            task_err = criterion_task(scores, target_nums)
            rec_err = criterion_reconstruction(embeddings, recreation)

            loss = ALPHA * task_err + (1 - ALPHA) * rec_err

            tot_loss += loss.item()
            count += 1

            loss.backward()
            optimizer.step()

            if (idx + 1) % BATCHES_PER_PRINT == 0:
                end_time = time.time()
                batch_time = end_time - start_time
                output = f"{BATCHES_PER_PRINT} batches processed in {batch_time:.2f} seconds. Average training loss: {tot_loss / count}"
                logging.info(output)

            if (idx + 1) % (len(train_dataloader) // LOGS_PER_EPOCH) == 0:
                val_loss, _ = evaluation_function(hadnet_model, val_dataloader, device, max_batches=150)
                output = f"Epoch {epoch + 1} - {idx + 1}/{len(train_dataloader)}. Avg loss: {tot_loss / count:.4f}. lr: {scheduler.get_last_lr()[0]:.6f}. val: {val_loss:.2f}\n"
                logging.info(output)

        if (epoch + 1) % EPOCHS_PER_SAVE == 0:
            save_file = f"../../modelsaves/{VERSION}/hadnet_{VERSION}_ep{epoch + 1}.pth"
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                'model_state_dict': hadnet_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, save_file)

        scheduler.step()

    # Final evaluation
    hadnet_model.eval()
    val_loss, _ = evaluation_function(hadnet_model, val_dataloader, device)
    output = f"Final evaluation: {val_loss:.2f}\n"
    logging.info(output)

if __name__ == "__main__":
    main_HADNet(version, results_folder)
