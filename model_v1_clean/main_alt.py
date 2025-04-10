import os
import time
import logging
import random
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from evaluate_masked import evaluate_model_dist as evaluation_function
# from datasets import RPMFullSentencesRaw_dataAug_wRowSwap as rpm_dataset
from datasets import RPMFullSentencesRaw_dataAug as rpm_dataset
# from datasets import RPMFullSentencesRaw_dataAug_noOuterRot as rpm_dataset
# from datasets import RPMFullSentencesRaw_dataAug_noOuterRot_wRowSwap as rpm_dataset
# from datasets import RPMFullSentencesRaw_base as rpm_dataset
from funs import gather_files_pgm
from models_alt import ReasoningModule

# Versioning
version = "Model_v1_itr28_pgmNeut"
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


def main(version, results_folder, model_class, model_params):
    """
    Main training and evaluation loop, agnostic to the model being used.

    Args:
        version (str): Version string for saving logs and models.
        results_folder (str): Path to the folder where results will be saved.
        model_class (type): The class of the model to instantiate.
        model_params (dict): A dictionary of parameters to pass to the model's constructor.
    """
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    # Initialize the model
    model = model_class(**model_params).to(device)

    # Initialize weights
    model.apply(initialize_weights_he)

    # Handle multiple GPUs
    if num_gpus > 1:
        model = nn.DataParallel(model)

    ''' Dataset setup '''
    # root_dir = '../../i_raven_data_full/'
    # root_dir = '../../pgm_data/extrapolation/'
    root_dir = '../../pgm_data/neutral/'
    train_files, val_files, test_files = gather_files_pgm(root_dir)

    train_dataset = rpm_dataset(train_files, device=device)
    val_dataset = rpm_dataset(val_files, device=device)
    test_dataset = rpm_dataset(test_files, device=device)

    ''' Hyperparameters '''
    EPOCHS = 10
    FIRST_EPOCH = 0
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    # LEARNING_RATE = 1e-4
    LOGS_PER_EPOCH = 120
    BATCHES_PER_PRINT = 20
    EPOCHS_PER_SAVE = 1
    ALPHA = 0.05  # Balancing factor between task and reconstruction losses
    ALPHA_GROWTH_RATE = 0

    ''' Data loaders, optimizer, criterion '''
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    criterion_task = nn.CrossEntropyLoss()
    criterion_reconstruction = nn.MSELoss()

    # Training loop
    for epoch in range(FIRST_EPOCH, FIRST_EPOCH + EPOCHS):

        count = 0
        tot_loss = 0
        model.train()

        for idx, (sentences, target_nums, _, _) in enumerate(train_dataloader):

            if idx % BATCHES_PER_PRINT == 0:
                start_time = time.time()

            optimizer.zero_grad()

            sentences = sentences.to(device)  # Shape: [batch_size, num_candidates, 9, 1, 160, 160]
            target_nums = target_nums.to(device)  # Shape: [batch_size]

            # Forward pass
            recreation, scores = model(sentences)

            # Calculate reconstruction error
            rec_err = criterion_reconstruction(sentences, recreation)
            task_err = criterion_task(scores, target_nums)  # target_nums shape: [batch_size]

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
                val_loss, _ = evaluation_function(model, val_dataloader, device, max_batches=150)
                output = f"Epoch {epoch + 1} - {idx + 1}/{len(train_dataloader)}. Avg loss: {tot_loss / count:.4f}. lr: {scheduler.get_last_lr()[0]:.6f}. val: {val_loss:.2f}. ALPHA :{ALPHA:.2f}\n"
                logging.info(output)

        if (epoch + 1) % EPOCHS_PER_SAVE == 0:
            save_file = f"../../modelsaves/{version}/{model_class.__name__}_{version}_ep{epoch + 1}.pth"
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, save_file)

        ALPHA = min(ALPHA*(1+ALPHA_GROWTH_RATE), 1)
        scheduler.step()

    # Final evaluation
    model.eval()
    val_loss, _ = evaluation_function(model, test_dataloader, device)
    output = f"Final evaluation: {val_loss:.2f}\n"
    logging.info(output)


if __name__ == "__main__":
    MODEL_CLASS = ReasoningModule
    MODEL_PARAMS = {
        "embed_dim": 768,
        "grid_size": 3,
        # "abs_depth": 2,
        # "trans_depth": 2,
        "ternary_depth": 1,
        # "abs_num_heads": 8,
        # "trans_num_heads": 8,
        "tern_num_heads": 32,
        # "abs_mlp_ratio": 4.0,
        # "trans_mlp_ratio": 4.0,
        "tern_mlp_ratio": 4.0,
        # "abs_proj_drop": 0,
        # "trans_proj_drop": 0,
        "tern_proj_drop": 0.5,
        # "abs_attn_drop": 0.3,
        # "trans_attn_drop": 0.3,
        "tern_attn_drop": 0.5,
        # "abs_drop_path_max": 0.3,
        # "trans_drop_path_max": 0.3,
        "tern_drop_path_max": 0.5,
        # "num_symbols_abs": 9,
        "num_symbols_ternary": 6,
        "norm_layer": nn.LayerNorm,
        "phi_mlp_hidden_dim": 4,
        "bb_depth": 1,
        "bb_num_heads": 4,
        "bb_mlp_ratio": 4,
        "bb_proj_drop": 0,
        "bb_attn_drop": 0,
        "bb_drop_path_max": 0,
        "bb_mlp_drop": 0,
        "decoder_mlp_drop": 0.5,
        # "symbol_factor_abs": 1,
        "symbol_factor_tern": 1,
        "use_bb_pos_enc": True
    }

    main(version, results_folder, MODEL_CLASS, MODEL_PARAMS)