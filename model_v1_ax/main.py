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
from datasets import RPMFullSentencesRaw_base as rpm_dataset
from funs import gather_files_pgm
from models import ReasoningModule

# Logging configuration
version = "Model_v1_itr0"
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
    """
    He initialization for Linear layers.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def save_validation_loss(val_loss, results_folder):
    """
    Save validation loss to a file for external use.
    """
    val_loss_file = os.path.join(results_folder, "val_loss.txt")
    with open(val_loss_file, "w") as f:
        f.write(f"{val_loss:.6f}")


def main(version, results_folder, model_class, model_params, hyperparams=None):
    """
    Main training and evaluation loop.

    Args:
        version (str): Version string for saving logs and models.
        results_folder (str): Path to the folder where results will be saved.
        model_class (type): The class of the model to instantiate.
        model_params (dict): A dictionary of parameters to pass to the model's constructor.
        hyperparams (dict, optional): Training hyperparameters. Defaults are used if None.
    """
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    # Initialize model
    model = model_class(**model_params).to(device)
    model.apply(initialize_weights_he)

    if num_gpus > 1:
        model = nn.DataParallel(model)

    ''' Dataset setup '''
    root_dir = '../../i_raven_data_full/'
    train_files, val_files, test_files = gather_files_pgm(root_dir)

    train_dataset = rpm_dataset(train_files, device=device)
    val_dataset = rpm_dataset(val_files, device=device)
    test_dataset = rpm_dataset(test_files, device=device)

    ''' Hyperparameters '''
    defaults = {
        "epochs": 25,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "alpha": 0.5,  # Task vs. reconstruction loss balance
        "logs_per_epoch": 15,
        "batches_per_print": 30,
        "epochs_per_save": 5,
    }
    hyperparams = hyperparams or defaults
    EPOCHS = hyperparams["epochs"]
    BATCH_SIZE = hyperparams["batch_size"]
    LEARNING_RATE = hyperparams["learning_rate"]
    ALPHA = hyperparams["alpha"]

    ''' Data loaders, optimizer, and criteria '''
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    criterion_task = nn.CrossEntropyLoss()
    criterion_reconstruction = nn.MSELoss()

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss, count = 0, 0

        for idx, (sentences, target_nums, _, _) in enumerate(train_dataloader):
            if idx % hyperparams["batches_per_print"] == 0:
                start_time = time.time()

            optimizer.zero_grad()

            sentences = sentences.to(device)  # Shape: [batch_size, num_candidates, 9, 1, 160, 160]
            target_nums = target_nums.to(device)  # Shape: [batch_size]

            # Forward pass
            reconstructed_sentences, scores = model(sentences)

            # Loss computation
            rec_err = criterion_reconstruction(sentences, reconstructed_sentences)
            task_err = criterion_task(scores, target_nums)
            loss = ALPHA * task_err + (1 - ALPHA) * rec_err

            total_loss += loss.item()
            count += 1

            loss.backward()
            optimizer.step()

            if (idx + 1) % hyperparams["batches_per_print"] == 0:
                elapsed_time = time.time() - start_time
                avg_loss = total_loss / count
                logging.info(f"{hyperparams['batches_per_print']} batches in {elapsed_time:.2f}s. Avg loss: {avg_loss:.4f}")

        # Validation
        val_loss, _ = evaluation_function(model, val_dataloader, device, max_batches=150)
        logging.info(f"Epoch {epoch + 1}/{EPOCHS}. Avg loss: {total_loss / count:.4f}. Val loss: {val_loss:.2f}")
        save_validation_loss(val_loss, results_folder)

        if (epoch + 1) % hyperparams["epochs_per_save"] == 0:
            save_path = f"../../modelsaves/{version}/{model_class.__name__}_{version}_ep{epoch + 1}.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path)

        scheduler.step()

    # Final evaluation
    val_loss, _ = evaluation_function(model, val_dataloader, device)
    logging.info(f"Final evaluation: {val_loss:.2f}")
    save_validation_loss(val_loss, results_folder)


if __name__ == "__main__":
    MODEL_CLASS = ReasoningModule
    MODEL_PARAMS = {
        "embed_dim": 512,
        "grid_size": 3,
        "abs_depth": 4,
        "trans_depth": 6,
        "ternary_depth": 3,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "proj_drop": 0,
        "attn_drop": 0,
        "drop_path_max": 0,
        "num_symbols_abs": 9,
        "num_symbols_ternary": 6,
        "norm_layer": nn.LayerNorm,
        "bb_proj_drop": 0,
        "bb_attn_drop": 0,
        "bb_drop_path_max": 0,
        "bb_mlp_drop": 0,
    }

    HYPERPARAMS = {
        "epochs": 25,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "alpha": 0.5,
        "logs_per_epoch": 15,
        "batches_per_print": 30,
        "epochs_per_save": 5,
    }

    main(version, results_folder, MODEL_CLASS, MODEL_PARAMS, HYPERPARAMS)
