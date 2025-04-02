import os
import logging
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.report_utils import exp_to_df
from evaluate_masked import evaluate_model_dist as evaluation_function
from datasets import RPMFullSentencesRaw_base as rpm_dataset
from funs import gather_files_pgm
from models_alt import ReasoningModule
import datetime
import random
import numpy as np


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def initialize_weights_he(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def train_and_evaluate(parameterization, epochs=1, use_max_batches=False, max_batches=3000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # root_dir = '../../i_raven_data_full/'
    root_dir = '../../pgm_data/neutral/'
    train_files, val_files, test_files = gather_files_pgm(root_dir)

    train_dataset = rpm_dataset(train_files, device=device)
    val_dataset = rpm_dataset(val_files, device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model_params = {
        "embed_dim": int(parameterization["embed_dim"]),
        "grid_size": 3,
        # "abs_depth": int(parameterization["abs_depth"]),
        # "trans_depth": int(parameterization["trans_depth"]),
        "ternary_depth": int(parameterization["ternary_depth"]),
        # "abs_num_heads": int(parameterization["abs_num_heads"]),
        # "trans_num_heads": int(parameterization["trans_num_heads"]),
        "tern_num_heads": int(parameterization["tern_num_heads"]),
        # "abs_mlp_ratio": int(parameterization["abs_mlp_ratio"]),
        # "trans_mlp_ratio": int(parameterization["trans_mlp_ratio"]),
        "tern_mlp_ratio": 4,
        "phi_mlp_hidden_dim": int(parameterization["phi_mlp_hidden_dim"]),
        # "abs_proj_drop": parameterization["abs_proj_drop"],
        # "trans_proj_drop": parameterization["trans_proj_drop"],
        "tern_proj_drop": parameterization["tern_proj_drop"],
        # "abs_attn_drop": parameterization["abs_attn_drop"],
        # "trans_attn_drop": parameterization["trans_attn_drop"],
        "tern_attn_drop": parameterization["tern_attn_drop"],
        # "abs_drop_path_max": parameterization["abs_drop_path_max"],
        # "trans_drop_path_max": parameterization["trans_drop_path_max"],
        "tern_drop_path_max": parameterization["tern_drop_path_max"],
        "symbol_factor_tern": parameterization["symbol_factor_tern"],
        "bb_depth": int(parameterization["bb_depth"]),
        "bb_num_heads": int(parameterization["bb_num_heads"]),
        "bb_mlp_ratio": int(parameterization["bb_mlp_ratio"]),
        "bb_proj_drop": parameterization["bb_proj_drop"],
        "bb_attn_drop": parameterization["bb_attn_drop"],
        "bb_drop_path_max": parameterization["bb_drop_path_max"],
        "bb_mlp_drop": parameterization["bb_mlp_drop"],
        "decoder_mlp_drop": parameterization["decoder_mlp_drop"],
        "use_bb_pos_enc": parameterization["use_bb_pos_enc"]
    }

    model = ReasoningModule(**model_params).to(device)
    model.apply(initialize_weights_he)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=parameterization["learning_rate"], weight_decay=1e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    criterion_task = nn.CrossEntropyLoss()
    criterion_reconstruction = nn.MSELoss()

    batch_count = 0
    for epoch in range(epochs):
        model.train()
        for sentences, target_nums, _, _ in train_dataloader:
            if use_max_batches and batch_count >= max_batches:
                break

            optimizer.zero_grad()
            sentences = sentences.to(device)
            target_nums = target_nums.to(device)

            reconstructed_sentences, scores = model(sentences)
            rec_err = criterion_reconstruction(sentences, reconstructed_sentences)
            task_err = criterion_task(scores, target_nums)
            loss = parameterization["alpha"] * task_err + (1 - parameterization["alpha"]) * rec_err
            loss.backward()
            optimizer.step()

            if use_max_batches:
                batch_count += 1

        if use_max_batches and batch_count >= max_batches:
            break

        scheduler.step()

    model.eval()
    val_acc, _ = evaluation_function(model, val_dataloader, device)
    return val_acc


def run_optimization(version):
    results_dir = f"../../tr_results/{version}"
    os.makedirs(results_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        filename=f"{results_dir}/{version}_ax_log.txt",
                        filemode="w")

    ax_client = AxClient()
    ax_client.create_experiment(
        name=f"reasoning_module_optimization_{version}",
        parameters=[
            {"name": "alpha", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "embed_dim", "type": "choice", "values": [256, 512, 768]},
            {"name": "learning_rate", "type": "range", "bounds": [5e-5, 3e-4], "log_scale": True},

            # Reasoning module parameters
            # {"name": "abs_depth", "type": "choice", "values": [1, 2, 3, 4]},
            # {"name": "trans_depth", "type": "choice", "values": [1, 2, 3, 4]},
            {"name": "ternary_depth", "type": "choice", "values": [1, 2, 3, 4, 5, 6]},
            # {"name": "abs_num_heads", "type": "choice", "values": [2, 4, 8, 16]},
            # {"name": "trans_num_heads", "type": "choice", "values": [2, 4, 8, 16]},
            {"name": "tern_num_heads", "type": "choice", "values": [2, 4, 8, 16, 32]},
            # {"name": "abs_proj_drop", "type": "range", "bounds": [0.0, 0.5]},
            # {"name": "trans_proj_drop", "type": "range", "bounds": [0.0, 0.5]},
            {"name": "tern_proj_drop", "type": "range", "bounds": [0.0, 0.5]},
            # {"name": "abs_attn_drop", "type": "range", "bounds": [0.0, 0.5]},
            # {"name": "trans_attn_drop", "type": "range", "bounds": [0.0, 0.5]},
            {"name": "tern_attn_drop", "type": "range", "bounds": [0.0, 0.5]},
            # {"name": "abs_drop_path_max", "type": "range", "bounds": [0.0, 0.5]},
            # {"name": "trans_drop_path_max", "type": "range", "bounds": [0.0, 0.5]},
            {"name": "tern_drop_path_max", "type": "range", "bounds": [0.0, 0.5]},
            # {"name": "abs_mlp_ratio", "type": "choice", "values": [2, 4, 6]},
            # {"name": "trans_mlp_ratio", "type": "choice", "values": [2, 4, 6]},
            {"name": "phi_mlp_hidden_dim", "type": "choice", "values": [2, 4, 6]},
            {"name": "symbol_factor_tern", "type": "choice", "values": [1, 2, 3]},

            # Backbone parameters
            {"name": "bb_depth", "type": "choice", "values": [1, 2, 3, 4]},
            {"name": "bb_num_heads", "type": "choice", "values": [2, 4, 8, 16]},
            {"name": "bb_mlp_ratio", "type": "choice", "values": [2, 4, 6]},
            {"name": "bb_proj_drop", "type": "range", "bounds": [0.0, 0.5]},
            {"name": "bb_attn_drop", "type": "range", "bounds": [0.0, 0.5]},
            {"name": "bb_drop_path_max", "type": "range", "bounds": [0.0, 0.5]},
            {"name": "bb_mlp_drop", "type": "range", "bounds": [0.0, 0.5]},
            {"name": "decoder_mlp_drop", "type": "range", "bounds": [0.0, 0.5]},
            {"name": "use_bb_pos_enc", "type": "choice", "values": [True, False]}
        ],
        objectives={"val_acc": ObjectiveProperties(minimize=False)},
    )

    results_path = f"../../tr_results/{version}/ax_results.csv"
    total_trials = 120
    trial_index = 0

    for trial in range(total_trials):
        start_time = datetime.datetime.now()
        logging.info(f"Starting trial {trial + 1} of {total_trials} at {start_time}...")
        try:
            # Get parameters for the trial
            parameters, trial_index = ax_client.get_next_trial()
            logging.info(f"Trial {trial + 1} parameters: {parameters}")  # Log the parameters being tried

            # Train and evaluate the model
            val_acc = train_and_evaluate(parameters, epochs=1, use_max_batches=True, max_batches=2500)
            logging.info(f"Trial {trial + 1} validation accuracy: {val_acc}")  # Log the validation accuracy

            # Mark the trial as complete in Ax
            ax_client.complete_trial(trial_index=trial_index, raw_data=val_acc)
            logging.info(f"Trial {trial + 1} completed successfully.")

        except Exception as e:
            ax_client.log_trial_failure(trial_index=trial_index)
            logging.error(f"Trial {trial + 1} failed: {e}")

        # Save Ax state after each trial
        try:
            ax_client.save_to_json_file(filepath=f"{results_dir}/ax_state.json")
        except Exception as save_err:
            logging.error(f"Failed to save Ax state: {save_err}")

        end_time = datetime.datetime.now()
        duration = end_time - start_time
        logging.info(f"Trial {trial + 1} duration: {duration}\n")

    # save best parameters
    best_parameters, metrics = ax_client.get_best_parameters()
    best_val_acc = metrics.get("val_acc", {}).get("value", None)
    logging.info(f"Best Trial - Parameters: {best_parameters}, Validation Accuracy: {best_val_acc}")

    # Final save of results
    experiment = ax_client.experiment
    results_df = exp_to_df(experiment)
    results_df.to_csv(results_path, index=False)


if __name__ == "__main__":
    version = "Model_v1_itr24"
    set_seed()
    run_optimization(version)