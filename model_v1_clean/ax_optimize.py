import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from ax.service.ax_client import AxClient
from ax.service.utils.report_utils import exp_to_df
from evaluate_masked import evaluate_model_dist as evaluation_function
from datasets import RPMFullSentencesRaw_base as rpm_dataset
from funs import gather_files_pgm
from models import ReasoningModule


def initialize_weights_he(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def train_and_evaluate(parameterization, epochs=5):
    """
    Train and evaluate the model for a given number of epochs with specified hyperparameters.
    Args:
        parameterization (dict): Hyperparameters for training.
        epochs (int): Number of epochs to train in each trial.
    Returns:
        float: Validation loss after the last epoch (to be minimized by Ax).
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = '../../i_raven_data_full/'
    train_files, val_files, test_files = gather_files_pgm(root_dir)

    train_dataset = rpm_dataset(train_files, device=device)
    val_dataset = rpm_dataset(val_files, device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=int(parameterization["batch_size"]), shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=int(parameterization["batch_size"]), shuffle=False)

    # Model initialization
    model_params = {
        "embed_dim": 512,
        "grid_size": 3,
        "abs_depth": int(parameterization["depth"]),
        "trans_depth": int(parameterization["depth"]),
        "ternary_depth": int(parameterization["depth"]),
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "proj_drop": parameterization["proj_drop"],
        "attn_drop": parameterization["attn_drop"],
        "drop_path_max": parameterization["drop_path_max"],
        "bb_proj_drop": parameterization["bb_proj_drop"],
        "bb_attn_drop": parameterization["bb_attn_drop"],
        "bb_drop_path_max": parameterization["bb_drop_path_max"],
        "bb_mlp_drop": parameterization["bb_mlp_drop"],
    }

    model = ReasoningModule(**model_params).to(device)
    model.apply(initialize_weights_he)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=parameterization["learning_rate"], weight_decay=1e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    criterion_task = nn.CrossEntropyLoss()
    criterion_reconstruction = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        for sentences, target_nums, _, _ in train_dataloader:
            optimizer.zero_grad()
            sentences = sentences.to(device)
            target_nums = target_nums.to(device)

            # Forward pass
            reconstructed_sentences, scores = model(sentences)

            # Compute loss
            rec_err = criterion_reconstruction(sentences, reconstructed_sentences)
            task_err = criterion_task(scores, target_nums)
            loss = parameterization["alpha"] * task_err + (1 - parameterization["alpha"]) * rec_err
            loss.backward()
            optimizer.step()

        scheduler.step()

    # Validation
    model.eval()
    val_loss, _ = evaluation_function(model, val_dataloader, device)
    return val_loss


def run_optimization():
    """
    Run Ax optimization loop.
    """
    ax_client = AxClient()
    ax_client.create_experiment(
        name="reasoning_module_optimization",
        parameters=[
            {"name": "batch_size", "type": "choice", "values": [16, 32, 64]},
            {"name": "learning_rate", "type": "range", "bounds": [1e-5, 1e-3], "log_scale": True},
            {"name": "alpha", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "proj_drop", "type": "range", "bounds": [0.0, 0.5]},
            {"name": "attn_drop", "type": "range", "bounds": [0.0, 0.5]},
            {"name": "drop_path_max", "type": "range", "bounds": [0.0, 0.5]},
            {"name": "bb_proj_drop", "type": "range", "bounds": [0.0, 0.5]},
            {"name": "bb_attn_drop", "type": "range", "bounds": [0.0, 0.5]},
            {"name": "bb_drop_path_max", "type": "range", "bounds": [0.0, 0.5]},
            {"name": "bb_mlp_drop", "type": "range", "bounds": [0.0, 0.5]},
            {"name": "depth", "type": "choice", "values": [2, 4, 6, 8]},  # Added depth parameter
        ],
        objective_name="val_loss",
        minimize=True,
    )

    for _ in range(20):  # Run 20 trials
        parameters, trial_index = ax_client.get_next_trial()
        val_loss = train_and_evaluate(parameters, epochs=5)  # Adjusted to 5 epochs
        ax_client.complete_trial(trial_index=trial_index, raw_data=val_loss)

    # Save the results
    experiment = ax_client.experiment
    results_df = exp_to_df(experiment)
    results_df.to_csv(f"../../tr_results/reasoning_module_optimization_results.csv", index=False)


if __name__ == "__main__":
    run_optimization()
