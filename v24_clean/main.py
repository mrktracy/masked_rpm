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
# from datasets import RPMFullSentencesRaw_dataAug as rpm_dataset
from datasets import RPMFullSentencesRaw_base as rpm_dataset
from models import TransformerModelv24, DynamicWeighting, DynamicWeightingRNN
import os
import logging
import math

version = "v24-itr59_full"

logfile = f"../../tr_results/{version}/runlog_{version}_1.txt"
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
    transformer_model = TransformerModelv24(embed_dim=512,
                                            symbol_factor=1,
                                            trans_depth=4,
                                            abs_1_depth=4,
                                            abs_2_depth=4,
                                            trans_num_heads=8,
                                            abs_1_num_heads=8,
                                            abs_2_num_heads=8,
                                            mlp_ratio=4,
                                            use_backbone_enc=True,
                                            decoder_num=2,  # 1 - MLP, 2 - Deconvolution, 3 - Backbone
                                            bb_depth=2,
                                            bb_num_heads=8,
                                            ternary_num=3,  # 1 - C, 2 - Hadamard, 3 - MLP
                                            proj_drop=0.5,
                                            attn_drop=0.5,
                                            drop_path_max=0.5,
                                            per_mlp_drop=0,
                                            ternary_drop=0.3,
                                            ternary_mlp_ratio=3,
                                            restrict_qk=False,
                                            feedback_dim=1024,
                                            meta_1_depth=2,
                                            meta_1_num_heads=8,
                                            meta_1_attn_drop=0.3,
                                            meta_1_proj_drop=0.3,
                                            meta_1_drop_path_max=0.5,
                                            meta_2_depth=2,
                                            meta_2_num_heads=32,
                                            meta_2_attn_drop=0.3,
                                            meta_2_proj_drop=0.3,
                                            meta_2_drop_path_max=0.5,
                                            score_rep=0,
                                            num_loss_terms=3,
                                            device=device
                                            ).to(device)

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
    EPOCHS_PER_SAVE = 4
    VERSION_SUBFOLDER = "" # e.g. "MNIST/" or ""
    BETA = 7.5
    BETA_GROWTH_RATE = 0.05
    L1_perception = 0
    L1_reas = 0
    ALPHA_short = 0.9 # parameter for exponential moving average
    ALPHA_long = 0.5  # parameter for exponential moving average
    WARMUP_EPOCHS = 1
    # WARMUP_IDX = 1500
    THRESHOLD = 0.005
    NU_explore = 15
    NU_exploit = 5

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

    if num_gpus > 1:
        optimizer_2 = torch.optim.Adam(list(transformer_model.module.loss_weight_mlp.parameters()), lr=LEARNING_RATE,
                                                weight_decay=1e-4)
    else:
        optimizer_2 = torch.optim.Adam(list(transformer_model.loss_weight_mlp.parameters()), lr=LEARNING_RATE,
                                   weight_decay=1e-4)

    scheduler_1 = ExponentialLR(optimizer_1, gamma=0.95)
    scheduler_2 = ExponentialLR(optimizer_2, gamma=0.95)

    criterion_1 = nn.CrossEntropyLoss()
    criterion_2 = nn.MSELoss()
    criterion_3 = nn.MSELoss()

    ''' Un-comment below to load saved model '''
    # state_dict = torch.load('../../modelsaves/v24-itr56_full/tf_v24-itr56_full_ep10.pth')
    #
    # transformer_model.load_state_dict(state_dict['transformer_model_state_dict'])
    #
    # optimizer_1.load_state_dict(state_dict['optimizer_1_state_dict'])
    # optimizer_2.load_state_dict(state_dict['optimizer_2_state_dict'])
    #
    # scheduler_1.load_state_dict(state_dict['scheduler_1_state_dict'])
    # scheduler_2.load_state_dict(state_dict['scheduler_2_state_dict'])

    ''' To evaluate model, uncomment this part '''
    # transformer_model.eval()
    #
    # val_loss = evaluation_function(transformer_model, val_dataloader, device)
    # output = f"val: {val_loss:.2f}\n"
    # logging.info(output)

    ''' End load saved model '''

    ema_long = None
    ema_short = 1e-6
    adjustment_factor = 1
    uniform_weights = torch.ones(3).to(device) / 3

    # Training loop
    for epoch in range(FIRST_EPOCH, FIRST_EPOCH + EPOCHS):
        count = 0
        tot_loss = 0
        feedback = None  # reset feedback at start of every epoch

        # logging.info("Initialized loop variables.\n")

        for idx, (sentences, target_nums, _, _) in enumerate(train_dataloader):

            if idx % BATCHES_PER_PRINT == 0:
                start_time = time.time()

            optimizer_1.zero_grad()
            optimizer_2.zero_grad()

            sentences = sentences.to(device) # passed to model to get output and recreation of inputs
            target_nums = target_nums.to(device)  # used to select from among candidates

            if feedback is not None:
                feedback = feedback.to(device)

            # logging.info("Running forward pass of model...\n")

            dist, recreation, embeddings, reas_raw, reas_decoded, reas_meta_reas, loss_weights, feedback = transformer_model(sentences, feedback)

            # if epoch == 0 and idx < WARMUP_IDX:
            if epoch < WARMUP_EPOCHS:
                loss_weights = uniform_weights
            else:
                loss_weights = F.softmax(loss_weights.view(num_gpus, -1).mean(dim=0, keepdim=False), dim=-1)

            task_err = criterion_1(dist, target_nums)
            rec_err = criterion_2(sentences, recreation)
            meta_err = criterion_3(reas_raw, reas_decoded)

            # logging.info("Calculating loss...\n")

            entropy_uniform = -torch.sum(uniform_weights * torch.log(uniform_weights + 1e-9))
            entropy_weights = -torch.sum(loss_weights * torch.log(loss_weights + 1e-9))
            entropy_penalty = torch.exp(entropy_uniform /(entropy_weights + 1e-9)) * (entropy_uniform - entropy_weights)**2

            ent_factor = (1 + adjustment_factor * BETA * entropy_penalty)

            loss = ((loss_weights[0]*task_err + loss_weights[1]*rec_err + loss_weights[2]*meta_err)*ent_factor +
                    L1_perception * torch.norm(embeddings, p=1) + L1_reas * torch.norm(reas_meta_reas, p=1))

            tot_loss += loss.item() # update running averages
            count += 1

            # update exponential moving average of losses
            if epoch == FIRST_EPOCH and idx == 0:  # On the first batch of the first epoch
                ema_long = loss.item()  # Initialize EMA with the first loss
                ema_short = loss.item()
            else:
                ema_long = (1 - ALPHA_long) * ema_long + ALPHA_long * loss.item()
                ema_short = (1 - ALPHA_short) * ema_short + ALPHA_short * loss.item()

            ema_delta = (ema_long - ema_short)/ema_long

            # Adjust BETA based on ema_delta
            if ema_delta < THRESHOLD:  # Recent performance is worse or stalled, increase BETA
                adjustment_factor = 1 / (1 + NU_explore * abs(ema_delta))  # Scale BETA lower, exploration encouraged
            else:  # Recent performance is better, decrease BETA
                adjustment_factor = math.exp(1 + NU_exploit * abs(ema_delta))  # Scale BETA higher, regularization increases

            # logging.info("Forward pass complete.\n")

            loss.backward()

            # logging.info("Backward pass complete.\n")

            optimizer_1.step()
            optimizer_2.step()

            if (idx+1) % BATCHES_PER_PRINT == 0:
                end_time = time.time()
                batch_time = end_time - start_time
                output = f"{BATCHES_PER_PRINT} batches processed in {batch_time:.2f} seconds. Average training loss: {tot_loss/count}"
                logging.info(output)
                logging.info(f"Weights: {loss_weights}, entropy: {entropy_weights}")
                logging.info(f"entropy_penalty: {entropy_penalty}")
                logging.info(f"ent_factor: {ent_factor}")
                logging.info(f"ema_short: {ema_short}, ema_long: {ema_long}")
                logging.info(f"ema_delta: {ema_delta}")
                logging.info(f"adjustment_factor: {adjustment_factor}\n")

            if (idx+1) % batches_per_log == 0:

                # Note: resets feedback to None
                val_loss, _ = evaluation_function(transformer_model, val_dataloader, device, max_batches=150, feedback=None)
                output = f"Epoch {epoch+1} - {idx+1}/{train_length}. Avg loss: {tot_loss/count:.4f}. lr: {scheduler_1.get_last_lr()[0]:.6f}. val: {val_loss:.2f}\n"
                logging.info(output)
                feedback = None

        if (epoch+1) % EPOCHS_PER_SAVE == 0:
            save_file = f"../../modelsaves/{VERSION}/{VERSION_SUBFOLDER}tf_{VERSION}_ep{epoch + 1}.pth"
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                'transformer_model_state_dict': transformer_model.state_dict(),
                # 'dynamic_weights_state_dict': dynamic_weights.state_dict(),
                'optimizer_1_state_dict': optimizer_1.state_dict(),
                'optimizer_2_state_dict': optimizer_2.state_dict(),
                'scheduler_1_state_dict': scheduler_1.state_dict(),
                'scheduler_2_state_dict': scheduler_2.state_dict(),
                'feedback': feedback
            }, save_file)

        scheduler_1.step()
        scheduler_2.step()
        BETA = BETA * (1 + BETA_GROWTH_RATE)

    # To evaluate model, uncomment this part
    transformer_model.eval()

    val_loss, _ = evaluation_function(transformer_model, val_dataloader, device, feedback=None)
    output = f"Final evaluation: {val_loss:.2f}\n"
    logging.info(output)

if __name__ == "__main__":
    main_BERT(version, results_folder)