import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from funs import gather_files_pgm, gather_files_by_type
import time
import random
from evaluate_masked import evaluate_model_dist as evaluation_function
from datasets import RPMFullSentencesRaw_base as rpm_dataset
from models import TransformerModelv17, TransformerModelv20, TransformerModelv21, TransformerModelv22
import os
import logging

version = "v22-itr4_full"

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

    # Initialize device, model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    # print(num_gpus)

    # transformer_model = TransformerModelv18(embed_dim=768,
    # transformer_model = TransformerModelv17(embed_dim=512,
    #                                         symbol_factor=1,
    #                                         depth=3,
    #                                         num_heads=16,
    #                                         mlp_ratio = 2,
    #                                         cat_pos=True,
    #                                         cat_output=True,
    #                                         use_backbone=True,
    #                                         backbone_depth=2,
    #                                         bb_num_heads = 8,
    #                                         proj_drop = 0.5,
    #                                         attn_drop = 0.5,
    #                                         mlp_drop = 0.5).to(device)
    #                                         symbol_factor=2,
    #                                         trans_depth=8,
    #                                         abs_1_depth=8,
    #                                         abs_2_depth=4,
    #                                         trans_num_heads=64,
    #                                         abs_1_num_heads=64,
    #                                         abs_2_num_heads=64,
    #                                         cat_pos=True,
    #                                         cat_output=True,
    #                                         use_backbone=True,
    #                                         bb_depth=4,
    #                                         bb_num_heads=32).to(device)
    # transformer_model = TransformerModelv19(embed_dim=768,
    #                                         symbol_factor=1,
    #                                         trans_depth=4,
    #                                         abs_1_depth=4,
    #                                         trans_num_heads=64,
    #                                         abs_1_num_heads=64,
    #                                         use_backbone=True,
    #                                         bb_depth=4,
    #                                         bb_num_heads=32,
    #                                         use_hadamard=False).to(device)
    # transformer_model = TransformerModelv20(embed_dim=512,
    #                                         symbol_factor=1,
    #                                         trans_depth=3,
    #                                         abs_1_depth=3,
    #                                         abs_2_depth=3,
    #                                         trans_num_heads=16,
    #                                         abs_1_num_heads=16,
    #                                         abs_2_num_heads=16,
    #                                         mlp_ratio=4,
    #                                         use_backbone=True,
    #                                         bb_depth=2,
    #                                         bb_num_heads=8,
    #                                         use_hadamard=False,
    #                                         mlp_drop=0.5,
    #                                         proj_drop=0.5,
    #                                         attn_drop=0.5).to(device)
    # transformer_model = TransformerModelv21(embed_dim=768,
    #                                         symbol_factor=1,
    #                                         trans_1_depth=4,
    #                                         trans_2_depth=4,
    #                                         abs_1_depth=4,
    #                                         trans_1_num_heads=64,
    #                                         trans_2_num_heads=64,
    #                                         abs_1_num_heads=64,
    #                                         use_backbone=True,
    #                                         bb_depth=4,
    #                                         bb_num_heads=32,
    #                                         use_hadamard=False).to(device)
    transformer_model = TransformerModelv22(embed_dim=512,
                                            symbol_factor=1,
                                            trans_depth=3,
                                            abs_1_depth=3,
                                            abs_2_depth=3,
                                            trans_num_heads=16,
                                            abs_1_num_heads=16,
                                            abs_2_num_heads=16,
                                            mlp_ratio=4,
                                            use_backbone=True,
                                            bb_depth=2,
                                            bb_num_heads=8,
                                            use_hadamard=False,
                                            mlp_drop=0.5,
                                            proj_drop=0.5,
                                            attn_drop=0.5).to(device)

    # initialize weights
    transformer_model.apply(initialize_weights_he)

    if num_gpus > 1:  # use multiple GPUs
        transformer_model = nn.DataParallel(transformer_model)
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
    root_dir = '../../pgm_data/neutral/'
    # root_dir = '../../i_raven_data_cnst/'
    # root_dir = '../../i_raven_data_full/'
    train_files, val_files, test_files = gather_files_pgm(root_dir)
    # train_files, val_files, test_files = gather_files_by_type(root_dir)

    ''' Transformer model v9 '''
    train_dataset = rpm_dataset(train_files, device=device)
    val_dataset = rpm_dataset(val_files, device=device)

    ''' Define Hyperparameters '''
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 0.00005
    # MOMENTUM = 0.90
    LOGS_PER_EPOCH = 5
    BATCHES_PER_PRINT = 60
    EPOCHS_PER_SAVE = 5
    VERSION_SUBFOLDER = "" # e.g. "MNIST/" or ""
    ALPHA = 0.5 # for relative importance of guess vs. autoencoder accuracy
    L1 = 0

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
    optimizer = torch.optim.Adam(list(transformer_model.parameters()),
                                 lr=LEARNING_RATE,
                                 weight_decay=1e-4)

    scheduler = ExponentialLR(optimizer, gamma=0.95)

    criterion_1 = nn.CrossEntropyLoss()
    criterion_2 = nn.MSELoss()

    # Training loop
    for epoch in range(EPOCHS):
        count = 0
        tot_loss = 0
        times = 0
        for idx, (sentences, target_nums, _, _) in enumerate(train_dataloader):

            if idx % BATCHES_PER_PRINT == 0:
                start_time = time.time()

            batch_size = sentences.size(0)

            sentences = sentences.to(device) # passed to model to get output and recreation of inputs
            target_nums = target_nums.to(device)  # used to select from among candidates

            dist, recreation, embeddings = transformer_model(sentences)

            loss = ALPHA*criterion_1(dist, target_nums) + (1-ALPHA)*criterion_2(sentences, recreation) + \
                L1*torch.norm(embeddings, p=1)

            # loss = criterion_1(dist, target_nums) * criterion_2(sentences, recreation) + \
            #        L1 * torch.norm(embeddings, p=1)

            tot_loss += loss.item() # update running averages
            count += 1

            loss.backward()
            optimizer.step()

            if (idx+1) % BATCHES_PER_PRINT == 0:
                end_time = time.time()
                batch_time = end_time - start_time
                output = f"{BATCHES_PER_PRINT} batches processed in {batch_time:.2f} seconds. Training loss: {tot_loss/count}"
                logging.info(output)

            if (idx+1) % batches_per_log == 0:
                val_loss = evaluation_function(transformer_model, val_dataloader, device, max_batches=150)
                output = f"Epoch {epoch+1} - {idx+1}/{train_length}. loss: {tot_loss/count:.4f}. lr: {scheduler.get_last_lr()[0]:.6f}. val: {val_loss:.2f}\n"
                logging.info(output)
                # with open(logfile, 'a') as file:
                #     file.write(output)

                tot_loss = 0
                count = 0

            optimizer.zero_grad()

        if (epoch+1) % EPOCHS_PER_SAVE == 0:
            save_file = f"../../modelsaves/{VERSION}/{VERSION_SUBFOLDER}tf_{VERSION}_ep{epoch + 1}.pth"
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save(transformer_model.state_dict(), save_file)

        scheduler.step()

if __name__ == "__main__":
    main_BERT(version, results_folder)