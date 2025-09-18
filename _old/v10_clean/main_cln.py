import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import time
import random
from evaluate_cln import evaluate_model_masked_BERT
from datasets_cln import RPMSentencesSupervisedRaw, RPMFullSentencesRaw
from models_cln import TransformerModelv10
import os
import re
# import logging # commented out because logging was giving errors

logfile = "../../tr_results/v10-itr15/runlog.txt"

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

def gather_files_pgm(root_dir):
    all_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.npz'):
                all_files.append(os.path.join(dirpath, filename))
    random.shuffle(all_files)

    train_pattern = "train"
    val_pattern = "val"
    test_pattern = "test"

    train_files = [filename for filename in all_files if re.search(train_pattern, filename)]
    val_files = [filename for filename in all_files if re.search(val_pattern, filename)]
    test_files = [filename for filename in all_files if re.search(test_pattern, filename)]

    return train_files, val_files, test_files

def main_BERT():

    # Initialize device, model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    transformer_model = TransformerModelv10(depth=20, num_heads=64, cat=False).to(device)

    # initialize weights
    transformer_model.apply(initialize_weights_he)

    if num_gpus > 1:  # use multiple GPUs
        transformer_model = nn.DataParallel(transformer_model)
        # transformer_model = nn.DataParallel(transformer_model, device_ids=["cuda:0", "cuda:3"])

    ''' Load saved model '''
    # state_dict_tr = torch.load('../../modelsaves/v9-itr0/tf_v9-itr0_ep200.pth')
    # transformer_model.load_state_dict(state_dict_tr)
    # transformer_model.eval()

    ''' Use for PGM or I-RAVEN dataset '''
    # root_dir = '../pgm/neutral/'
    root_dir = '../../i_raven_data_cnst/'
    train_files, val_files, test_files = gather_files_pgm(root_dir)
    # train_files = train_files[:5]
    # val_files = val_files[:5]

    ''' Transformer model v10 '''
    train_dataset = RPMSentencesSupervisedRaw(train_files, \
                                           embed_dim=768, \
                                           device=device)
    val_dataset = RPMFullSentencesRaw(val_files, \
                                            embed_dim=768, \
                                            device=device)

    ''' Define Hyperparameters '''
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    # MOMENTUM = 0.90
    LOGS_PER_EPOCH = 1
    BATCHES_PER_PRINT = 50
    EPOCHS_PER_SAVE = 5
    VERSION = "v10-itr15"
    VERSION_SUBFOLDER = "" # e.g. "MNIST/" or ""

    ''' Instantiate data loaders, optimizer, criterion '''
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    train_length = len(train_dataloader)
    batches_per_log = train_length // LOGS_PER_EPOCH

    # optimizer = torch.optim.SGD(list(transformer_model.parameters()),
    #                              lr=LEARNING_RATE, momentum = MOMENTUM)
    optimizer = torch.optim.Adam(list(transformer_model.parameters()), lr=LEARNING_RATE)

    scheduler = ExponentialLR(optimizer, gamma=0.995)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(EPOCHS):
        count = 0
        tot_loss = 0
        times = 0
        for idx, (inputs, targets, mask_tensors) in enumerate(train_dataloader):

            if idx % BATCHES_PER_PRINT == 0:
                start_time = time.time()

            inputs = inputs.to(device)
            targets = targets.to(device)
            mask_tensors = mask_tensors.to(device)

            outputs = transformer_model(inputs, mask_tensors) # (B,1,160,160)
            loss = criterion(outputs,targets)

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
                val_loss = evaluate_model_masked_BERT(transformer_model, val_dataloader, device, max_batches=150)
                output = f"Epoch {epoch+1} - {idx+1}/{train_length}. loss: {tot_loss/count:.4f}. lr: {scheduler.get_last_lr()[0]:.6f}. val: {val_loss:.2f}\n"

                print(output)

                # logging.info(output)
                with open(logfile, 'a') as file:
                    file.write(output)

                tot_loss = 0
                count = 0

                if times%5 == 0:

                    gradfile = f"../../tr_results/{VERSION}/grads_ep{epoch+1}_sv{times//5}.txt"

                    # Inspect gradients
                    for name, param in transformer_model.named_parameters():
                        if param.grad is not None:
                            with open(gradfile, 'a') as file:
                                file.write(f"Gradient for {name}: {param.grad}\n")
                        else:
                            with open(logfile, 'a') as file:
                                file.write(f"No gradient for {name}\n")

                    np.savez_compressed(f"../../tr_results/{VERSION}/{VERSION_SUBFOLDER}imgs_ep{epoch + 1}_btch{idx}.npz",
                                        input=np.array(inputs[0, :, :, :, :].squeeze().cpu()),
                                        output=np.array(outputs[0, :, :, :].squeeze().detach().cpu()),
                                        target=np.array(targets[0, :, :, :].squeeze().cpu()))
                    times += 1

            optimizer.zero_grad()

        if (epoch+1) % EPOCHS_PER_SAVE == 0:
            save_file = f"../../modelsaves/{VERSION}/{VERSION_SUBFOLDER}tf_{VERSION}_ep{epoch + 1}.pth"
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save(transformer_model.state_dict(), save_file)

        scheduler.step()

if __name__ == "__main__":
    main_BERT()