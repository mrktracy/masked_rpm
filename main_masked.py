import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from main_ae import ResNetAutoencoder, gather_files, gather_files_pgm
import time
import random
from evaluate_masked import evaluate_model_masked, evaluate_model_masked_BERT
from datasets import RPMSentencesSupervised, RPMFullSentences, RPMSentencesSupervisedRaw, RPMFullSentencesRaw
from models import TransformerModelv9, TransformerModelv8, TransformerModelv10
import os
import logging

logfile = "../tr_results/v10-itr0/runlog.txt"

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

def main_GPT():

    # Initialize device, model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    # print(num_gpus)

    transformer_model = TransformerModelv8(depth=20, num_heads=32).to(device)

    # initialize weights
    transformer_model.apply(initialize_weights_he)

    # initialize autoencoder
    autoencoder = ResNetAutoencoder(embed_dim=768).to(device)

    if num_gpus > 1:  # use multiple GPUs
        transformer_model = nn.DataParallel(transformer_model)
        # transformer_model = nn.DataParallel(transformer_model, device_ids=["cuda:0", "cuda:3"])
        autoencoder = nn.DataParallel(autoencoder) # uncomment if using PGM

    # load autoencoder state dict
    state_dict = torch.load('../modelsaves/ae-v2-itr0/ae-v2-itr0_ep10.pth') # for I-RAVEN
    # state_dict = torch.load('../modelsaves/autoencoder_v1_ep1.pth') # for PGM
    # state_dict = torch.load('../modelsaves/autoencoder_v0.pth') # for RAVEN
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()

    ''' Load saved model '''
    # state_dict_tr = torch.load('../modelsaves/v8-itr10/tf_v8-itr10_ep10.pth')
    # transformer_model.load_state_dict(state_dict_tr)
    # transformer_model.eval()

    ''' Use for PGM or I-RAVEN dataset '''
    # root_dir = '../pgm/neutral/'
    root_dir = '../i_raven_data/'
    train_files, val_files, test_files = gather_files_pgm(root_dir)

    ''' Use RAVEN dataset '''
    # root_dir = '../RAVEN-10000'
    # all_files = gather_files(root_dir)
    # num_files = len(all_files)
    # train_proportion = 0.7
    # val_proportion = 0.15
    # # test proportion is 1 - train_proportion - val_proportion
    # train_files = all_files[:int(num_files * train_proportion)]
    # val_files = all_files[int(num_files * train_proportion):int(num_files * (train_proportion + val_proportion))]
    # # test_files = all_files[int(num_files * (train_proportion + val_proportion)):]

    ''' Use MNIST dataset '''
    # train_proportion = 0.85
    # val_proportion = 0.15
    # mnist_data = MNIST(root='../MNIST/', train=True, download=True, \
    #                    transform=transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()]))
    # mnist_len = len(mnist_data)
    # train_len = int(mnist_len*train_proportion)
    # val_len = int(mnist_len*val_proportion)
    #
    # mnist_train, mnist_val = random_split(mnist_data, [train_len, val_len])

    ''' Transformer model v8 '''
    # train_dataset = RPMSentencesViT_Masked(train_files, \
    #                                 ViT_model_name="google/vit-base-patch16-224-in21k", \
    #                                 device = device, num_gpus = num_gpus)
    # val_dataset = RPMFullSentencesViT_Masked(val_files, \
    #                               ViT_model_name="google/vit-base-patch16-224-in21k", \
    #                               device = device, num_gpus = num_gpus)

    train_dataset = RPMSentencesAE_Masked(train_files, \
                                           autoencoder = autoencoder, \
                                           device=device, num_gpus=num_gpus, inv=False)
    val_dataset = RPMFullSentencesAE_Masked(val_files, \
                                             autoencoder = autoencoder, \
                                             device=device, num_gpus=num_gpus, inv=False)

    ''' MNIST transformer model '''
    # train_dataset = CustomMNIST(mnist_train, num_samples=100000)
    # val_dataset = CustomMNIST(mnist_val, num_samples=10000)

    ''' Define Hyperparameters '''
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    MOMENTUM = 0.90
    LOGS_PER_EPOCH = 20
    BATCHES_PER_PRINT = 100
    EPOCHS_PER_SAVE = 1
    VERSION = "v8-itr12"
    VERSION_SUBFOLDER = "" # e.g. "MNIST/" or ""

    ''' Instantiate data loaders, optimizer, criterion '''
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_length = len(train_dataloader)
    batches_per_log = train_length // LOGS_PER_EPOCH

    # optimizer = torch.optim.SGD(list(transformer_model.parameters()),
    #                              lr=LEARNING_RATE, momentum = MOMENTUM)
    optimizer = torch.optim.Adam(list(transformer_model.parameters()), lr=LEARNING_RATE)

    scheduler = ExponentialLR(optimizer, gamma=0.98)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(EPOCHS):
        count = 0
        tot_loss = 0
        times = 0
        for idx, (inputs, first_patch, targets) in enumerate(train_dataloader):

            if idx % BATCHES_PER_PRINT == 0:
                start_time = time.time()

            inputs = inputs.to(device)
            first_patch = first_patch.to(device)
            targets = targets.to(device)

            outputs = transformer_model(inputs, first_patch) # (B,embed_dim)
            loss = criterion(outputs,targets)

            tot_loss += loss.item() # update running averages
            count += 1

            loss.backward()
            optimizer.step()

            if (idx+1) % BATCHES_PER_PRINT == 0:
                end_time = time.time()
                batch_time = end_time - start_time
                print(f"{BATCHES_PER_PRINT} batches processed in {batch_time:.2f} seconds. Training loss: {tot_loss/count}")

            if (idx+1) % batches_per_log == 0:
                val_loss = evaluate_model_masked(transformer_model, val_dataloader, device, max_batches=150)
                output = f"Epoch {epoch+1} - {idx+1}/{train_length}. loss: {tot_loss/count:.4f}. lr: {scheduler.get_last_lr()[0]:.6f}. val: {val_loss:.2f}\n"
                print(output)
                # logging.info(output)
                with open(logfile, 'a') as file:
                    file.write(output)

                tot_loss = 0
                count = 0

                if times%5 == 0:

                    gradfile = f"../tr_results/{VERSION}/grads_ep{epoch+1}_sv{times//5}.txt"

                    # Inspect gradients
                    for name, param in transformer_model.named_parameters():
                        if param.grad is not None:
                            with open(gradfile, 'a') as file:
                                file.write(f"Gradient for {name}: {param.grad}\n")
                        else:
                            with open(logfile, 'a') as file:
                                file.write(f"No gradient for {name}\n")
                    times += 1

            optimizer.zero_grad()

        if (epoch+1) % EPOCHS_PER_SAVE == 0:
            save_file = f"../modelsaves/{VERSION}/{VERSION_SUBFOLDER}tf_{VERSION}_ep{epoch + 1}.pth"
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save(transformer_model.state_dict(), save_file)

        scheduler.step()

    def save_to_npz(inputs, outputs, candidates, idx, VERSION, VERSION_SUBFOLDER, inv=False):

        if inv:
            input_images = np.array([autoencoder.module.decode_inv(input.unsqueeze(0)).cpu().detach().numpy() for input in inputs])
            output_images = autoencoder.module.decode_inv(outputs.unsqueeze(0)).cpu().detach().numpy()
            candidate_images = np.array([autoencoder.module.decode_inv(candidate.unsqueeze(0)).cpu().detach().numpy() for candidate in candidates])
        else:
            input_images = np.array([autoencoder.module.decode(input.unsqueeze(0)).cpu().detach().numpy() for input in inputs])
            output_images = autoencoder.module.decode(outputs.unsqueeze(0)).cpu().detach().numpy()
            candidate_images = np.array([autoencoder.module.decode(candidate.unsqueeze(0)).cpu().detach().numpy() for candidate in candidates])

        # Save to npz file
        np.savez_compressed(f"../tr_results/{VERSION}/{VERSION_SUBFOLDER}imgs_{idx}.npz",
                            inputs=input_images,
                            outputs=output_images,
                            candidates=candidate_images)

    # Iterate over the dataset
    for idx, (inputs, candidates, targets) in enumerate(val_dataloader):
        if (idx+1) % 22 == 0:  # Check if the idx is a multiple of 22
            print(f"Processing index: {idx}")

            # move images to the device
            inputs = inputs.to(device)  # shape (B,9,model_dim)
            candidates = candidates.to(device)  # shape (B, 8, embed_dim)
            targets = targets.to(device)  # shape (B,)

            transformer_model.eval()
            with torch.no_grad():  # Disable gradient computation for inference
                # Perform a forward pass to get the outputs
                outputs = transformer_model(inputs)

                batch_indices = torch.arange(candidates.size(0), device=candidates.device)
                selected_candidates = candidates[batch_indices, targets, :]
                inputs[:,8,:] = selected_candidates

                img_inputs = inputs[0,:,:].squeeze()
                img_outputs = outputs[0, :].squeeze()
                img_candidates = candidates[0, :, :].squeeze()

                # Convert the tensors to images and save them
                save_to_npz(img_inputs, img_outputs, img_candidates, (idx+1)//22, VERSION, VERSION_SUBFOLDER, inv=False)

    print("Finished processing all items.")

def main_BERT():

    # Initialize device, model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    # print(num_gpus)

    transformer_model = TransformerModelv10(depth=10, num_heads=32, cat=True).to(device)

    # initialize weights
    transformer_model.apply(initialize_weights_he)

    # initialize autoencoder
    # autoencoder = ResNetAutoencoder(embed_dim=768).to(device)

    if num_gpus > 1:  # use multiple GPUs
        transformer_model = nn.DataParallel(transformer_model)
        # transformer_model = nn.DataParallel(transformer_model, device_ids=["cuda:0", "cuda:3"])
        # autoencoder = nn.DataParallel(autoencoder) # uncomment if using PGM

    # load autoencoder state dict
    # state_dict = torch.load('../modelsaves/ae-v2-itr0/ae-v2-itr0_ep10.pth') # for I-RAVEN
    # state_dict = torch.load('../modelsaves/autoencoder_v1_ep1.pth') # for PGM
    # state_dict = torch.load('../modelsaves/autoencoder_v0.pth') # for RAVEN
    # autoencoder.load_state_dict(state_dict)
    # autoencoder.eval()

    ''' Load saved model '''
    # state_dict_tr = torch.load('../modelsaves/v9-itr0/tf_v9-itr0_ep200.pth')
    # transformer_model.load_state_dict(state_dict_tr)
    # transformer_model.eval()

    ''' Use for PGM or I-RAVEN dataset '''
    # root_dir = '../pgm/neutral/'
    root_dir = '../i_raven_data/'
    train_files, val_files, test_files = gather_files_pgm(root_dir)
    # train_files = train_files[:100]
    # val_files = val_files[:100]

    ''' Use RAVEN dataset '''
    # root_dir = '../RAVEN-10000'
    # all_files = gather_files(root_dir)
    # num_files = len(all_files)
    # train_proportion = 0.7
    # val_proportion = 0.15
    # # test proportion is 1 - train_proportion - val_proportion
    # train_files = all_files[:int(num_files * train_proportion)]
    # val_files = all_files[int(num_files * train_proportion):int(num_files * (train_proportion + val_proportion))]
    # # test_files = all_files[int(num_files * (train_proportion + val_proportion)):]

    ''' Transformer model v9 '''
    train_dataset = RPMSentencesSupervisedRaw(train_files, \
                                           embed_dim=768, \
                                           device=device)
    # create dataset for printing results of problems in training set
    # train_print_dataset = RPMFullSentencesRaw(train_files, \
    #                                        embed_dim=768, \
    #                                        device=device)
    val_dataset = RPMFullSentencesRaw(val_files, \
                                            embed_dim=768, \
                                            device=device)

    ''' Define Hyperparameters '''
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    MOMENTUM = 0.90
    LOGS_PER_EPOCH = 50
    BATCHES_PER_PRINT = 50
    EPOCHS_PER_SAVE = 1
    VERSION = "v10-itr0"
    VERSION_SUBFOLDER = "" # e.g. "MNIST/" or ""

    ''' Instantiate data loaders, optimizer, criterion '''
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # train_print_dataloader = DataLoader(train_print_dataset, batch_size=BATCH_SIZE, shuffle=True) # for saving images
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_length = len(train_dataloader)
    batches_per_log = train_length // LOGS_PER_EPOCH

    # optimizer = torch.optim.SGD(list(transformer_model.parameters()),
    #                              lr=LEARNING_RATE, momentum = MOMENTUM)
    optimizer = torch.optim.Adam(list(transformer_model.parameters()), lr=LEARNING_RATE)

    scheduler = ExponentialLR(optimizer, gamma=0.98)
    criterion = nn.MSELoss(reduction='sum')

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
            # print(outputs.size(), targets.size())
            loss = criterion(outputs,targets)

            tot_loss += loss.item() # update running averages
            count += 1

            loss.backward()
            optimizer.step()

            if (idx+1) % BATCHES_PER_PRINT == 0:
                end_time = time.time()
                batch_time = end_time - start_time
                print(f"{BATCHES_PER_PRINT} batches processed in {batch_time:.2f} seconds. Training loss: {tot_loss/count}")

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

                    gradfile = f"../tr_results/{VERSION}/grads_ep{epoch+1}_sv{times//5}.txt"

                    # Inspect gradients
                    for name, param in transformer_model.named_parameters():
                        if param.grad is not None:
                            with open(gradfile, 'a') as file:
                                file.write(f"Gradient for {name}: {param.grad}\n")
                        else:
                            with open(logfile, 'a') as file:
                                file.write(f"No gradient for {name}\n")
                    times += 1

            optimizer.zero_grad()

        if (epoch+1) % EPOCHS_PER_SAVE == 0:
            save_file = f"../modelsaves/{VERSION}/{VERSION_SUBFOLDER}tf_{VERSION}_ep{epoch + 1}.pth"
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save(transformer_model.state_dict(), save_file)

        scheduler.step()

    # def save_to_npz(inputs, outputs, candidates, idx, VERSION, VERSION_SUBFOLDER, inv=False):
    #
    #     if inv:
    #         input_images = np.array([autoencoder.module.decode_inv(input.unsqueeze(0)).cpu().detach().numpy() for input in inputs])
    #         output_images = autoencoder.module.decode_inv(outputs.unsqueeze(0)).cpu().detach().numpy()
    #         candidate_images = np.array([autoencoder.module.decode_inv(candidate.unsqueeze(0)).cpu().detach().numpy() for candidate in candidates])
    #     else:
    #         input_images = np.array([autoencoder.module.decode(input.unsqueeze(0)).cpu().detach().numpy() for input in inputs])
    #         output_images = autoencoder.module.decode(outputs.unsqueeze(0)).cpu().detach().numpy()
    #         candidate_images = np.array([autoencoder.module.decode(candidate.unsqueeze(0)).cpu().detach().numpy() for candidate in candidates])
    #
    #     # Save to npz file
    #     np.savez_compressed(f"../tr_results/{VERSION}/{VERSION_SUBFOLDER}imgs_{idx}.npz",
    #                         inputs=input_images,
    #                         outputs=output_images,
    #                         candidates=candidate_images)
    #
    # # Iterate over the dataset
    # for idx, (inputs, targets, _, target_nums, embeddings, mask_tensors) in enumerate(train_print_dataloader):
    #     # if (idx+1) % 22 == 0:  # Check if the idx is a multiple of 22
    #     if True:  # Check if the idx is a multiple of 22
    #         print(f"Processing index: {idx}")
    #
    #         # move images to the device
    #         inputs = inputs.to(device)  # shape (B,9,model_dim)
    #         candidates = embeddings[:,8:,:].to(device)  # shape (B, 8, embed_dim)
    #         targets = target_nums.to(device)  # shape (B,)
    #
    #         transformer_model.eval()
    #         with torch.no_grad():  # Disable gradient computation for inference
    #             # Perform a forward pass to get the outputs
    #             outputs = transformer_model(inputs, mask_tensors) # get embedding of guess
    #             indices = list(range(inputs.size(0))) + []
    #             inputs[:,8,:] = targets.unsqueeze(1) # fill in the answer in the grid
    #
    #             # sample just one example per batch
    #             img_inputs = inputs[0,:,:].squeeze()
    #             img_outputs = outputs[0, :].squeeze()
    #             img_candidates = candidates[0, :, :].squeeze()
    #
    #             # Convert the tensors to images and save them
    #             # save_to_npz(img_inputs, img_outputs, img_candidates, (idx+1)//22, VERSION, VERSION_SUBFOLDER, inv=False)
    #             save_to_npz(img_inputs, img_outputs, img_candidates, idx, VERSION, VERSION_SUBFOLDER, inv=False)
    #
    # print("Finished processing all items.")

if __name__ == "__main__":
    # main_GPT()
    main_BERT()