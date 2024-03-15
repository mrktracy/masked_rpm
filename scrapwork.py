import numpy as np
import random
import os
import time
import matplotlib
import matplotlib.pyplot as plt
# from main_ae import ResNetAutoencoder, gather_files, gather_files_pgm
# from datasets import RPMSentencesRaw
import torch.nn as nn
# import torch
# from torch.utils.data import DataLoader
import sys
matplotlib.use('TkAgg')
random.seed(time.time())

print(sys.executable)

def calc_mean_std():

    write_file = '../visualize_data/i_raven/mean_std.txt'
    os.makedirs(os.path.dirname(write_file),exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # root_dir = '../pgm/neutral/'
    # train_files, _, _ = gather_files_pgm(root_dir)
    # train_files = train_files[0:32]  # delete this after test

    root_dir = '../i_raven_data/'
    train_files, val_files, test_files = gather_files_pgm(root_dir)

    train_dataset = RPMSentencesRaw(train_files)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    num_batches = len(train_dataloader)

    n = 0
    mean = 0
    M2 = 0

    for idx, (inputs, targets) in enumerate(train_dataloader):

        inputs = inputs.to(device)
        inputs = torch.reshape(inputs, shape=(-1,))

        # Calculate the batch statistics
        batch_mean = inputs.mean().item()
        batch_var = inputs.var().item()
        batch_size = inputs.size(0)

        delta = batch_mean - mean
        mean = mean + delta * batch_size / (n + batch_size)
        M2 = M2 + batch_size * (batch_var + delta * (batch_mean - mean))
        n = n + batch_size

        if idx % 50 == 0:
            print(f"Batch {idx}/{num_batches} complete.")

    if n < 2:
        return float('nan')
    else:
        variance = M2 / (n - 1)

    std = np.sqrt(variance)

    print(f"mean: {mean}")
    print(f"std: {std}")

    with open(write_file, "w") as file:
        file.write(f"mean: {mean}\n")
        file.write(f"std: {std}")
def visualizedata():

    save_dir = "../data/visualize_data/i_raven/"
    os.makedirs(save_dir, exist_ok=True)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # num_gpus = torch.cuda.device_count()
    #
    # # initialize autoencoder
    # autoencoder = ResNetAutoencoder().to(device)
    #
    # if num_gpus > 1:  # use multiple GPUs
    #     autoencoder = nn.DataParallel(autoencoder)
    #
    # state_dict = torch.load('../modelsaves/autoencoder_v1_ep1.pth')
    # autoencoder.load_state_dict(state_dict)
    # autoencoder.eval()

    # root_dir = '../pgm/neutral/'
    # train_files, _, _ = gather_files_pgm(root_dir)
    # train_files = train_files[0:32]  # delete this after test

    root_dir = '../data/i_raven_data/distribute_four'
    train_files, val_files, test_files  = gather_files_pgm(root_dir)
    train_files = train_files[0:32]

    train_dataset = RPMSentencesRaw(train_files)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    solutions = []
    for idx, (inputs, targets) in enumerate(train_dataloader):

        solutions.extend(targets.tolist())
        images = inputs.squeeze(0)

        fig1, axs1 = plt.subplots(3, 3)
        for i in range(3):
            for j in range(3):
                if i==2 & j==2:
                    axs1[i,j].imshow(np.zeros([160,160]), cmap="gray")
                    axs1[i,j].axis('off')
                else:
                    axs1[i,j].imshow(images[i*3+j, :, :, :].squeeze().cpu().detach().numpy(), cmap="gray")
                    axs1[i,j].axis('off')

        fig2, axs2 = plt.subplots(2, 4)
        for i in range(2):
            for j in range(4):
                axs2[i,j].imshow(images[8 + i*4 + j, :, :, :].squeeze().cpu().detach().numpy(), cmap="gray")
                axs2[i,j].axis('off')

        save_con_path = os.path.join(save_dir, f'context_{idx}.png')
        save_can_path = os.path.join(save_dir, f'candidates_{idx}.png')
        fig1.savefig(save_con_path, bbox_inches='tight')
        fig2.savefig(save_can_path, bbox_inches='tight')
        plt.close(fig1)
        plt.close(fig2)

    save_sol_path = os.path.join(save_dir, 'solutions.txt')
    with open(save_sol_path, "w") as file:
        for idx, sol in enumerate(solutions):
            file.write(f"Solution to problem {idx}: {sol}\n")

def displayresults_ae():
    filepath = "../ae_results/ae-v2-itr1/"
    files = os.listdir(filepath)
    random.shuffle(files)

    fig, axs = plt.subplots(5, 2)
    idx = 0
    for file in files[0:5]:
        path = os.path.join(filepath, file)
        data = np.load(path)
        image = data['image'].squeeze()
        output = data['output'].squeeze()
        axs[idx, 0].imshow(image, cmap='gray')
        axs[idx, 1].imshow(output, cmap='gray')
        idx += 1

def displayresults_tr():
    filepath = "../results/tr_results/v2"
    files = os.listdir(filepath)
    random.shuffle(files)

    guesses = []
    fig, axs = plt.subplots(5, 2)
    idx = 0
    for file in files[0:5]:
        path = os.path.join(filepath, file)
        data = np.load(path)
        image = data['guess'].squeeze()
        output = data['target'].squeeze()

        axs[idx, 0].imshow(image, cmap='gray')
        axs[idx, 1].imshow(output, cmap='gray')
        guesses.append(image)

        idx += 1

    print(np.allclose(guesses, guesses[0]*len(guesses)))

def displayresults_tr_grid():
    filepath = "../results/tr_results/v2"
    files = os.listdir(filepath)
    random.shuffle(files)

    # guesses = []
    fig1, axs1 = plt.subplots(3, 3)
    fig2, axs2 = plt.subplots(3, 3)
    fig3, axs3 = plt.subplots(1, 8)
    fig4, axs4 = plt.subplots(1,1)

    file = files[0]

    path = os.path.join(filepath, file)
    data = np.load(path)
    output_grid = data['output_image_grid']
    image_grid = data['imagetensor']
    target = data['target']

    for i in range(3):
        for j in range(3):
            axs1[i, j].imshow(output_grid[i*3 + j,:].squeeze(0), cmap='gray')
            if i==2 and j==2:
                axs2[i, j].imshow(np.zeros([160,160]), cmap='gray')
            else:
                axs2[i, j].imshow(image_grid[i*3 + j, :].squeeze(0), cmap='gray')

    for i in range(8):
        axs3[i].imshow(image_grid[8+i,:].squeeze(0), cmap='gray')

    axs4.imshow(target.squeeze(0), cmap='gray')

def displayresults_tr_grid_masked():
    random.seed(time.time())
    filepath = "../tr_results/v9-itr0/"
    files = os.listdir(filepath)
    npz_files = [file for file in files if file.endswith(".npz")]

    random.shuffle(npz_files)

    # guesses = []
    fig1, axs1 = plt.subplots(3, 3)
    fig2, axs2 = plt.subplots(2, 4)
    fig3, axs3 = plt.subplots(1,1)

    file = npz_files[0]
    print(file)

    path = os.path.join(filepath, file)
    data = np.load(path)
    problem_grid = data['inputs']
    outputs = data['outputs']
    candidates = data['candidates']

    for i in range(3):
        for j in range(3):
            axs1[i, j].imshow(problem_grid[i*3 + j,:].squeeze(), cmap='gray')

    for i in range(2):
        for j in range(4):
            axs2[i, j].imshow(candidates[2*i + j, :].squeeze(), cmap='gray')

    axs3.imshow(outputs.squeeze(), cmap='gray')

def displayresults_BERT():
    random.seed(time.time())
    filepath = "../tr_results/v15-itr4/"
    files = os.listdir(filepath)
    npz_files = [file for file in files if file.endswith(".npz")]

    random.shuffle(npz_files)
    # npz_files.sort(reverse=True)

    # guesses = []
    fig1, axs1 = plt.subplots(3, 3)
    fig2, axs2 = plt.subplots(1, 1)
    fig3, axs3 = plt.subplots(1,1)

    # file = npz_files[0]
    # print(file)
    file = "imgs_ep150_btch0.npz"

    path = os.path.join(filepath, file)
    data = np.load(path)
    problem_grid = data['input']
    output = data['output']
    target = data['target']

    print(np.shape(target))

    for i in range(3):
        for j in range(3):
            axs1[i, j].imshow(problem_grid[i*3 + j,:,:].squeeze(), cmap='gray')

    # for i in range(3):
    #     for j in range(3):
    #         axs2[i, j].imshow(output[i*3 + j,:,:].squeeze(), cmap='gray')
    axs2.imshow(output, cmap='gray')

    # for i in range(3):
    #     for j in range(3):
    #         axs3[i, j].imshow(target[i*3 + j,:,:].squeeze(), cmap='gray')
    axs3.imshow(target, cmap='gray')

if __name__ == "__main__":
    # calc_mean_std()
    # visualizedata()
    # displayresults_ae()
    # displayresults_tr_grid()
    # displayresults_tr_grid_masked()
    displayresults_BERT()

    plt.show()
    while plt.get_fignums():
        plt.pause(0.1)