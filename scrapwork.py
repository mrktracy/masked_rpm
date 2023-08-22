import numpy as np
import random
import os
import matplotlib.pyplot as plt
from main_ae import ResNetAutoencoder, gather_files, gather_files_pgm
from datasets import RPMSentencesNew
import torch.nn as nn
import torch

def visualizedata():

    save_dir = "../visualize_data/"
    os.makedirs(save_dir, exists_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    # initialize autoencoder
    autoencoder = ResNetAutoencoder().to(device)

    if num_gpus > 1:  # use multiple GPUs
        autoencoder = nn.DataParallel(autoencoder)

    state_dict = torch.load('../modelsaves/autoencoder_v1_ep1.pth')
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()

    root_dir = '../pgm/neutral/'
    train_files, _, _ = gather_files_pgm(root_dir)
    train_files = train_files[0:32]  # delete this after test

    train_dataset = RPMSentencesNew(train_files, autoencoder, device=device)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    for idx, (inputs, _, targets) in enumerate(train_dataloader):
        images = autoencoder.decode(inputs.permute(1,0,2,3))
        fig, axs = plt.subplots(4, 4)
        for i in range(4):
            for j in range(4):
                axs[i,j].imshow(images[i*4+j].squeeze().cpu().detach().numpy(), cmap="gray")
                axs.axis('off')

        save_path = os.path.join(save_dir, f'image_{idx}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

def displayresults_ae():
    filepath = "../results/ae_results/v1/"
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

if __name__ == "__main__":
    visualizedata()
    # displayresults_ae()
    # displayresults_tr_grid()
    # plt.show()
    # while plt.get_fignums():
    #     plt.pause(0.1)