import numpy as np
import random
import os
import time
import matplotlib
import matplotlib.pyplot as plt
# import torch.nn as nn
# import torch
# from torch.utils.data import DataLoader
matplotlib.use('TkAgg')
random.seed(time.time())

def displayresults_BERT():
    random.seed(time.time())
    filepath = "../../tr_results/v11-itr0/"
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
    file = "imgs_ep41_btch5.npz"

    path = os.path.join(filepath, file)
    data = np.load(path)
    problem_grid = data['input']
    output = data['output']
    target = data['target']

    for i in range(3):
        for j in range(3):
            axs1[i, j].imshow(problem_grid[i*3 + j,:,:].squeeze(), cmap='gray')

    axs2.imshow(target.squeeze(), cmap='gray')
    axs3.imshow(output.squeeze(), cmap='gray')

if __name__ == "__main__":
    displayresults_BERT()

    plt.show()
    while plt.get_fignums():
        plt.pause(0.1)