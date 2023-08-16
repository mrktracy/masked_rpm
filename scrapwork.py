import numpy as np
import random
import os
import matplotlib.pyplot as plt

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
    displayresults_ae()
    # displayresults_tr_grid()
    plt.show()
    while plt.get_fignums():
        plt.pause(0.1)