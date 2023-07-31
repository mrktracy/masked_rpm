import os
import numpy as np
import random
import matplotlib.pyplot as plt

def displayresults():
    filepath = "../results/ae_results/"
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

    fig.show()

if __name__ == "__main__":
    displayresults()
    while plt.get_fignums():
        plt.pause(0.1)