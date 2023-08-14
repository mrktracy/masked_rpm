import numpy as np
import torch

def evaluate_model(model, dataloader, device):

    model.eval()
    with torch.no_grad():

        num_correct = 0
        for _, (inputs, _, targets) in enumerate(dataloader):

            # move images to the device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # forward pass
            outputs = model(inputs) # (batch_size,8)
            guesses = torch.argmax(outputs, dim=1)
            num_correct += torch.eq(guesses, targets).sum().item()

    return num_correct / len(dataloader.dataset)