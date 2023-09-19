import numpy as np
import torch

def evaluate_model(model, dataloader, device, max_batches = None):

    model.eval()
    with torch.no_grad():

        num_correct = 0
        num_samples = 0
        for idx, (inputs, targets) in enumerate(dataloader):

            # move images to the device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # forward pass
            outputs = model(inputs) # (batch_size,8)
            guesses = torch.argmax(outputs, dim=1)
            num_correct += torch.eq(guesses, targets).sum().item()
            num_samples += inputs.size(0)

            if max_batches is not None and idx == max_batches:
                break

    return 100*(num_correct / num_samples)