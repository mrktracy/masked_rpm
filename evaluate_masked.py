import torch

def evaluate_model_masked(model, dataloader, device, max_batches = None):

    model.eval()
    with torch.no_grad():

        num_correct = 0
        num_samples = 0
        for idx, (inputs, candidates, targets) in enumerate(dataloader):

            # batch_size = inputs.shape[0]

            # move images to the device
            inputs = inputs.to(device) # shape (B,9,model_dim)
            candidates = candidates.to(device) # shape (B, 8, embed_dim)
            targets = targets.to(device) # shape (B,)


            # forward pass
            outputs = model(inputs) # (batch_size,embed_dim)

            print(f"{inputs.shape=}")
            print(f"{targets.shape=}")
            print(f"{candidates.shape=}")
            print(f"{outputs.shape=}")

            guesses = torch.argmin(torch.sum((candidates - outputs)**2, dim=-1), dim = -1)
            num_correct += torch.eq(guesses, targets).sum().item()
            num_samples += inputs.size(0)

            if max_batches is not None and idx + 1 == max_batches:
                break

    return 100*(num_correct / num_samples)