import torch

def evaluate_model_masked_BERT(model, dataloader, device, max_batches = None):

    model.eval()
    with torch.no_grad():

        num_correct = 0
        num_samples = 0
        for idx, (inputs, mask_tensors, _, target_nums, imagetensors) in enumerate(dataloader):

            # move images to the device
            inputs = inputs.to(device) # shape (B,9,1,160,160)
            candidates = imagetensors[:,8:,:,:,:].to(device) # shape (B, 8, 1, 160, 160)
            target_nums = target_nums.to(device)
            mask_tensors = mask_tensors.to(device)

            # forward pass
            outputs = model(inputs,mask_tensors).unsqueeze(1) # (B, 1, 1, 160, 160)
            guesses = torch.argmin(torch.sum((candidates - outputs)**2, dim=[2,3,4]), dim = -1) # take least squares guess
            num_correct += torch.eq(guesses, target_nums).sum().item()
            num_samples += inputs.size(0)

            if max_batches is not None and idx + 1 == max_batches:
                break

    return 100*(num_correct / num_samples)