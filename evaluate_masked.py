import torch
import torch.nn.functional as F

def evaluate_model_masked(model, dataloader, device, max_batches = None):

    model.eval()
    with torch.no_grad():

        num_correct = 0
        num_samples = 0
        for idx, (inputs, candidates, targets) in enumerate(dataloader):

            # move images to the device
            inputs = inputs.to(device) # shape (B,9,model_dim)
            candidates = candidates.to(device) # shape (B, 8, embed_dim)
            targets = targets.to(device) # shape (B,)

            # forward pass
            outputs = model(inputs).unsqueeze(1) # (batch_size,1,embed_dim)
            guesses = torch.argmin(torch.sum((candidates - outputs)**2, dim=-1), dim = -1)
            num_correct += torch.eq(guesses, targets).sum().item()
            num_samples += inputs.size(0)

            if max_batches is not None and idx + 1 == max_batches:
                break

    return 100*(num_correct / num_samples)

def evaluate_model_masked_BERT_v14(model, dataloader, device, max_batches = None):

    model.eval()
    with torch.no_grad():

        num_correct = 0
        num_samples = 0
        for idx, (inputs, cands, target_nums) in enumerate(dataloader):

            batch_size = inputs.size(0)

            # move images to the device
            inputs = inputs.to(device) # shape (B,9,1,160,160)
            target_nums = target_nums.to(device)

            # forward pass
            dists, _, _ = model(inputs)

            dists_softmax = F.softmax(dists)

            guesses = torch.argmax(dists, dim = -1) # take highest probability guess

            num_correct += torch.eq(guesses, target_nums).sum().item()
            num_samples += inputs.size(0)

            if max_batches is not None and idx + 1 == max_batches:
                break

    return 100*(num_correct / num_samples)

def evaluate_model_masked_BERT_v13(model, dataloader, device, max_batches = None):

    model.eval()
    with torch.no_grad():

        num_correct = 0
        num_samples = 0
        for idx, (inputs, mask_tensors, _, target_nums, imagetensors) in enumerate(dataloader):

            batch_size = inputs.size(0)

            # move images to the device
            inputs = inputs.to(device) # shape (B,9,1,160,160)
            candidates = imagetensors[:,8:,:,:,:].to(device) # shape (B, 8, 1, 160, 160)
            target_nums = target_nums.to(device)
            # mask_tensors = mask_tensors.to(device) # 1s where the mask is, 0s elsewhere

            candidates_embed = model.encode(candidates.reshape(batch_size*8, 1, 160, 160)).reshape(batch_size, 8, -1)

            # forward pass
            outputs, _ = model(inputs)
            outputs = outputs.unsqueeze(1)

            guesses = torch.argmin(torch.sum((candidates_embed - outputs)**2, dim=2), dim = -1) # take least squares guess

            num_correct += torch.eq(guesses, target_nums).sum().item()
            num_samples += inputs.size(0)

            if max_batches is not None and idx + 1 == max_batches:
                break

    return 100*(num_correct / num_samples)

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
            mask_tensors = mask_tensors.to(device) # 1s where the mask is, 0s elsewhere

            # forward pass
            outputs, _ = model(inputs)
            outputs = outputs.unsqueeze(1)
            # outputs = torch.sum(outputs*mask_tensors, dim=1).unsqueeze(1) # extract only guess

            guesses = torch.argmin(torch.sum((candidates - outputs)**2, dim=[2,3,4]), dim = -1) # take least squares guess
            # guesses = torch.argmin(torch.mean(torch.abs(candidates - outputs), dim=[2,3,4]), dim = -1) # take closest L1 guess

            num_correct += torch.eq(guesses, target_nums).sum().item()
            num_samples += inputs.size(0)

            if max_batches is not None and idx + 1 == max_batches:
                break

    return 100*(num_correct / num_samples)