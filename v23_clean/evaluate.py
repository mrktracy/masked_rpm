import pandas as pd
import torch
import torch.nn.functional as F

def evaluate_model_embed(model, dataloader, device, max_batches = None):

    model.eval()
    with torch.no_grad():

        num_correct = 0
        num_samples = 0
        for idx, (inputs, cands_image, target_nums, targets) in enumerate(dataloader):

            batch_size = inputs.size(0)

            # move images to the device
            inputs = inputs.to(device) # shape (B,9,1,160,160)
            target_nums = target_nums.to(device)

            # forward pass
            outputs, _, cands_embed = model(inputs, cands_image)

            # get take "guesses" as closest in MSE
            outputs = outputs.unsqueeze(1) # (B, 1, embed_dim)
            guesses = torch.argmin(torch.sum((cands_embed - outputs) ** 2, dim=-1), dim=-1)

            # tally the correct answers
            num_correct += torch.eq(guesses, target_nums).sum().item()
            num_samples += inputs.size(0)

            if max_batches is not None and idx + 1 == max_batches:
                break

    return 100*(num_correct / num_samples)

def evaluate_model_dist(model, dataloader, device, max_batches = None):

    model.eval()
    with torch.no_grad():

        num_correct = 0
        num_samples = 0
        for idx, (sentences, target_nums, _, _) in enumerate(dataloader):

            batch_size = sentences.size(0)

            # move images to the device
            sentences = sentences.to(device) # shape (B,9,1,160,160)
            target_nums = target_nums.to(device)

            # forward pass
            dists, _, _ = model(sentences)

            dists_softmax = F.softmax(dists, dim = 1)

            guesses = torch.argmax(dists, dim = -1) # take highest probability guess

            num_correct += torch.eq(guesses, target_nums).sum().item()
            num_samples += sentences.size(0)

            if max_batches is not None and idx + 1 == max_batches:
                break

    return 100*(num_correct / num_samples)

def evaluate_model_by_type(model, dataloader, device, max_batches = None):

    model.eval()
    with torch.no_grad():

        record = pd.DataFrame(columns = ["file", "folder", "correct"])
        for idx, (sentences, target_nums, folders, files) in enumerate(dataloader):

            batch_size = sentences.size(0)

            # move images to the device
            sentences = sentences.to(device) # shape (B,9,1,160,160)
            target_nums = target_nums.to(device)

            # forward pass
            dists, _, _ = model(sentences)

            dists_softmax = F.softmax(dists, dim = 1)

            guesses = torch.argmax(dists, dim = -1) # take highest probability guess

            row = pd.DataFrame({"file": files, "folder": folders, "correct": torch.eq(guesses, target_nums).cpu()})
            record = pd.concat(objs=[record, row], ignore_index=True)

            if max_batches is not None and idx + 1 == max_batches:
                break

    return record