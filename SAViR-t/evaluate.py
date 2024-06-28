import pandas as pd
import torch

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
            dists, _ = model(sentences)

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
            dists, _ = model(sentences)

            guesses = torch.argmax(dists, dim = -1) # take highest probability guess

            row = pd.DataFrame({"file": files, "folder": folders, "correct": torch.eq(guesses, target_nums).cpu()})
            record = pd.concat(objs=[record, row], ignore_index=True)

            if max_batches is not None and idx + 1 == max_batches:
                break

    return record