import pandas as pd
import torch
import torch.nn.functional as F

def evaluate_model_dist(model, dataloader, device, max_batches=None, reset_feedback=True):

    model.eval()
    if reset_feedback and hasattr(model, 'reset_feedback'):
        model.reset_feedback()

    with torch.no_grad():

        num_correct = 0
        num_samples = 0
        for idx, (sentences, target_nums, _, _) in enumerate(dataloader):

            batch_size = sentences.size(0)

            # move images to the device
            sentences = sentences.to(device) # shape (B,9,1,160,160)
            target_nums = target_nums.to(device)

            # forward pass
            dists, _, _, _, _, _, _ = model(sentences)

            dists_softmax = F.softmax(dists, dim = 1)

            guesses = torch.argmax(dists, dim = -1) # take highest probability guess

            num_correct += torch.eq(guesses, target_nums).sum().item()
            num_samples += sentences.size(0)

            if max_batches is not None and idx + 1 == max_batches:
                break

    if reset_feedback and hasattr(model, 'reset_feedback'):
        model.reset_feedback()

    return 100*(num_correct / num_samples)

def evaluate_model_by_type(model, dataloader, device, max_batches=None, reset_feedback=True):

    model.eval()
    if reset_feedback and hasattr(model, 'reset_feedback'):
        model.reset_feedback()

    with torch.no_grad():

        record = pd.DataFrame(columns = ["file", "folder", "correct"])
        for idx, (sentences, target_nums, folders, files) in enumerate(dataloader):

            batch_size = sentences.size(0)

            # move images to the device
            sentences = sentences.to(device) # shape (B,9,1,160,160)
            target_nums = target_nums.to(device)

            # forward pass
            dists, _, _, _, _, _, _ = model(sentences)

            dists_softmax = F.softmax(dists, dim = 1)

            guesses = torch.argmax(dists, dim = -1) # take highest probability guess

            row = pd.DataFrame({"file": files, "folder": folders, "correct": torch.eq(guesses, target_nums).cpu()})
            record = pd.concat(objs=[record, row], ignore_index=True)

            if max_batches is not None and idx + 1 == max_batches:
                break

    if reset_feedback and hasattr(model, 'reset_feedback'):
        model.reset_feedback()

    return record