import pandas as pd
import torch
import torch.nn.functional as F

def evaluate_model_dist(model_1, model_2, model_3, dataloader, device, max_batches=None,
                        feedback_1=None, feedback_2=None, feedback_3=None):

    model_1.eval()
    model_2.eval()
    model_3.eval()

    with torch.no_grad():

        num_correct = 0
        num_samples = 0
        for idx, (sentences, target_nums, _, _) in enumerate(dataloader):

            batch_size = sentences.size(0)

            # move images to the device
            sentences = sentences.to(device) # shape (B,9,1,160,160)
            target_nums = target_nums.to(device)

            if feedback_1 is not None:
                feedback_1 = feedback_1.to(device)

            if feedback_2 is not None:
                feedback_2 = feedback_2.to(device)

            if feedback_3 is not None:
                feedback_3 = feedback_3.to(device)

            # forward pass
            dist_1, _, _, _, _, _, _, feedback = model_1(sentences, feedback_3)
            dist_2, _, _, _, _, _, _, feedback = model_1(sentences, feedback_1)
            dist_3, _, _, _, _, _, _, feedback = model_1(sentences, feedback_2)

            dist = torch.stack([dist_1, dist_2, dist_3]).mean(dim=0)

            dist_softmax = F.softmax(dist, dim=-1)

            guesses = torch.argmax(dist_softmax, dim=-1) # take highest probability guess

            num_correct += torch.eq(guesses, target_nums).sum().item()
            num_samples += sentences.size(0)

            if max_batches is not None and idx + 1 == max_batches:
                break

    return 100*(num_correct / num_samples), None