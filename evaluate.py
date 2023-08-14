import numpy as np
import torch
import os

# Define model evaluation
def pick_answers(outputs, candidates):
    mse = torch.mean((outputs.unsqueeze(1) - candidates)**2)
    min_indices = torch.argmin(mse, dim=-1)

    return min_indices

def evaluate_model(model, dataloader, autoencoder, save_path, device):
    os.makedirs(save_path, exist_ok=True)  # make file path if it doesn't exist, do nothing otherwise

    model.eval()
    with torch.no_grad():

        num_correct = 0
        imgnum = 0
        for idx, (inputs, targets, imagetensors, target_nums, embeddings, mask_tensors) in enumerate(dataloader):

            batch_size = len(inputs)
            offset_target_nums = target_nums + 8 # offset by 8

            # move images to the device
            inputs = inputs.to(device)
            mask_tensors = mask_tensors.to(device)

            # forward pass
            outputs = model(inputs) # (batch_size,9,512)
            guesses = (outputs * mask_tensors).sum(dim=1)
            guesses_cut = guesses[:,0:256]
            outputs_cut = outputs[:,:,0:256]

            candidates = embeddings[:,8:,:].to(device) # embeddings is shape (batch_size, 16, 256)

            min_indices = pick_answers(guesses_cut, candidates).cpu()
            num_correct += torch.sum(min_indices == target_nums)

            guess_images = autoencoder.decode(guesses_cut) # get image form of guesses
            target_images = imagetensors[torch.arange(batch_size), offset_target_nums] # get image form of target
            decoded_target_images = autoencoder.decode(targets)

            output_image_list = []
            for b in range(outputs_cut.shape[0]):
                inner_list = []
                for i in range(outputs_cut.shape[1]):
                    vector = outputs_cut[b,i,:]
                    vector_decoded = autoencoder.decode(vector.unsqueeze(0))
                    inner_list.append(vector_decoded.squeeze(0))
                output_image_list.append(torch.stack(inner_list))

            output_image_grids = torch.stack(output_image_list)

            # print(f"guess_images shape: {guess_images.shape}")
            # print(f"target_images shape: {target_images.shape}")
            # print(f"decoded_target_images shape: {decoded_target_images.shape}")

            idx = 0
            for guess, target, decoded_target, imagetensor, output_image_grid in \
                    zip(guess_images, target_images, decoded_target_images, imagetensors, \
                        output_image_grids):
                if idx >= 1:  # only save first 1 images from each mini-batch
                    break
                guess = guess.cpu().numpy()
                target = target.cpu().numpy()
                decoded_target = decoded_target.cpu().numpy()
                imagetensor = imagetensor.cpu().numpy()
                output_image_grid = output_image_grid.cpu().numpy()

                filename = f"eval_{imgnum}"
                np.savez(os.path.join(save_path, filename), guess=guess, \
                         target=target, decoded_target=decoded_target, \
                         imagetensor=imagetensor, output_image_grid=output_image_grid)
                imgnum += 1
                idx += 1

    return num_correct / len(dataloader.dataset)