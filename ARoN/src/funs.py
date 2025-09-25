# 1. Dataset
import os
import random
import re
import torch
import torch.nn as nn
import pandas as pd

def build_mlp_pool(in_dim, out_dim, depth=1, hidden_dim=None, dropout=0.0):
    layers = []
    curr_in = in_dim
    hidden_dim = hidden_dim or out_dim  # default to out_dim if not specified
    for _ in range(depth - 1):
        layers.append(nn.Linear(curr_in, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        curr_in = hidden_dim
    layers.append(nn.Linear(curr_in, out_dim))
    layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def gather_files(root_dir):
    all_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.npz'):
                all_files.append(os.path.join(dirpath, filename))
    random.shuffle(all_files)
    return all_files


def gather_files_pgm(root_dir):
    all_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.npz'):
                all_files.append(os.path.join(dirpath, filename))
    random.shuffle(all_files)

    train_pattern = "train"
    val_pattern = "val"
    test_pattern = "test"

    train_files = [filename for filename in all_files if re.search(train_pattern, filename)]
    val_files = [filename for filename in all_files if re.search(val_pattern, filename)]
    test_files = [filename for filename in all_files if re.search(test_pattern, filename)]

    return train_files, val_files, test_files


def gather_files_by_type(root_dir):
    all_files = pd.DataFrame(columns=["folder", "file"])
    for dirpath, foldernames, filenames in os.walk(root_dir):
        for foldername in foldernames:
            folder_path = os.path.join(dirpath, foldername)
            for filename in os.listdir(folder_path):
                if filename.endswith('.npz'):
                    file_path = os.path.join(dirpath, foldername, filename)
                    row = pd.DataFrame({"folder": [foldername], "file": [file_path]})
                    all_files = pd.concat(objs=[all_files, row], ignore_index=True)

    train_pattern = "train"
    val_pattern = "val"
    test_pattern = "test"

    train_df = all_files[[bool(re.search(train_pattern, x)) for x in all_files['file']]]
    val_df = all_files[[bool(re.search(val_pattern, x)) for x in all_files['file']]]
    test_df = all_files[[bool(re.search(test_pattern, x)) for x in all_files['file']]]

    return train_df, val_df, test_df