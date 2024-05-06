# 1. Dataset
import os
import random
import re

import pandas as pd


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
                    file_path = os.path.join(dirpath, filename)
                    row = pd.DataFrame({"folder": [foldername], "file": [file_path]})
                    all_files = pd.concat(objs=[all_files, row], ignore_index=True)

    train_pattern = "train"
    val_pattern = "val"
    test_pattern = "test"

    train_df = all_files.loc[map(lambda x: re.search(train_pattern, x), all_files['file'])]
    val_df = all_files.loc[map(lambda x: re.search(val_pattern, x), all_files['file'])]
    test_df = all_files.loc[map(lambda x: re.search(test_pattern, x), all_files['file'])]

    return train_df, val_df, test_df