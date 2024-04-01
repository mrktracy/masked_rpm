# 1. Dataset
import os
import random
import re

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