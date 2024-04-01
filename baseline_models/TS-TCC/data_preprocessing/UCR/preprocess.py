import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pdb


dataset_path = "baseline_models/datasets/UCR"
dataset_list = os.listdir(dataset_path)

def preprocess_UCR(dataset):
    path = "baseline_models/datasets/UCR"
    train_file = os.path.join(path, dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join(path, dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    output_dir = f"baseline_models/TS-TCC/data/UCR/{dataset}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    y_train_val = train_df.iloc[:, 0].to_numpy()
    y_train_val = y_train_val
    X_train_val = train_df.iloc[:, 1:].to_numpy()
    y_test = test_df.iloc[:, 0].to_numpy()
    X_test = test_df.iloc[:, 1:].to_numpy()
    
    mean = np.nanmean(X_train_val)
    std = np.nanstd(X_train_val)
    X_train_val = (X_train_val - mean) / std
    X_test = (X_test - mean) / std
    
    labels = np.unique(y_train_val)
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i
    
    y_train_val = np.vectorize(transform.get)(y_train_val)
    y_test = np.vectorize(transform.get)(y_test)

    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=32)
    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(X_train).unsqueeze(1)
    dat_dict["labels"] = torch.from_numpy(y_train)
    torch.save(dat_dict, os.path.join(output_dir, "train.pt"))

    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(X_val).unsqueeze(1)
    dat_dict["labels"] = torch.from_numpy(y_val)
    torch.save(dat_dict, os.path.join(output_dir, "val.pt"))

    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(X_test).unsqueeze(1)
    dat_dict["labels"] = torch.from_numpy(y_test)
    torch.save(dat_dict, os.path.join(output_dir, "test.pt"))

for dataset in dataset_list:
    preprocess_UCR(dataset)