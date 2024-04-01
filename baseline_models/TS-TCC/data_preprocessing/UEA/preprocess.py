import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from scipy.io.arff import loadarff
import pdb


dataset_path = "baseline_models/datasets/UEA/"
dataset_list = os.listdir(dataset_path)

def preprocess_UEA(dataset):
    path = "baseline_models/datasets/UEA/"
    train_data = loadarff(os.path.join(path, dataset,f"{dataset}_TRAIN.arff"))[0]
    test_data = loadarff(os.path.join(path, dataset,f"{dataset}_TEST.arff"))[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    X_train_val, y_train_val = extract_data(train_data) # train_X = (instance, timestamp, features)
    X_test, y_test = extract_data(test_data)
    X_train_val = np.nan_to_num(X_train_val) # fill nan to 0
    
    output_dir = f"baseline_models/TS-TCC/data/UEA/{dataset}"
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
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

    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=22)
    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(X_train)
    dat_dict["labels"] = torch.from_numpy(y_train)
    torch.save(dat_dict, os.path.join(output_dir, "train.pt"))

    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(X_val)
    dat_dict["labels"] = torch.from_numpy(y_val)
    torch.save(dat_dict, os.path.join(output_dir, "val.pt"))

    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(X_test)
    dat_dict["labels"] = torch.from_numpy(y_test)
    torch.save(dat_dict, os.path.join(output_dir, "test.pt"))


for dataset in dataset_list:
    preprocess_UEA(dataset)