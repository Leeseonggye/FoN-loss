import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from .augmentations import DataTransform
from .clustering import get_k_means_cluster_pos_dist
import pdb

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, clustering, num_cluster, training_mode, dataset_name):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]
        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if dataset_name == 'UEA':
            print("UEA permutate")
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]
        if training_mode == "self_supervised":  # no need to apply Augmentations in other modes

            self.aug1, self.aug2 = DataTransform(self.x_data, config)
            self.loss_weight_matrix = 0
            
            if clustering:
                # Do clustering instance domain
                if dataset_name == 'UCR':                
                    data = self.x_data.squeeze(1)
                    labels, loss_weight_matrix = get_k_means_cluster_pos_dist(data, 'euclidean', num_cluster)
                    self.loss_weight_matrix = loss_weight_matrix
                    labels = labels.reshape(labels.shape[0], 1, 1)
                    labels = labels.repeat(self.x_data.shape[-1], axis = -1)
                    labels = torch.from_numpy(labels)
                    self.x_data = torch.cat((self.x_data, labels), dim=1)
                
                if dataset_name == 'UEA':
                    data = self.x_data
                    labels, loss_weight_matrix = get_k_means_cluster_pos_dist(data, 'euclidean', num_cluster)
                    self.loss_weight_matrix = loss_weight_matrix
                    labels = labels.reshape(labels.shape[0], 1, 1)
                    labels = labels.repeat(self.x_data.shape[1], axis = 1)
                    labels = labels.repeat(self.x_data.shape[-1], axis = -1)
                    labels = torch.from_numpy(labels)
                    self.x_data = torch.cat((self.x_data, labels), dim=1)
            
                    
        
    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index], self.loss_weight_matrix
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len


def data_generator(data_path, configs, clustering, num_cluster, training_mode, dataset_name):

    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))
    train_dataset = Load_Dataset(train_dataset, configs, clustering, num_cluster, training_mode, dataset_name)
    valid_dataset = Load_Dataset(valid_dataset, configs, 0, 0, training_mode, dataset_name)
    test_dataset = Load_Dataset(test_dataset, configs, 0, 0, training_mode, dataset_name)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader