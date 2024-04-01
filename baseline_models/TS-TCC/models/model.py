from torch import nn
import os
import pandas as pd
import math
from scipy.io.arff import loadarff
import numpy as np

import pdb

class base_Model(nn.Module):
    def __init__(self, configs, dataset, dataset_name):
        super(base_Model, self).__init__()
        
        dataset_path = f"/root/baseline_models/TS-TCC/datasets/{dataset}/{dataset_name}"
        
        if dataset == "UCR":
            train_file = os.path.join(dataset_path, dataset_name+ "_TRAIN.tsv")
            train_df = pd.read_csv(train_file, sep='\t', header=None)
            num_classes = len(train_df.iloc[:,0].unique())
            features_len = train_df.shape[1]-1
        
        if dataset == "UEA":
            path = "/root/baseline_models/TS-TCC/datasets/UEA"
            train_data = loadarff(os.path.join(path, dataset_name, f"{dataset_name}_TRAIN.arff"))[0]
            
            def extract_data(data):
                res_data = []
                res_labels = []
                for t_data, t_label in data:
                    t_data = np.array([ d.tolist() for d in t_data ])
                    t_label = t_label.decode("utf-8")
                    res_data.append(t_data)
                    res_labels.append(t_label)
                return np.array(res_data).swapaxes(1, 2), np.array(res_labels)

            X_train_val, y_train_val = extract_data(train_data)
            input_channels = X_train_val.shape[-1]
            features_len = X_train_val.shape[1]
            num_classes = len(np.unique(y_train_val))
            
        def conv_1d_feature_len(features_len):
            padded_feature = features_len +1
            
            if padded_feature % 2 ==0:
                padded_feature = padded_feature/2 + 1
            
            else:
                padded_feature = math.ceil(padded_feature/2)
            
            return int(padded_feature)
            
        print(f"Raw features_len(# of timestamps):{features_len}")
        features_len = conv_1d_feature_len(conv_1d_feature_len(conv_1d_feature_len(features_len)))
        print(f"features_len(# of reduced timestamps):{features_len}")
        print(f"num_classes:{num_classes}")
        
        
        if dataset == 'UEA':
            self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout))
        
        else:               
            self.conv_block1 = nn.Sequential(
                nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                        stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                nn.Dropout(configs.dropout)
            )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
                
        # model_output_dim = configs.features_len
        
        model_output_dim = features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, num_classes)


    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        
        return logits, x

