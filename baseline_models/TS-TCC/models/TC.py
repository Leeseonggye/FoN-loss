import torch
import torch.nn as nn
import numpy as np
from .attention import Seq_Transformer
import os
import pandas as pd
import pdb


class TC(nn.Module):
    def __init__(self, configs, dataset, dataset_name, device, clustering):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        self.timestep = configs.TC.timesteps
        if dataset == 'UCR':
            dataset_path = f"/root/baseline_models/TS-TCC/datasets/{dataset}/{dataset_name}"
            train_file = os.path.join(dataset_path, dataset_name+ "_TRAIN.tsv")
            train_df = pd.read_csv(train_file, sep='\t', header=None)
            features_len = train_df.shape[1]
            timestep = int((features_len//8)//2.5)
            self.timestep = timestep
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = device
        self.clustering = clustering
        
        self.projection_head = nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=configs.TC.hidden_dim, depth=4, heads=4, mlp_dim=64)

    def forward(self, features_aug1, features_aug2, cluster_labels, loss_weight_matrix, clustering):
        z_aug1 = features_aug1  # features are (batch_size, #channels, seq_len)
        seq_len = z_aug1.shape[2]
        z_aug1 = z_aug1.transpose(1, 2)

        z_aug2 = features_aug2
        z_aug2 = z_aug2.transpose(1, 2)
        batch = z_aug1.shape[0]
        # pdb.set_trace()
        if seq_len <= self.timestep:
            self.timestep = int(seq_len//2)
            
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)  # randomly pick time stamps
        
        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        
        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.num_channels)
        forward_seq = z_aug1[:, :t_samples + 1, :]
        
        c_t = self.seq_transformer(forward_seq)

        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        # 예측 -> W_k(c_t)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        
        # if loss_weight_matrix[0] == 0:
        if clustering == 0:
            for i in np.arange(0, self.timestep):
                total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
                nce += torch.sum(torch.diag(self.lsoftmax(total)))
        
        else:
            loss_weight_matrix = loss_weight_matrix[0,:,:].to('cpu').numpy()
            cluster_labels = cluster_labels.to('cpu').numpy()
            cluster_labels = cluster_labels[:,0]
            times = 0
            for cluster_label in cluster_labels:
                look_up = loss_weight_matrix[int(cluster_label)]
                dist = torch.from_numpy(look_up[cluster_labels.astype(int)]).unsqueeze(0).to(self.device)
                if times == 0:
                    total_dist_matrix = dist
                else:
                    total_dist_matrix = torch.cat((total_dist_matrix, dist), dim = 0)
                
                times = times + 1
                
            for i in np.arange(0, self.timestep):
                total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
                # weight = torch.log(total_dist_matrix)/total + 1
                # weighted_total = torch.mul(total, weight)
                weighted_total = total + torch.log(total_dist_matrix)
                nce += torch.sum(torch.diag(self.lsoftmax(weighted_total)))            
            

        nce /= -1. * batch * self.timestep
        return nce, self.projection_head(c_t)