import torch
from torch import nn
import torch.nn.functional as F
import pdb
import itertools
import numpy as np
import wandb

def hierarchical_contrastive_loss(z1, z2, cl1, cl2, cl_matrix, task_type ,alpha=0.5, temporal_unit=0):
    """
    z1, z2 : latent embed
    cl1, cl2: cluster1/cluster2 label -> z1, z2의 데이터가 각각 어느 cluster에 속하는지 정보
    cl_matrix: cl간의 가중치 정보를 담아 놓은 matrix
    """
    loss = torch.tensor(0., device=z1.device)
    d = 0

    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2, cl1, cl2, cl_matrix, task_type)
            
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2, cl1, cl2, cl_matrix, task_type)
                #
        
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        if type(cl1) != int:
            cl1 = cl1[:,1::2,:]
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
        if type(cl2) != int:
            cl2 = cl2[:,1::2,:]
        
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2, cl1, cl2, cl_matrix, task_type)
        d += 1
    return loss / d

def instance_contrastive_loss(z1, z2, cl1, cl2, cl_matrix, task_type):
    B, T = z1.size(0), z1.size(1) # B = 2, T = 2925 (Time index), Batch, Time_index, latent embedding dim
    
    if B == 1:
        return z1.new_tensor(0.)
        
    if type(cl1) == int: # do not clustering
        z = torch.cat([z1, z2], dim=0)  # 2B x T x C
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)
        
        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2

    else:
        domain = 'instance'
        z = torch.cat([z1, z2], dim=0)  # 2B x T x C
        cl_cat = torch.cat([cl1, cl2], dim=0) # 2B x T x 1 
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
        
        # cl_cat_weight 가 선형결합으로 반영될 수 있도록 변경하는 부분 k= log alpha/r11 + 1
        cl_cat_weight_raw = get_weight(domain, cl_cat, cl_matrix).to(device=z1.device) # T x 2B x 2B
        # pdb.set_trace()
        # cl_cat_weight = torch.log(cl_cat_weight_raw)/sim + 1
        # weighted_sim = torch.mul(sim, cl_cat_weight)
        
        weighted_sim = sim + (torch.log(cl_cat_weight_raw))
        
        logits = torch.tril(weighted_sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
        logits += torch.triu(weighted_sim, diagonal=1)[:, :, 1:]     # T x 2B x (2B-1)
        logits = -F.log_softmax(logits, dim=-1)
        
        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        
        del cl_cat_weight_raw, weighted_sim, logits
        torch.cuda.empty_cache()
        
    return loss

def temporal_contrastive_loss(z1, z2, cl1, cl2, cl_matrix, task_type):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    
    if type(cl1) == int or task_type == 'classification': # do not clustering
        z = torch.cat([z1, z2], dim=1)  # B x 2T x C
        sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)
        
        t = torch.arange(T, device=z1.device)
        loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    
    else: # do clustering
        domain = 'temporal'
        z = torch.cat([z1, z2], dim=1)  # B x 2T x C
        cl_cat = torch.cat([cl1, cl2], dim=1) # B x 2T x 1 
        # pdb.set_trace()
        sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
        
        # cl_cat_weight 가 선형결합으로 반영될 수 있도록 변경하는 부분 k= log alpha/r11 + 1
        cl_cat_weight_raw = get_weight(domain, cl_cat, cl_matrix).to(device=z1.device) # B x 2T x 2T
        # cl_cat_weight = torch.log(cl_cat_weight_raw)
        
        weighted_sim = sim + (torch.log(cl_cat_weight_raw))
        logits = torch.tril(weighted_sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
        logits += torch.triu(weighted_sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)
        
        t = torch.arange(T, device=z1.device)
        loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
        
        del cl_cat_weight_raw, weighted_sim, logits
        torch.cuda.empty_cache()
        
    return loss

def get_weight(domain, cl_cat, cl_matrix):

    if domain == 'instance':
        cl_cat = cl_cat.transpose(0, 1) # Instance: T x 2B x 1
        t = 0
        for cl_cat_t in cl_cat: # Instance T 기준, Temporal B 기준
            # pdb.set_trace()
            cl_cat_t = np.array(cl_cat_t.squeeze().to('cpu'))
            idx = np.array(list(map(str, list(itertools.product(cl_cat_t, cl_cat_t)))))
            v = np.unique(idx)
            
            cl_cat_t_dist = np.zeros(idx.size)
            for v_i in v:
                if 'n' in v_i:
                    cl_cat_t_dist[np.where(idx == v_i)[0]] = 1
                else:
                    cl_cat_t_dist[np.where(idx == v_i)[0]] = cl_matrix[(int(eval(v_i)[0]),int(eval(v_i)[1]))]
            
            cl_cat_t_dist = cl_cat_t_dist.reshape(1, len(cl_cat_t), len(cl_cat_t))
     
            if t == 0:
                cl_cat_weight = cl_cat_t_dist
            
            else:
                cl_cat_weight = np.concatenate((cl_cat_weight, cl_cat_t_dist), axis = 0)
            
            t += 1
            
        cl_cat_weight = torch.from_numpy(cl_cat_weight).to(device=cl_cat.device)
        
                
    else:
        t = 0 
        for cl_cat_b in cl_cat: # Temporal B 기준
            half = int(cl_cat_b.size(0)/2)
            cl_cat_b = np.array(cl_cat_b[:half, :].squeeze().to('cpu'))
            idx = np.array(list(map(str, list(itertools.product(cl_cat_b, cl_cat_b)))))
            v = np.unique(idx)
            cl_cat_b_dist = np.zeros(idx.size)
                        
            for v_i in v:
                # cl_cat_b_dist[np.where(idx == v_i)[0]] = cl_matrix[eval(v_i)]
                if 'n' in v_i:
                    cl_cat_b_dist[np.where(idx == v_i)[0]] = 1
                else:
                    cl_cat_b_dist[np.where(idx == v_i)[0]] = cl_matrix[(int(eval(v_i)[0]),int(eval(v_i)[1]))]
            
            cl_cat_b_dist = cl_cat_b_dist.reshape(1, len(cl_cat_b), len(cl_cat_b))

            if t == 0:
                cl_cat_weight = cl_cat_b_dist
            
            else:
                cl_cat_weight = np.concatenate((cl_cat_weight, cl_cat_b_dist), axis = 0)
            
            t += 1
        
        cl_cat_weight = torch.from_numpy(cl_cat_weight).to(device=cl_cat.device)
        cl_cat_weight = cl_cat_weight.repeat(1,2,2)
    

                
    
    return cl_cat_weight










# def get_weight(domain, cl_cat, cl_matrix):
    
#     if domain == 'instance':
#         cl_cat = cl_cat.transpose(0, 1) # Instance: T x 2B x 1, Temporal: B x 2T x 1
            
#         for t in range(cl_cat.size(0)): # Instance T 기준, Temporal B 기준
#             # cl_cat_t_weight = torch.zeros((1, cl_cat.size(1),cl_cat.size(1)))
#             cl_cat_t = cl_cat[t, :, :]
#             cl_cat_t = torch.nan_to_num(cl_cat_t, nan=0.0)
#             for i in range(cl_cat.size(1)):
#                 look_up = cl_matrix[int(cl_cat_t[i].item())]
#                 cl_cat_t_np = cl_cat_t[:int(cl_cat.size(1))].to(torch.device('cpu')).numpy()
#                 cl_cat_t_np = cl_cat_t_np.reshape(int(cl_cat.size(1))).astype(int)
#                 cl_cat_t_dist = look_up[cl_cat_t_np] # T 차원 numpy in cpu
#                 cl_cat_t_dist = torch.from_numpy(cl_cat_t_dist).to(device=cl_cat.device)
#                 cl_cat_t_dist = cl_cat_t_dist.reshape(1, int(cl_cat.size(1)),1)
            
#             if t == 0:
#                 cl_cat_weight = cl_cat_t_dist
            
#             else:
#                 cl_cat_weight = torch.cat((cl_cat_weight, cl_cat_t_dist), dim = 0)
                
#     else: 
#         for b in range(cl_cat.size(0)): # Temporal B 기준
#             # cl_cat_b_weight = torch.zeros((1, int(cl_cat.size(1)/2),int(cl_cat.size(1)/2))) # B 
#             cl_cat_b = cl_cat[b, :, :] # Temporal: 2T x 1
#             cl_cat_b = torch.nan_to_num(cl_cat_b, nan=0.0)
#             for i in range(int(cl_cat.size(1)/2)):
#                 look_up = cl_matrix[int(cl_cat_b[i].item())]
#                 cl_cat_b_np = cl_cat_b[:int(cl_cat.size(1)/2)].to(torch.device('cpu')).numpy()
#                 cl_cat_b_np = cl_cat_b_np.reshape(int(cl_cat.size(1)/2)).astype(int)
#                 cl_cat_b_dist = look_up[cl_cat_b_np] # T 차원 numpy in cpu
#                 cl_cat_b_dist = torch.from_numpy(cl_cat_b_dist).to(device=cl_cat.device)
#                 cl_cat_b_dist = cl_cat_b_dist.reshape(1, int(cl_cat.size(1)/2),1)

#                 if i == 0:
#                     cl_cat_b_dist_total = cl_cat_b_dist
                
#                 else:
#                     cl_cat_b_dist_total = torch.cat((cl_cat_b_dist_total, cl_cat_b_dist), dim = 2)
                    
            
            
#             cl_cat_b_weight = cl_cat_b_dist_total.repeat(1, 2, 2)
            
#             if b == 0:
#                 cl_cat_weight = cl_cat_b_weight
            
#             else:
#                 cl_cat_weight = torch.cat((cl_cat_weight, cl_cat_b_weight), dim = 0)
                
    
#     return cl_cat_weight





















    

# def instance_contrastive_loss(z1, z2, cl1, cl2, cl_matrix):
#     B, T = z1.size(0), z1.size(1) # B = 2, T = 2925 (Time index), Batch, Time_index, latent embedding dim
    
#     if B == 1:
#         return z1.new_tensor(0.)
        
#     # if type(cl1) == int: # do not clustering
#     z = torch.cat([z1, z2], dim=0)  # 2B x T x C
#     z = z.transpose(0, 1)  # T x 2B x C
#     sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
#     logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
#     logits += torch.triu(sim, diagonal=1)[:, :, 1:]
#     logits = -F.log_softmax(logits, dim=-1)
    
#     i = torch.arange(B, device=z1.device)
#     loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2

#     # else:
#     #     z = torch.cat([z1, z2], dim=0)  # 2B x T x C
#     #     # cluster_labels = torch.cat([cl1, cl2], dim=0) # 2B x T x 1
#     #     cluster_labels = cl1.to('cpu').numpy().squeeze().T # T X B
#     #     for cluster in cluster_labels:
#     #         # look_up = cl_matrix[int(cluster)]
#     #         # cluster_dist = look_up[cluster_labels]
#     #         print(cluster)
        
        
                
#     #     pdb.set_trace()

#     #     z = z.transpose(0, 1)  # T x 2B x C
#     #     sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
        
   
        
        
#     #     # cl_cat_weight 가 선형결합으로 반영될 수 있도록 변경하는 부분 k= log alpha/r11 + 1
#     #     cl_cat_weight_raw = get_weight(domain, cl_cat, cl_matrix).to(device=z1.device) # T x 2B x 2B
        
#     #     # cl_cat_weight = torch.log(cl_cat_weight_raw)/sim + 1
#     #     # weighted_sim = torch.mul(sim, cl_cat_weight)
        
#     #     weighted_sim = sim + (torch.log(cl_cat_weight_raw))
        
#     #     logits = torch.tril(weighted_sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
#     #     logits += torch.triu(weighted_sim, diagonal=1)[:, :, 1:]     # T x 2B x (2B-1)
#     #     logits = -F.log_softmax(logits, dim=-1)
        
#     #     i = torch.arange(B, device=z1.device)
#     #     loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        
#     #     del cl_cat_weight_raw, weighted_sim, logits
#     #     torch.cuda.empty_cache()
        
#     return loss

# def temporal_contrastive_loss(z1, z2, cl1, cl2, cl_matrix):
#     B, T = z1.size(0), z1.size(1)
#     # pdb.set_trace()
#     if T == 1:
#         return z1.new_tensor(0.)
    
#     z = torch.cat([z1, z2], dim=1)  # B x 2T x C
#     sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
#     logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
#     logits += torch.triu(sim, diagonal=1)[:, :, 1:]
#     logits = -F.log_softmax(logits, dim=-1)
    
#     t = torch.arange(T, device=z1.device)
#     loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    
#     if type(cl1) == int: # do not clustering
#         z = torch.cat([z1, z2], dim=1)  # B x 2T x C
#         sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
#         logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
#         logits += torch.triu(sim, diagonal=1)[:, :, 1:]
#         logits = -F.log_softmax(logits, dim=-1)
        
#         t = torch.arange(T, device=z1.device)
#         loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    
#     else: # do clustering
#         domain = 'temporal'
#         z = torch.cat([z1, z2], dim=1)  # B x 2T x C
#         cl_cat = torch.cat([cl1, cl2], dim=1) # B x 2T x 1 
#         sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
        
#         # cl_cat_weight 가 선형결합으로 반영될 수 있도록 변경하는 부분 k= log alpha/r11 + 1
#         cl_cat_weight_raw = get_weight(domain, cl_cat, cl_matrix).to(device=z1.device) # B x 2T x 2T
#         # cl_cat_weight = torch.log(cl_cat_weight_raw)
        
#         weighted_sim = sim + (torch.log(cl_cat_weight_raw))
#         logits = torch.tril(weighted_sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
#         logits += torch.triu(weighted_sim, diagonal=1)[:, :, 1:]
#         logits = -F.log_softmax(logits, dim=-1)
        
#         t = torch.arange(T, device=z1.device)
#         loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
        
#         del cl_cat_weight_raw, weighted_sim, logits
#         torch.cuda.empty_cache()
        
#     return loss

# def get_weight(domain, cl_cat, cl_matrix):
    
#     if domain == 'instance':
#         cl_cat = cl_cat.transpose(0, 1) # Instance: T x 2B x 1, Temporal: B x 2T x 1
            
#         for t in range(cl_cat.size(0)): # Instance T 기준, Temporal B 기준
#             # cl_cat_t_weight = torch.zeros((1, cl_cat.size(1),cl_cat.size(1)))
#             cl_cat_t = cl_cat[t, :, :]
#             for i in range(cl_cat.size(1)):
#                 look_up = cl_matrix[int(cl_cat_t[i].item())]
#                 cl_cat_t_np = cl_cat_t[:int(cl_cat.size(1))].to(torch.device('cpu')).numpy()
#                 cl_cat_t_np = cl_cat_t_np.reshape(int(cl_cat.size(1))).astype(int)
#                 cl_cat_t_dist = look_up[cl_cat_t_np] # T 차원 numpy in cpu
#                 cl_cat_t_dist = torch.from_numpy(cl_cat_t_dist).to(device=cl_cat.device)
#                 cl_cat_t_dist = cl_cat_t_dist.reshape(1, int(cl_cat.size(1)),1)
            
#             if t == 0:
#                 cl_cat_weight = cl_cat_t_dist
            
#             else:
#                 cl_cat_weight = torch.cat((cl_cat_weight, cl_cat_t_dist), dim = 0)
                
#     else: 
#         for b in range(cl_cat.size(0)): # Temporal B 기준
#             # cl_cat_b_weight = torch.zeros((1, int(cl_cat.size(1)/2),int(cl_cat.size(1)/2))) # B 
#             cl_cat_b = cl_cat[b, :, :] # Temporal: 2T x 1
#             for i in range(int(cl_cat.size(1)/2)):
#                 look_up = cl_matrix[int(cl_cat_b[i].item())]
#                 cl_cat_b_np = cl_cat_b[:int(cl_cat.size(1)/2)].to(torch.device('cpu')).numpy()
#                 cl_cat_b_np = cl_cat_b_np.reshape(int(cl_cat.size(1)/2)).astype(int)
#                 cl_cat_b_dist = look_up[cl_cat_b_np] # T 차원 numpy in cpu
#                 cl_cat_b_dist = torch.from_numpy(cl_cat_b_dist).to(device=cl_cat.device)
#                 cl_cat_b_dist = cl_cat_b_dist.reshape(1, int(cl_cat.size(1)/2),1)

#                 if i == 0:
#                     cl_cat_b_dist_total = cl_cat_b_dist
                
#                 else:
#                     cl_cat_b_dist_total = torch.cat((cl_cat_b_dist_total, cl_cat_b_dist), dim = 2)
                    
            
            
#             cl_cat_b_weight = cl_cat_b_dist_total.repeat(1, 2, 2)
            
#             if b == 0:
#                 cl_cat_weight = cl_cat_b_weight
            
#             else:
#                 cl_cat_weight = torch.cat((cl_cat_weight, cl_cat_b_weight), dim = 0)
                
    
#     return cl_cat_weight


# def temporal_contrastive_loss(z1, z2, cl1, cl2, cl_matrix):
#     B, T = z1.size(0), z1.size(1)
#     # pdb.set_trace()
#     if T == 1:
#         return z1.new_tensor(0.)
    
#     z = torch.cat([z1, z2], dim=1)  # B x 2T x C
#     sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
#     logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
#     logits += torch.triu(sim, diagonal=1)[:, :, 1:]
#     logits = -F.log_softmax(logits, dim=-1)
    
#     t = torch.arange(T, device=z1.device)
#     loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    
#     if type(cl1) == int: # do not clustering
#         z = torch.cat([z1, z2], dim=1)  # B x 2T x C
#         sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
#         logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
#         logits += torch.triu(sim, diagonal=1)[:, :, 1:]
#         logits = -F.log_softmax(logits, dim=-1)
        
#         t = torch.arange(T, device=z1.device)
#         loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    
#     else: # do clustering
#         domain = 'temporal'
#         z = torch.cat([z1, z2], dim=1)  # B x 2T x C
#         cl_cat = torch.cat([cl1, cl2], dim=1) # B x 2T x 1 
#         sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
        
#         # cl_cat_weight 가 선형결합으로 반영될 수 있도록 변경하는 부분 k= log alpha/r11 + 1
#         cl_cat_weight_raw = get_weight(domain, cl_cat, cl_matrix).to(device=z1.device) # B x 2T x 2T
#         # cl_cat_weight = torch.log(cl_cat_weight_raw)
        
#         weighted_sim = sim + (torch.log(cl_cat_weight_raw))
#         logits = torch.tril(weighted_sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
#         logits += torch.triu(weighted_sim, diagonal=1)[:, :, 1:]
#         logits = -F.log_softmax(logits, dim=-1)
        
#         t = torch.arange(T, device=z1.device)
#         loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
        
#         del cl_cat_weight_raw, weighted_sim, logits
#         torch.cuda.empty_cache()
        
#     return loss

# def get_weight(domain, cl_cat, cl_matrix):
    
#     # if domain == 'instance':
#     #     cl_cat = cl_cat.transpose(0, 1) # Instance: T x 2B x 1, Temporal: B x 2T x 1
            
#     #     for t in range(cl_cat.size(0)): # Instance T 기준, Temporal B 기준
#     #         cl_cat_t_weight = torch.zeros((1, cl_cat.size(1),cl_cat.size(1)))
#     #         cl_cat_t = cl_cat[t, :, :]
#     #         for i in range(cl_cat.size(1)):
#     #             for j in range(cl_cat.size(1)):
#     #                 cl_cat_t_weight[:,i,j] = cl_matrix[int(cl_cat_t[i].item())][int(cl_cat_t[j].item())]
            
#     #         if t == 0:
#     #             cl_cat_weight = cl_cat_t_weight
            
#     #         else:
#     #             cl_cat_weight = torch.cat((cl_cat_weight, cl_cat_t_weight), dim = 0)

#     if domain == 'instance':
#         cl_cat = cl_cat.transpose(0, 1) # Instance: T x 2B x 1, Temporal: B x 2T x 1
            
#         for t in range(cl_cat.size(0)): # Instance T 기준, Temporal B 기준
#             # cl_cat_t_weight = torch.zeros((1, cl_cat.size(1),cl_cat.size(1)))
#             cl_cat_t = cl_cat[t, :, :]
#             for i in range(cl_cat.size(1)):
#                 look_up = cl_matrix[int(cl_cat_t[i].item())]
#                 cl_cat_t_np = cl_cat_t[:int(cl_cat.size(1))].to(torch.device('cpu')).numpy()
#                 cl_cat_t_np = cl_cat_t_np.reshape(int(cl_cat.size(1))).astype(int)
#                 cl_cat_t_dist = look_up[cl_cat_t_np] # T 차원 numpy in cpu
#                 cl_cat_t_dist = torch.from_numpy(cl_cat_t_dist).to(device=cl_cat.device)
#                 cl_cat_t_dist = cl_cat_t_dist.reshape(1, int(cl_cat.size(1)),1)
            
#             if t == 0:
#                 cl_cat_weight = cl_cat_t_dist
            
#             else:
#                 cl_cat_weight = torch.cat((cl_cat_weight, cl_cat_t_dist), dim = 0)
                
#     else: 
#         for b in range(cl_cat.size(0)): # Temporal B 기준
#             # cl_cat_b_weight = torch.zeros((1, int(cl_cat.size(1)/2),int(cl_cat.size(1)/2))) # B 
#             cl_cat_b = cl_cat[b, :, :] # Temporal: 2T x 1
#             for i in range(int(cl_cat.size(1)/2)):
#                 look_up = cl_matrix[int(cl_cat_b[i].item())]
#                 cl_cat_b_np = cl_cat_b[:int(cl_cat.size(1)/2)].to(torch.device('cpu')).numpy()
#                 cl_cat_b_np = cl_cat_b_np.reshape(int(cl_cat.size(1)/2)).astype(int)
#                 cl_cat_b_dist = look_up[cl_cat_b_np] # T 차원 numpy in cpu
#                 cl_cat_b_dist = torch.from_numpy(cl_cat_b_dist).to(device=cl_cat.device)
#                 cl_cat_b_dist = cl_cat_b_dist.reshape(1, int(cl_cat.size(1)/2),1)

#                 if i == 0:
#                     cl_cat_b_dist_total = cl_cat_b_dist
                
#                 else:
#                     cl_cat_b_dist_total = torch.cat((cl_cat_b_dist_total, cl_cat_b_dist), dim = 2)
                    
            
            
#             cl_cat_b_weight = cl_cat_b_dist_total.repeat(1, 2, 2)
            
#             if b == 0:
#                 cl_cat_weight = cl_cat_b_weight
            
#             else:
#                 cl_cat_weight = torch.cat((cl_cat_weight, cl_cat_b_weight), dim = 0)
                
    
#     return cl_cat_weight


# # 0720 과거 get_weight
# def get_weight(domain, cl_cat, cl_matrix):

#     if domain == 'instance':
#         cl_cat = cl_cat.transpose(0, 1) # Instance: T x 2B x 1
#         t = 0
#         for cl_cat_t in cl_cat: # Instance T 기준, Temporal B 기준
#             cl_cat_t = np.array(cl_cat_t.squeeze().to('cpu'))
#             idx = np.array(list(map(str, list(itertools.product(cl_cat_t, cl_cat_t)))))
#             v = np.unique(idx)
            
#             cl_cat_t_dist = np.zeros(idx.size)
#             for v_i in v:
#                 if type(v_i[0]) != int or type(v_i[1]) != int:
#                     cl_cat_t_dist[np.where(idx == v_i)[0]] = 1
#                 else:
#                     cl_cat_t_dist[np.where(idx == v_i)[0]] = cl_matrix[(int(eval(v_i)[0]),int(eval(v_i)[1]))]
            
#             cl_cat_t_dist = cl_cat_t_dist.reshape(1, len(cl_cat_t), len(cl_cat_t))
     
#             if t == 0:
#                 cl_cat_weight = cl_cat_t_dist
            
#             else:
#                 cl_cat_weight = np.concatenate((cl_cat_weight, cl_cat_t_dist), axis = 0)
            
#             t += 1
            
#         cl_cat_weight = torch.from_numpy(cl_cat_weight).to(device=cl_cat.device)
                
#     else:
#         t = 0 
#         for cl_cat_b in cl_cat: # Temporal B 기준
#             half = int(cl_cat_b.size(0)/2)
#             cl_cat_b = np.array(cl_cat_b[:half, :].squeeze().to('cpu'))
#             idx = np.array(list(map(str, list(itertools.product(cl_cat_b, cl_cat_b)))))
#             v = np.unique(idx)
#             cl_cat_b_dist = np.zeros(idx.size)
                        
#             for v_i in v:
#                 # cl_cat_b_dist[np.where(idx == v_i)[0]] = cl_matrix[eval(v_i)]
#                 if type(v_i[0]) != int or type(v_i[1]) != int:
#                     cl_cat_b_dist[np.where(idx == v_i)[0]] = 1
#                 else:
#                     cl_cat_b_dist[np.where(idx == v_i)[0]] = cl_matrix[(int(eval(v_i)[0]),int(eval(v_i)[1]))]
            
#             cl_cat_b_dist = cl_cat_b_dist.reshape(1, len(cl_cat_b), len(cl_cat_b))

#             if t == 0:
#                 cl_cat_weight = cl_cat_b_dist
            
#             else:
#                 cl_cat_weight = np.concatenate((cl_cat_weight, cl_cat_b_dist), axis = 0)
            
#             t += 1
        
#         cl_cat_weight = torch.from_numpy(cl_cat_weight).to(device=cl_cat.device)
#         cl_cat_weight = cl_cat_weight.repeat(1,2,2)
    
#     # else: 
#     #     for b in range(cl_cat.size(0)): # Temporal B 기준
#     #         # cl_cat_b_weight = torch.zeros((1, int(cl_cat.size(1)/2),int(cl_cat.size(1)/2))) # B 
#     #         cl_cat_b = cl_cat[b, :, :] # Temporal: 2T x 1
#     #         for i in range(int(cl_cat.size(1)/2)):
#     #             look_up = cl_matrix[int(cl_cat_b[i].item())]
#     #             cl_cat_b_np = cl_cat_b[:int(cl_cat.size(1)/2)].to(torch.device('cpu')).numpy()
#     #             cl_cat_b_np = cl_cat_b_np.reshape(int(cl_cat.size(1)/2)).astype(int)
#     #             cl_cat_b_dist = look_up[cl_cat_b_np] # T 차원 numpy in cpu
#     #             cl_cat_b_dist = torch.from_numpy(cl_cat_b_dist).to(device=cl_cat.device)
#     #             cl_cat_b_dist = cl_cat_b_dist.reshape(1, int(cl_cat.size(1)/2),1)

#     #             if i == 0:
#     #                 cl_cat_b_dist_total = cl_cat_b_dist
                
#     #             else:
#     #                 cl_cat_b_dist_total = torch.cat((cl_cat_b_dist_total, cl_cat_b_dist), dim = 2)
                    
            
            
#     #         cl_cat_b_weight = cl_cat_b_dist_total.repeat(1, 2, 2)
            
#     #         if b == 0:
#     #             cl_cat_weight = cl_cat_b_weight
            
#     #         else:
#     #             cl_cat_weight = torch.cat((cl_cat_weight, cl_cat_b_weight), dim = 0)
    
#     # else:
#     #     t = 0 
#     #     for cl_cat_b in cl_cat: # Temporal B 기준
#     #         cl_cat_b = np.array(cl_cat_b.squeeze().to('cpu'))
#     #         idx = np.array(list(map(str, list(itertools.product(cl_cat_b, cl_cat_b)))))
#     #         v = np.unique(idx)
#     #         cl_cat_b_dist = np.zeros(idx.size)
            
#     #         for v_i in v:
#     #             # cl_cat_b_dist[np.where(idx == v_i)[0]] = cl_matrix[eval(v_i)]
#     #             cl_cat_b_dist[np.where(idx == v_i)[0]] = cl_matrix[(int(eval(v_i)[0]),int(eval(v_i)[1]))]
            
#     #         cl_cat_b_dist = cl_cat_b_dist.reshape(1, len(cl_cat_b), len(cl_cat_b))

#     #         if t == 0:
#     #             cl_cat_weight = cl_cat_b_dist
            
#     #         else:
#     #             cl_cat_weight = np.concatenate((cl_cat_weight, cl_cat_b_dist), axis = 0)
            
#     #         t += 1
        
#     #     cl_cat_weight = torch.from_numpy(cl_cat_weight).to(device=cl_cat.device)
    
                
    
#     return cl_cat_weight

# def get_weight(domain, cl_cat, cl_matrix):

#     if domain == 'instance':
#         cl_cat = cl_cat.transpose(0, 1) # Instance: T x 2B x 1
#         t = 0
#         for cl_cat_t in cl_cat: # Instance T 기준, Temporal B 기준
#             cl_cat_t = np.array(cl_cat_t.squeeze().to('cpu'))
#             idx = np.array(list(map(str, list(itertools.product(cl_cat_t, cl_cat_t)))))
#             v = np.unique(idx)
            
#             cl_cat_t_dist = np.zeros(idx.size)
#             for v_i in v:
#                 if type(v_i[0]) != int or type(v_i[1]) != int:
#                     cl_cat_t_dist[np.where(idx == v_i)[0]] = 1
#                 else:
#                     cl_cat_t_dist[np.where(idx == v_i)[0]] = cl_matrix[(int(eval(v_i)[0]),int(eval(v_i)[1]))]
            
#             cl_cat_t_dist = cl_cat_t_dist.reshape(1, len(cl_cat_t), len(cl_cat_t))
     
#             if t == 0:
#                 cl_cat_weight = cl_cat_t_dist
            
#             else:
#                 cl_cat_weight = np.concatenate((cl_cat_weight, cl_cat_t_dist), axis = 0)
            
#             t += 1
            
#         cl_cat_weight = torch.from_numpy(cl_cat_weight).to(device=cl_cat.device)
                
    
#     else: 
#         for b in range(cl_cat.size(0)): # Temporal B 기준
#             # cl_cat_b_weight = torch.zeros((1, int(cl_cat.size(1)/2),int(cl_cat.size(1)/2))) # B 
#             cl_cat_b = cl_cat[b, :, :] # Temporal: 2T x 1
#             for i in range(int(cl_cat.size(1)/2)):
#                 look_up = cl_matrix[int(cl_cat_b[i].item())]
#                 cl_cat_b_np = cl_cat_b[:int(cl_cat.size(1)/2)].to(torch.device('cpu')).numpy()
#                 cl_cat_b_np = cl_cat_b_np.reshape(int(cl_cat.size(1)/2)).astype(int)
#                 cl_cat_b_dist = look_up[cl_cat_b_np] # T 차원 numpy in cpu
#                 cl_cat_b_dist = torch.from_numpy(cl_cat_b_dist).to(device=cl_cat.device)
#                 cl_cat_b_dist = cl_cat_b_dist.reshape(1, int(cl_cat.size(1)/2),1)

#                 if i == 0:
#                     cl_cat_b_dist_total = cl_cat_b_dist
                
#                 else:
#                     cl_cat_b_dist_total = torch.cat((cl_cat_b_dist_total, cl_cat_b_dist), dim = 2)
                    
            
            
#             cl_cat_b_weight = cl_cat_b_dist_total.repeat(1, 2, 2)
            
#             if b == 0:
#                 cl_cat_weight = cl_cat_b_weight
            
#             else:
#                 cl_cat_weight = torch.cat((cl_cat_weight, cl_cat_b_weight), dim = 0)
    
                
    
#     return cl_cat_weight