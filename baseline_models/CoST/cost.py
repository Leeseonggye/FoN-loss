import sys, math, random, copy
from typing import Union, Callable, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from torch.utils.data import TensorDataset, DataLoader, Dataset

import numpy as np
from einops import rearrange, repeat, reduce

from models.encoder import CoSTEncoder
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan
import pdb


class PretrainDataset(Dataset):

    def __init__(self,
                 data,
                 loss_weight_matrix,
                 clustering,
                 sigma,
                 p=0.5,
                 multiplier=10):
        super().__init__()
        self.clustering = clustering
        if clustering:
            self.data = data[:,:,:-1]
            self.cluster_labels = data[:,:,-1].unsqueeze(dim = 2)
            self.loss_weight_matrix = loss_weight_matrix
        
        else:
            self.data = data
            self.cluster_labels = 0
            self.loss_weight_matrix = 0
            
        self.p = p
        self.sigma = sigma
        self.multiplier = multiplier
        self.N, self.T, self.D = self.data.shape # num_ts, time, dim

    def __getitem__(self, item):
        ts = self.data[item % self.N]
        
        if self.clustering:
            loss_weight_matrix = self.loss_weight_matrix
            cluster_labels = self.cluster_labels[item % self.N]
            
        else:
            loss_weight_matrix = 0
            cluster_labels = 0

        return self.transform(ts), self.transform(ts), cluster_labels, loss_weight_matrix

    def __len__(self):
        return self.data.size(0) * self.multiplier

    def transform(self, x):
        return self.jitter(self.shift(self.scale(x)))

    def jitter(self, x):
        if random.random() > self.p:
            return x
        return x + (torch.randn(x.shape) * self.sigma)

    def scale(self, x):
        if random.random() > self.p:
            return x
        return x * (torch.randn(x.size(-1)) * self.sigma + 1)

    def shift(self, x):
        if random.random() > self.p:
            return x
        return x + (torch.randn(x.size(-1)) * self.sigma)


class CoSTModel(nn.Module):
    def __init__(self,
                 encoder_q: nn.Module, encoder_k: nn.Module,
                 kernels: List[int],
                 device: Optional[str] = 'cuda',
                 dim: Optional[int] = 128,
                 alpha: Optional[float] = 0.05,
                 K: Optional[int] = 65536,
                 m: Optional[float] = 0.999, # momentum
                 T: Optional[float] = 0.07,
                 clustering = 0):
        super().__init__()
        
        self.K = K
        self.m = m
        self.T = T
        self.device = device

        self.kernels = kernels

        self.alpha = alpha
        self.clustering = clustering

        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        # create the encoders
        self.head_q = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.head_k = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        # self.register_buffer('queue', F.normalize(torch.randn(dim, K), dim=0)) # 128, 256 -> self.queue
        
        self.register_buffer('queue', F.normalize(torch.randn(dim, K), dim=0)) # 128, 256 -> self.queue
        self.register_buffer('cluster_queue', torch.full((K,), -1).unsqueeze(0)) # 1, 256 -> cluster_queue 초기화
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        


    def compute_loss(self, q, k, k_negs, cluster_negs, cluster_labels, loss_weight_matrix, clustering):
        
        # compute logits
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) # 출력 하면 지수 (exp 위에)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, k_negs]) # 출력 하면 지수 (exp 위에)
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        
        if clustering != 0: 
            # pdb.set_trace()
            if cluster_negs[:,-1].item() != -1:
                cluster_negs = cluster_negs.squeeze().to('cpu').numpy()
                loss_weight_matrix = loss_weight_matrix[0,:,:]
                status = 0
                for cluster in cluster_labels:
                    # print(f"cluster: {cluster}")
                    # print(f"cluster labels len:{len(cluster_labels)}")
                    has_nan = torch.any(torch.isnan(cluster_labels))
                    if has_nan:
                        fill_value = 0
                        cluster_labels[torch.isnan(cluster_labels)] = fill_value
                    look_up = loss_weight_matrix[int(cluster)]
                    cluster_dist = look_up[cluster_negs].unsqueeze(dim = 0)
                    
                    if status == 0:
                        cluster_dist_total = cluster_dist
                    else:
                        cluster_dist_total = torch.cat((cluster_dist_total, cluster_dist), dim = 0)
                    
                    status = status + 1
                
                logits = torch.cat([l_pos/self.T, l_neg/self.T + torch.log(cluster_dist_total)], dim=1)
                                
        
        # labels: positive key indicators - first dim of each batch
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = F.cross_entropy(logits, labels)
        # del logits, l_pos, l_neg
        # torch.cuda.empty_cache()
        return loss

    def convert_coeff(self, x, eps=1e-6):
        amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))
        phase = torch.atan2(x.imag, x.real + eps)
        return amp, phase

    def instance_contrastive_loss(self, z1, z2, cluster_labels, loss_weight_matrix, clustering):
        B, T = z1.size(0), z1.size(1)
        z = torch.cat([z1, z2], dim=0)  # 2B x T x C
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
        
        if clustering != 0:
            print("Do clustering")
            cluster_labels = cluster_labels.squeeze().to('cpu').numpy()
            loss_weight_matrix = loss_weight_matrix[0,:,:]
            status = 0
            for cluster in cluster_labels:
                look_up = loss_weight_matrix[int(cluster)]
                cluster_dist = look_up[cluster_labels].unsqueeze(dim = 1)
                
                if status == 0:
                    cluster_dist_total = cluster_dist
                else:
                    cluster_dist_total = torch.cat((cluster_dist_total, cluster_dist), dim = -1)
                
                status = status +1
            
                
            sim = torch.matmul(z, z.transpose(1, 2))
            weight = cluster_dist_total.repeat(1, 2, 2)
            weight = weight.repeat(sim.shape[0], 1, 1)
            sim = sim + (torch.log(weight))
    
            
            
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)
            
            
        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        
        # del logits, sim
        # torch.cuda.empty_cache()
        
        return loss

    def forward(self, x_q, x_k, cluster_labels, loss_weight_matrix, clustering):
        # compute query features
        rand_idx = np.random.randint(0, x_q.shape[1])
        q_t, q_s = self.encoder_q(x_q)
        
        if q_t is not None:
            q_t = F.normalize(self.head_q(q_t[:, rand_idx]), dim=-1)
            
        # Get cluster_labels in query
        if clustering != 0:
            cluster_labels = cluster_labels[:,rand_idx,:]

        # compute key features
        with torch.no_grad():  # no gradient for keys
            self._momentum_update_key_encoder()  # update key encoder
            k_t, k_s = self.encoder_k(x_k)
            if k_t is not None:
                k_t = F.normalize(self.head_k(k_t[:, rand_idx]), dim=-1)
    
        # pdb.set_trace()
        loss = 0

        loss += self.compute_loss(q_t, k_t, self.queue.clone().detach(), self.cluster_queue.clone(),cluster_labels, loss_weight_matrix, self.clustering)
        self._dequeue_and_enqueue(k_t) # batch size 만큼 ptr이 밀림
        # print(f"Keys: {self.queue_ptr}")
        self._dequeue_and_enqueue_cluster(cluster_labels) # batch size 만큼 ptr이 밀림
        # print(f"Cluster: {self.queue_ptr}")
        q_s = F.normalize(q_s, dim=-1)
        _, k_s = self.encoder_q(x_k)
        k_s = F.normalize(k_s, dim=-1)

        q_s_freq = fft.rfft(q_s, dim=1)
        k_s_freq = fft.rfft(k_s, dim=1)
        q_s_amp, q_s_phase = self.convert_coeff(q_s_freq)
        k_s_amp, k_s_phase = self.convert_coeff(k_s_freq)

        seasonal_loss = self.instance_contrastive_loss(q_s_amp, k_s_amp, cluster_labels, loss_weight_matrix, self.clustering) + \
                        self.instance_contrastive_loss(q_s_phase, k_s_phase, cluster_labels, loss_weight_matrix, self.clustering)

        loss += (self.alpha * (seasonal_loss/2))
        

        
        return loss

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update for key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        # replace keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T

        ptr = (ptr + batch_size) % self.K
        # self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_cluster(self, cluster_labels):
        batch_size = cluster_labels.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        # replace keys at ptr (dequeue and enqueue)
        self.cluster_queue[:, ptr:ptr + batch_size] = cluster_labels.T

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr


class CoST:
    def __init__(self,
                 clustering: int,
                 loss_weight_matrix,
                 input_dims: int,
                 kernels: List[int],
                 alpha: bool,
                 max_train_length: int,
                 output_dims: int = 320,
                 hidden_dims: int = 64,
                 depth: int = 10,
                 device: 'str' ='cuda',
                 lr: float = 0.001,
                 batch_size: int = 16,
                 after_iter_callback: Union[Callable, None] = None,
                 after_epoch_callback: Union[Callable, None] = None,
                 ):

        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.clustering = clustering
        self.loss_weight_matrix = loss_weight_matrix
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length

        if kernels is None:
            kernels = []

        self.net = CoSTEncoder(
            input_dims=input_dims, output_dims=output_dims,
            kernels=kernels,
            length=max_train_length,
            hidden_dims=hidden_dims, depth=depth,
        ).to(self.device)

        self.cost = CoSTModel(
            self.net,
            copy.deepcopy(self.net),
            kernels=kernels,
            dim=self.net.component_dims,
            alpha=alpha,
            K=256,
            device=self.device,
            clustering = self.clustering
        ).to(self.device)

        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0

    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):
        assert train_data.ndim == 3

        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600
            print(f"n_iter: {n_iters}")

        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)
                
        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]

        multiplier = 1 if train_data.shape[0] >= self.batch_size else math.ceil(self.batch_size / train_data.shape[0])
        train_dataset = PretrainDataset(torch.from_numpy(train_data).to(torch.float), sigma=0.5, multiplier=multiplier, clustering= self.clustering, loss_weight_matrix=self.loss_weight_matrix)
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)
        
        optimizer = torch.optim.SGD([p for p in self.cost.parameters() if p.requires_grad],
                                    lr=self.lr,
                                    momentum=0.9,
                                    weight_decay=1e-4)
        
        loss_log = []
        
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            n_epoch_iters = 0
            
            interrupted = False
            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                x_q, x_k, cluster_labels, loss_weight_matrix = map(lambda x: x.to(self.device), batch)
                if self.max_train_length is not None and x_q.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x_q.size(1) - self.max_train_length + 1)
                    x_q = x_q[:, window_offset : window_offset + self.max_train_length]
                    x_k = x_k[:, window_offset : window_offset + self.max_train_length]
                    if self.clustering != 0:
                        cluster_labels = cluster_labels[:, window_offset : window_offset + self.max_train_length]
                optimizer.zero_grad()
                # pdb.set_trace()
                loss = self.cost(x_q, x_k, cluster_labels, loss_weight_matrix, self.clustering)

                loss.backward()
                optimizer.step()

                cum_loss += loss.item()
                n_epoch_iters += 1
                
                self.n_iters += 1
                
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())

                if n_iters is not None:
                    adjust_learning_rate(optimizer, self.lr, self.n_iters, n_iters)
            
            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1

            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)

            if n_epochs is not None:
                adjust_learning_rate(optimizer, self.lr, self.n_epochs, n_epochs)
            
        return loss_log
    
    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out_t, out_s = self.net(x.to(self.device, non_blocking=True))  # l b t d
        out = torch.cat([out_t[:, -1], out_s[:, -1]], dim=-1)
        return rearrange(out.cpu(), 'b d -> b () d')
    
    def encode(self, data, mode, mask=None, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0, batch_size=None):
        if mode == 'forecasting':
            encoding_window = None
            slicing = None
        else:
            raise NotImplementedError(f"mode {mode} has not been implemented")

        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()
        
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            batch_num = 0
            for batch in loader:
                print(batch_num)
                batch_num +=1
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not casual else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l<0 else 0,
                            right=r-ts_l if r>ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slicing,
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slicing,
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slicing,
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0
                    
                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size = out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)
                        
                output.append(out)
                
            output = torch.cat(output, dim=0)

        self.net.train(org_training)
        return output.numpy()
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)


def adjust_learning_rate(optimizer, lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
