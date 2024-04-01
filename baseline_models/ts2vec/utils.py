import os
import pandas as pd
import numpy as np
import pickle
import torch
import random
from datetime import datetime

def result_to_csv(task, method_name, dataset, path, pred_lens, eval_res):
    result_path = os.path.join(path, task ,f"{dataset}_results.csv")
    # if os.path.exists(result_path):
    #     results = pd.read_csv(result_path)
    
    column_names = [f"MSE_{pred_len}_norm" for pred_len in pred_lens] + [f"MAE_{pred_len}_norm" for pred_len in pred_lens] + [f"MSE_{pred_len}_raw" for pred_len in pred_lens] + [f"MAE_{pred_len}_raw" for pred_len in pred_lens]
    results = pd.DataFrame()
    
    value_dict ={}
    value_dict['Method'] = method_name
    results['Method'] = method_name
    for pred_len in eval_res['ours']:
        value_dict[f'MSE_{pred_len}_norm'] = eval_res['ours'][pred_len]['norm']['MSE']
        value_dict[f'MAE_{pred_len}_norm'] = eval_res['ours'][pred_len]['norm']['MAE']
        value_dict[f'MSE_{pred_len}_raw'] = eval_res['ours'][pred_len]['raw']['MSE']
        value_dict[f'MAE_{pred_len}_raw'] = eval_res['ours'][pred_len]['raw']['MAE']
    
    for column in column_names:
        results[column] = value_dict[column]
    
    results['ts2vec_inference_time'] = eval_res['ts2vec_infer_time']
    value_dict['ts2vec_inference_time'] = eval_res['ts2vec_infer_time']
    
    results = results.append(value_dict, ignore_index=True)
    
    if not os.path.exists(os.path.join(path, task)):
        os.makedirs(os.path.join(path, task))
    
    if os.path.exists(result_path):
        past = pd.read_csv(result_path, index_col=0)
        total = pd.concat([past, results], axis=0)
        total.to_csv(result_path)
    
    else:
        results.to_csv(result_path)
        
    print(value_dict)
    return results


def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)

def pkl_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    
def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr
    
def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)

def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs

def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]

def data_dropout(arr, p):
    B, T = arr.shape[0], arr.shape[1]
    mask = np.full(B*T, False, dtype=np.bool)
    ele_sel = np.random.choice(
        B*T,
        size=int(B*T*p),
        replace=False
    )
    mask[ele_sel] = True
    res = arr.copy()
    res[mask.reshape(B, T)] = np.nan
    return res

def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")

def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        
    return devices if len(devices) > 1 else devices[0]

