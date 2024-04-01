import numpy as np
import pandas as pd
import os
import time
from . import _eval_protocols as eval_protocols

def generate_pred_samples(features, data, pred_len, drop=0):
    n = data.shape[1]
    features = features[:, :-pred_len]
    labels = np.stack([ data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
    features = features[:, drop:]
    labels = labels[:, drop:]
    return features.reshape(-1, features.shape[-1]), \
            labels.reshape(-1, labels.shape[2]*labels.shape[3])

def cal_metrics(pred, target):
    return {
        'MSE': ((pred - target) ** 2).mean(),
        'MAE': np.abs(pred - target).mean()
    }
    
def eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, clustering, validation):
    padding = 200
    
    t = time.time()
    if clustering:
        data = data[:,:,:-1]
        
    all_repr = model.encode(
        data,
        casual=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )
    ts2vec_infer_time = time.time() - t
    
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    
    train_data = data[:, train_slice, n_covariate_cols:]
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]
    
    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    
    if validation:
        for pred_len in pred_lens:
            train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
            valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
            test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)
            
            t = time.time()
            lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
            lr_train_time[pred_len] = time.time() - t
            
            t = time.time()
            valid_pred = lr.predict(valid_features)
            lr_infer_time[pred_len] = time.time() - t

            ori_shape = valid_data.shape[0], -1, pred_len, valid_data.shape[2]
            valid_pred = valid_pred.reshape(ori_shape)
            valid_labels = valid_labels.reshape(ori_shape)
            
            if valid_data.shape[0] > 1:
                valid_pred_inv = scaler.inverse_transform(valid_pred.swapaxes(0, 3)).swapaxes(0, 3)
                valid_labels_inv = scaler.inverse_transform(valid_labels.swapaxes(0, 3)).swapaxes(0, 3)
            else:
                valid_pred_inv = scaler.inverse_transform(valid_pred)
                valid_labels_inv = scaler.inverse_transform(valid_labels)
                
            out_log[pred_len] = {
                'norm': valid_pred,
                'raw': valid_pred_inv,
                'norm_gt': valid_labels,
                'raw_gt': valid_labels_inv
            }
            ours_result[pred_len] = {
                'norm': cal_metrics(valid_pred, valid_labels),
                'raw': cal_metrics(valid_pred_inv, valid_labels_inv)
            }
            
        eval_res = {
            'ours': ours_result,
            'ts2vec_infer_time': ts2vec_infer_time,
            'lr_train_time': lr_train_time,
            'lr_infer_time': lr_infer_time
        }
        
    if validation == 0:    
        for pred_len in pred_lens:
            train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
            valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
            test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)
            
            t = time.time()
            lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
            lr_train_time[pred_len] = time.time() - t
            
            t = time.time()
            test_pred = lr.predict(test_features)
            lr_infer_time[pred_len] = time.time() - t

            ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
            test_pred = test_pred.reshape(ori_shape)
            test_labels = test_labels.reshape(ori_shape)
            
            if test_data.shape[0] > 1:
                test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
                test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
            else:
                test_pred_inv = scaler.inverse_transform(test_pred)
                test_labels_inv = scaler.inverse_transform(test_labels)
                
            out_log[pred_len] = {
                'norm': test_pred,
                'raw': test_pred_inv,
                'norm_gt': test_labels,
                'raw_gt': test_labels_inv
            }
            ours_result[pred_len] = {
                'norm': cal_metrics(test_pred, test_labels),
                'raw': cal_metrics(test_pred_inv, test_labels_inv)
            }
            
        eval_res = {
            'ours': ours_result,
            'ts2vec_infer_time': ts2vec_infer_time,
            'lr_train_time': lr_train_time,
            'lr_infer_time': lr_infer_time
        }
        
    return out_log, eval_res

def forecast_result_to_csv(task, method_name, dataset, path, pred_lens, eval_res, random_distance, random_distance_seed, validation, iter_times,train_time):
    result_path = os.path.join(path,'ts2vec', task ,f"{dataset}_results.csv")
    if iter_times != 1:
        result_path = os.path.join(path, 'ts2vec', task ,'iter_times', f"{dataset}_results_{iter_times}.csv")

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
    
    results['ts2vec_train_time'] = train_time
    value_dict['ts2vec_train_time'] = train_time
    
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