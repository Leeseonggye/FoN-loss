import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from models.clustering import get_k_means_cluster_pos_dist
import pdb
from sklearn.model_selection import train_test_split

def load_UCR(dataset, n_cluster, clustering, validation):
    train_file = os.path.join('../datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('../datasets/UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    trian_df_fill = train_df.fillna(0)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(trian_df_fill)
    test_array = np.array(test_df)
    
    # For clustering
    trian_df_fill = train_df.fillna(0)
    cluster_train = trian_df_fill.dropna(axis=1)
    cluster_train_array = np.array(cluster_train)
    cluster_train_array = cluster_train_array[:, 1:].astype(np.float64)
    
    # validation initialization
    valid =0
    valid_labels =0
    
    
    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])
    
    if validation:
        train, valid, train_labels, valid_labels = train_test_split(train, train_labels, test_size=0.2, random_state=12, shuffle=True)

    # loss_weight_matrix initialization
    loss_weight_matrix = 0
    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        if clustering:
            cluster_labels, loss_weight_matrix = get_k_means_cluster_pos_dist(train, 'euclidean', n_cluster)
            cluster_labels = cluster_labels.reshape(len(cluster_labels),1) # (instance, 1)
            cluster_labels = cluster_labels.repeat(train.shape[1], axis = 1) # train.shape = (instance, ts, feature)
            cluster_labels = cluster_labels[..., np.newaxis]
            train = train[..., np.newaxis]
            
            train = np.concatenate((train, cluster_labels), axis = -1)
        
        else:
            train = train[..., np.newaxis]
        
        if validation:
            valid = valid[..., np.newaxis]
        
        return train, train_labels, valid, valid_labels, test[..., np.newaxis], test_labels, loss_weight_matrix
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    
    
    
    if clustering:
        cluster_labels, loss_weight_matrix = get_k_means_cluster_pos_dist(train, 'euclidean', n_cluster)
        cluster_labels = cluster_labels.reshape(len(cluster_labels),1) # (instance, 1)
        cluster_labels = cluster_labels.repeat(train.shape[1], axis = 1) # train.shape = (instance, ts, feature)
        cluster_labels = cluster_labels[..., np.newaxis]
        train = train[..., np.newaxis]
        train = train
        
        train = np.concatenate((train, cluster_labels), axis = -1)
    else:
        train = train[..., np.newaxis]
    
    if validation:
        valid = valid[..., np.newaxis]
    
    return train, train_labels, valid, valid_labels,test[..., np.newaxis], test_labels, loss_weight_matrix



def load_UEA(dataset, n_cluster, clustering, validation):
    train_data = loadarff(f'./datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'./datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data) # train_X = (instance, timestamp, features)
    test_X, test_y = extract_data(test_data)
    
    # validation initialization
    valid_X = 0
    valid_y = 0

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    if validation:
        train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.2, random_state=12)

    
    cluster_train_array = np.nan_to_num(train_X, nan = 0) # (instance, ts, features)
    
    # loss_weight_matrix initialization
    loss_weight_matrix = 0
    
    if clustering:
        cluster_labels, loss_weight_matrix = get_k_means_cluster_pos_dist(cluster_train_array, 'euclidean', n_cluster)
        cluster_labels = cluster_labels.reshape(len(cluster_labels), 1, 1) # (ts, 1)

        # train.shape = (instance, ts, feature)
        cluster_labels = cluster_labels.repeat(train_X.shape[1], axis = 1)
    
    if clustering:
        train_X = np.concatenate((train_X, cluster_labels), axis = -1)
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    valid_y = np.vectorize(transform.get)(valid_y)
    
    
    return train_X, train_y, valid_X, valid_y, test_X, test_y, loss_weight_matrix
    
    
def load_forecast_npy(name, univar=False):
    data = np.load(f'datasets/{name}.npy')    
    if univar:
        data = data[: -1:]
        
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0


def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)


def load_forecast_csv(name, n_cluster, clustering, univar=False):
    data = pd.read_csv(f'./datasets/{name}.csv', index_col='date', parse_dates=True)
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]
    
    loss_weight_matrix = 0
    
    if clustering:
        cluster_labels, loss_weight_matrix = get_k_means_cluster_pos_dist(data, 'euclidean', n_cluster)
        # cluster_labels, loss_weight_matrix = get_k_means_cluster_pos_dist(data, 'euclidean', n_cluster)
    
    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        else:
            data = data.iloc[:,-1]
    
    data = data.to_numpy()
        
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24, 16*30*24)
        test_slice = slice(16*30*24, 20*30*24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4, 16*30*24*4)
        test_slice = slice(16*30*24*4, 20*30*24*4)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    if name in ('electricity'):
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0)
    
    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)
        
    # cluster_labels get same dim to data
    if clustering:
        cluster_labels = np.expand_dims(cluster_labels, axis=[0,2])
        if name == 'electricity':
            cluster_labels = np.repeat(cluster_labels, data.shape[0], axis=0)
        
        data = np.concatenate((data, cluster_labels), axis=2)
        
    if name in ('ETTh1', 'ETTh2', 'electricity', 'WTH', 'exchange_rate'):
        pred_lens = [24, 48, 168, 336, 720]
    else:
        pred_lens = [24, 48, 96, 288, 672]
    
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, loss_weight_matrix


def load_anomaly(name, num_cluster, clustering):
    res = pkl_load(f'datasets/{name}.pkl')
    loss_weight_matrix = 0
    return res['all_train_data'], res['all_train_labels'], res['all_train_timestamps'], \
           res['all_test_data'],  res['all_test_labels'],  res['all_test_timestamps'], \
           res['delay'], loss_weight_matrix


def gen_ano_train_data(all_train_data):
    maxl = np.max([ len(all_train_data[k]) for k in all_train_data ])
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(all_train_data[k], maxl, axis=0)
        pretrain_data.append(train_data)
    pretrain_data = np.expand_dims(np.stack(pretrain_data), 2)
    return pretrain_data