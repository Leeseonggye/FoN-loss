import numpy as np
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting module


import pdb

def eval_classification(model, train_data, train_labels, test_data, test_labels, clustering,eval_protocol='linear'):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    
    if clustering:
        train_data = train_data[:,:-1]
        
    train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
    test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
    

    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)
    clf = fit_clf(train_repr, train_labels)

    acc = clf.score(test_repr, test_labels)
    if eval_protocol == 'linear':
        y_score = clf.predict_proba(test_repr)
    else:
        y_score = clf.decision_function(test_repr)
    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    auprc = average_precision_score(test_labels_onehot, y_score)
    
    
    return y_score, { 'acc': acc, 'auprc': auprc }

# 2D
def visualization(model, dataset_name, test_data, test_labels, clustering, iter_times):
    test_repr = model.encode(test_data, encoding_window='full_series')
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(test_repr)
    silhouette_avg = silhouette_score(test_repr, test_labels)
    
    result_path = os.path.join(f"baseline_models/ts2vec/tsne_result_with_iter_refined/silhouette_avg_iter_times_raw.csv")
    
    silhouette_dict = {'Method': f"{dataset_name}_{clustering}_{iter_times}", 'Silhouette_Avg': silhouette_avg}
    
    if not os.path.exists(result_path):
        df = pd.DataFrame()
        df['Method'] = 0
        df['Silhouette_Avg'] = 0
        df = df.append(pd.Series(silhouette_dict), ignore_index=True)
    
    else:
        df = pd.read_csv(result_path)
        df = df.append(pd.Series(silhouette_dict), ignore_index=True)
    
    df.to_csv(result_path, index = False)

    

def classification_result_to_csv(task, method_name, path, eval_res, random_distance, random_distance_seed, validation,train_time, iter_times):
    """
    eval_res: dict -> acc, auprc
    """
    if 'UEA' in method_name:
        dataset = 'UEA'
    else:
        dataset = 'UCR'
        
    result_path = os.path.join(path, 'ts2vec', task ,dataset, f"results_{dataset}.csv")
    
    if iter_times != 1:
        result_path = os.path.join(path, 'ts2vec', task ,dataset, f"results_{dataset}_iter_times_{iter_times}_epoch.csv")
    
    if random_distance:
        result_path = os.path.join(path, 'ts2vec', task ,dataset, f"aaai_results_{dataset}_ablation_{random_distance_seed}.csv")
    
    if validation:
        result_path = os.path.join(path, 'ts2vec', task ,"validation", f"validation_results.csv")
        
    eval_res['Method'] = method_name
    eval_res['ts2vec_train_time'] = train_time
    
    if not os.path.exists(result_path):
        results = pd.DataFrame(pd.Series(eval_res)).T
        columns = ['Method', 'acc', 'auprc', 'ts2vec_train_time']
        results = results[columns]
    
    else:
        past = pd.read_csv(result_path, index_col=0)
        eval_res['Method'] = method_name
        results = past.append(eval_res, ignore_index=True)
    
    results.to_csv(result_path)
        
    return results