import numpy as np
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
import os
import pandas as pd
import pdb

def eval_classification(model, train_data, train_labels, test_data, test_labels, clustering, eval_protocol='linear'):
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
    
    return y_score, {'acc': acc, 'auprc': auprc }

def classification_result_to_csv(task, method_name, path, eval_res, train_time):
    """
    eval_res: dict -> acc, auprc
    """
    result_path = os.path.join(path, task ,"results_tstcc.csv")
    eval_res['Method'] = method_name
    
    if not os.path.exists(result_path):
        results = pd.DataFrame(pd.Series(eval_res)).T
        columns = ['Method', 'acc', 'auprc']
        results = results[columns]
        results['ts2vec_train_time'] = train_time
        results.to_csv(result_path)
    
    else:
        past = pd.read_csv(result_path, index_col=0)
        eval_res['Method'] = method_name
        results = past.append(eval_res, ignore_index=True)
        results.to_csv(result_path)
        
    return results