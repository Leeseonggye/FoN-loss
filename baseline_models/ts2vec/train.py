import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from ts2vec import TS2Vec
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
import pdb
import wandb

def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default= 'run_1',type = str, help='run_name')
    parser.add_argument('--dataset', type = str, help='The dataset name')
    parser.add_argument('--project_name', type = str, help='The wandb project name')
    parser.add_argument('--save_path', type = str, default="baseline_models/ts2vec/results",help='The save path')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=1, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=4, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', type=int, default = 1, help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    parser.add_argument('--iter_times', type=float, default=1, help='The ratio of times of raw iter (defaults to 1)')
    parser.add_argument('--clustering', type=int, default=1, help='Do clustering or not')
    parser.add_argument('--num_cluster', type=int, default=4, help='The number of cluster')
    parser.add_argument('--random_distance', type=int, default=0, help='For ablation, random distance or not')
    parser.add_argument('--random_distance_seed', type=int, default=0, help='The random seed of random_distance')
    parser.add_argument('--visualization', type=int, default=0, help='visualize or not')
    parser.add_argument('--validation', type=int, default=0, help='validation or not')
    
    
    args = parser.parse_args()
    
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    if args.loader == 'forecast_csv':
        project_name = f"{args.seed}_{args.dataset}_multivariate_k-means_clustering_{args.clustering}_num_cluster_{args.num_cluster}"
    else:
        project_name = f"{args.dataset}_{args.seed}_{args.loader}_k-means_clustering_{args.clustering}_num_cluster_{args.num_cluster}"
        
    args.project_name = project_name
    print(project_name)
    wandb.init(
        project = args.project_name,
        name = args.project_name, 
        reinit = True,
        )
    
    wandb.config.update(args)
    
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    
    print('Loading data... ', end='')
    if args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, valid_data, valid_labels, test_data, test_labels, loss_weight_matrix = datautils.load_UCR(args.dataset, args.num_cluster, args.clustering, args.validation)
        if args.validation:
            train_data, train_labels, valid_data, valid_labels, test_data, test_labels, loss_weight_matrix = datautils.load_UCR(args.dataset, args.num_cluster, args.clustering, args.validation)
            
        if args.random_distance:
            loss_weight_matrix = np.random.random(loss_weight_matrix.shape)
            loss_weight_matrix = loss_weight_matrix / loss_weight_matrix.mean(axis=1, keepdims=True)
        
    elif args.loader == 'UEA':
        task_type = 'classification' 
        train_data, train_labels, valid_data, valid_labels, test_data, test_labels, loss_weight_matrix = datautils.load_UEA(args.dataset, args.num_cluster, args.clustering, args.validation)
        
        if args.random_distance:
            loss_weight_matrix = np.random.random(loss_weight_matrix.shape)
            loss_weight_matrix = loss_weight_matrix / loss_weight_matrix.mean(axis=1, keepdims=True)        
        
    elif args.loader == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, loss_weight_matrix = datautils.load_forecast_csv(args.dataset, args.num_cluster, args.clustering)
        
        if args.random_distance:
            np.random.seed(args.random_distance_seed)
            loss_weight_matrix = np.random.random(loss_weight_matrix.shape)
            # loss_weight_matrix = loss_weight_matrix / loss_weight_matrix.mean(axis=1, keepdims=True)
            
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, loss_weight_matrix = datautils.load_forecast_csv(args.dataset, args.num_cluster, args.clustering, univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset, args.num_cluster ,univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'anomaly':
        task_type = 'anomaly_detection'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay, loss_weight_matrix = datautils.load_anomaly(args.dataset, args.num_cluster, args.clustering)
        train_data = datautils.gen_ano_train_data(all_train_data)
        
    elif args.loader == 'anomaly_coldstart':
        task_type = 'anomaly_detection_coldstart'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
        train_data, _, _, _ = datautils.load_UCR('FordA')
        
    else:
        raise ValueError(f"Unknown loader {args.loader}.")
        
    
    if args.irregular > 0:
        if task_type == 'classification':
            train_data = data_dropout(train_data, args.irregular)
            test_data = data_dropout(test_data, args.irregular)
        else:
            raise ValueError(f"Task type {task_type} is not supported when irregular>0.")
    print('done')
    
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )
    
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    run_dir = 'training/' + args.dataset + '__' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    t = time.time()
    
    if args.clustering:
        model = TS2Vec(
            input_dims=train_data.shape[-1]-1, # cluster labels 제외
            device=device,
            clustering = args.clustering,
            task_type = task_type,
            iter_times = args.iter_times,
            **config
        )
        
    else:
        model = TS2Vec(
            input_dims=train_data.shape[-1],
            device=device,
            clustering = args.clustering,
            task_type = task_type,
            iter_times = args.iter_times,
            **config
        )
    loss_log, loss = model.fit(
        train_data,
        loss_weight_matrix,
        wandb,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )
    model.save(f'{run_dir}/model.pkl')
    
    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
        
    if task_type == 'classification':
        if args.loader == 'UCR':
            tasks.classification_validate_dataframe(args.seed, args.num_cluster, args.save_path ,args.dataset, loss.item(), args.loader, task_type)
        
        else:
            tasks.classification_validate_dataframe(args.seed, args.num_cluster, args.save_path ,args.dataset, loss.item(), args.loader, task_type)
    
    if args.validation:
        if task_type == 'classification':
            out, eval_res = tasks.eval_classification(model, train_data, train_labels, valid_data, valid_labels, args.clustering, eval_protocol='svm')
            tasks.classification_result_to_csv(task_type, args.project_name, args.save_path, eval_res, args.random_distance, args.random_distance_seed, args.validation, train_time = datetime.timedelta(seconds=t), iter_times = args.iter_times)
        
        if task_type == 'forecasting':
            print("validation")
            out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, args.clustering, args.validation)
            tasks.forecast_result_to_csv(task_type, args.project_name, args.dataset, args.save_path, pred_lens, eval_res, args.random_distance, args.random_distance_seed, args.validation, train_time = datetime.timedelta(seconds=t))
            

    if args.eval:
        if task_type == 'classification':
            out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, args.clustering, eval_protocol='svm')
            if args.visualization:
                tasks.visualization(model, args.dataset, test_data, test_labels, args.clustering, args.iter_times)           
            else:
                tasks.classification_result_to_csv(task_type, args.project_name, args.save_path, eval_res, args.random_distance, args.random_distance_seed, args.validation ,train_time = datetime.timedelta(seconds=t), iter_times = args.iter_times)
            
        elif task_type == 'forecasting':
            out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, args.clustering, args.validation)
            tasks.forecast_result_to_csv(task_type, args.project_name, args.dataset, args.save_path, pred_lens, eval_res, args.random_distance, args.random_distance_seed, args.validation, args.iter_times,train_time = datetime.timedelta(seconds=t))
            
        elif task_type == 'anomaly_detection':
            out, eval_res = tasks.eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        elif task_type == 'anomaly_detection_coldstart':
            out, eval_res = tasks.eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        else:
            assert False
        pkl_save(f'{run_dir}/out.pkl', out)
        pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        print('Evaluation result:', eval_res)

    print("Finished.")
