import argparse
import os
import time
import datetime
import math
import numpy as np
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
import pdb
import torch
# import methods
from cost import CoST


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
    parser.add_argument('--run_name', type = str, default= 'run1', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--dataset', type = str, help='The dataset name')
    parser.add_argument('--project_name', type = str, help='The wandb project name')
    parser.add_argument('--archive' ,type=str, default='forecast_csv', required=True, help='The archive name that the dataset belongs to. This can be set to forecast_csv, or forecast_csv_univar')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=128, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=201, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=8, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', type = int, default = 1, help='Whether to perform evaluation after training')
    parser.add_argument('--validation', type = int, default = 0, help='validation or not')
    parser.add_argument('--clustering', type=int, default=0, help='Do clustering or not')
    parser.add_argument('--num_cluster', type=int, default=0, help='The number of cluster')
    parser.add_argument('--kernels', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128], help='The kernel sizes used in the mixture of AR expert layers')
    parser.add_argument('--alpha', type=float, default=0.0005, help='Weighting hyperparameter for loss function')
    parser.add_argument('--save_path', type = str, default="/root/baseline_models/ts2vec/results",help='The save path')
    
    

    args = parser.parse_args()

    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    project_name = f"{args.seed}_{args.dataset}_{args.archive}_k-means_clustering_{args.clustering}_num_cluster_{args.num_cluster}"
    args.project_name = project_name
    print(project_name)
    
    
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    if args.archive == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, loss_weight_matrix = datautils.load_forecast_csv(args.dataset, args.num_cluster, args.clustering)
        train_data = data[:, train_slice]
    elif args.archive == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, args.num_cluster, args.clustering, univar=True)
        train_data = data[:, train_slice]
    elif args.archive == 'forecast_npy':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset)
        train_data = data[:, train_slice]
    elif args.archive == 'forecast_npy_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset, univar=True)
        train_data = data[:, train_slice]
    else:
        raise ValueError(f"Archive type {args.archive} is not supported.")

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
    )
    
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    run_dir = f"training/{args.dataset}/{name_with_datetime(args.run_name)}"

    os.makedirs(run_dir, exist_ok=True)
    
    t = time.time()

    if args.clustering:
        print(train_data.shape[-1])
        model = CoST(
            input_dims=train_data.shape[-1]-1, # Except cluster labels 
            device=device,
            clustering = args.clustering,
            loss_weight_matrix = loss_weight_matrix,
            kernels=args.kernels,
            alpha=args.alpha,
            max_train_length=args.max_train_length,
            **config
        )
        
    else:
        print(train_data.shape[-1])
        model = CoST(
            input_dims=train_data.shape[-1],
            kernels=args.kernels,
            alpha=args.alpha,
            max_train_length=args.max_train_length,
            device=device,
            loss_weight_matrix = loss_weight_matrix,
            clustering = args.clustering,
            **config
        )

    loss_log = model.fit(
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )
    model.save(f'{run_dir}/model.pkl')

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
    train_time = datetime.timedelta(seconds=t)

    if args.eval:
        out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, args.max_train_length-1, args.clustering, args.validation)
        print('Evaluation result:', eval_res)
        tasks.forecast_result_to_csv(task_type, args.project_name, args.dataset, args.save_path, pred_lens, eval_res, train_time, args.validation)
        pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        pkl_save(f'{run_dir}/out.pkl', out)
    
    if args.validation:
        out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, args.max_train_length-1, args.clustering, args.validation)
        print('Validation result:', eval_res)
        tasks.forecast_result_to_csv(task_type, args.project_name, args.dataset, args.save_path, pred_lens, eval_res, train_time, args.validation)

    print("Finished.")
