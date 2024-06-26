import torch

import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad
from dataloader.dataloader import data_generator
from trainer.trainer import Trainer, model_evaluate
from models.TC import TC
from utils import _calc_metrics, copy_Files
from models.model import base_Model
import pandas as pd
import pdb

# Args selections
start_time = datetime.now()


parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='UCR', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--clustering', default=0, type=int,
                    help='Do clustering or not')
parser.add_argument('--num_cluster', default=5, type=int,
                    help='# of cluster')
parser.add_argument('--training_mode', default='supervised', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
parser.add_argument('--selected_dataset', default='UCR', type=str,
                    help='Dataset of choice: UCR, UEA, sleepEDF, HAR, Epilepsy, pFD')
parser.add_argument('--dataset_name', default='Yoga', type=str,
                    help='Specific dataset name in UCR or UEA')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')

parser.add_argument('--validation', default=1, type=int,
                    help='Do validation or not')

args = parser.parse_args()



device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'TS-TCC'
training_mode = args.training_mode
run_description = args.run_description
clustering = args.clustering

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)


exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0


# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets
# if args.selected_dataset == "UCR" or args.selected_dataset == "UEA":
data_path = f"/baseline_models/TS-TCC/data/{data_type}/{args.dataset_name}"


train_dl, valid_dl, test_dl,  = data_generator(data_path, configs, args.clustering, args.num_cluster ,training_mode, args.selected_dataset)
logger.debug("Data loaded ...")

# Load Model
model = base_Model(configs, args.selected_dataset, args.dataset_name).to(device)
temporal_contr_model = TC(configs, args.selected_dataset, args.dataset_name, device, args.clustering).to(device)
if training_mode == "fine_tune":
    # load saved model of this experiment
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

if training_mode == "train_linear" or "tl" in training_mode:
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # delete these parameters (Ex: the linear layer at the end)
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    set_requires_grad(model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.

if training_mode == "random_init":
    model_dict = model.state_dict()

    # delete all the parameters except for logits
    del_list = ['logits']
    pretrained_dict_copy = model_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del model_dict[i]
    set_requires_grad(model, model_dict, requires_grad=False)  # Freeze everything except last layer.


model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

if training_mode == "self_supervised":  # to do it only once
    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

# Trainer
Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, device, logger, configs, experiment_log_dir, training_mode, clustering)
result_path = f"baseline_models/ts2vec/results/tstcc/{args.selected_dataset}/result_tstcc.csv"

if args.validation:
    result_path = f"baseline_models/ts2vec/results/tstcc/{args.selected_dataset}/validation_result.csv"
    
if training_mode == "self_supervised":
    method = f"{args.seed}_{args.dataset_name}_{args.clustering}_{args.num_cluster}"
    if not os.path.exists(result_path):
        results = pd.DataFrame({'Method': [method], "Time": [datetime.now()-start_time]})
        results.to_csv(result_path)
    
    else:
        past = pd.read_csv(result_path, index_col = 0)
        now = pd.DataFrame({'Method': [method], "Time": [datetime.now()-start_time]})
        results = past.append(now, ignore_index = True)
        results.to_csv(result_path)

if training_mode != "self_supervised":
    # Validation
    if args.validation:
        outs = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        total_loss, total_acc, pred_labels, true_labels = outs
        method = f"{args.seed}_{args.dataset_name}_{args.clustering}_{args.num_cluster}"
        if not os.path.exists(result_path):
            results = pd.DataFrame({'Method': [method], "ACC": [total_acc.item()], "Time": [datetime.now()-start_time]})
            results.to_csv(result_path)
        
        else:
            past = pd.read_csv(result_path, index_col = 0)
            now = pd.DataFrame({'Method': [method], "ACC": [total_acc.item()], "Time": [datetime.now()-start_time]})
            results = past.append(now, ignore_index = True)
            results.to_csv(result_path)   
        
    # Testing
    else:
        outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        total_loss, total_acc, pred_labels, true_labels = outs
        method = f"{args.seed}_{args.dataset_name}_{args.clustering}_{args.num_cluster}"
        if not os.path.exists(result_path):
            results = pd.DataFrame({'Method': [method], "ACC": [total_acc.item()], "Time": [datetime.now()-start_time]})
            results.to_csv(result_path)
        
        else:
            past = pd.read_csv(result_path, index_col = 0)
            now = pd.DataFrame({'Method': [method], "ACC": [total_acc.item()], "Time": [datetime.now()-start_time]})
            results = past.append(now, ignore_index = True)
            results.to_csv(result_path)


logger.debug(f"Training time is : {datetime.now()-start_time}")


