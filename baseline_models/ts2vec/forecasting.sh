#!/bin/bash
# Dataset usage
# Each dataset is available on following page.
# ETT: https://github.com/zhouhaoyi/ETDataset , should be put into "baseline_models/datasets/ETTh1.csv", "baseline_models/datasets/ETTh2.csv", "baseline_models/datasets/ETTm1.csv"
# Exchange-rate: https://github.com/laiguokun/multivariate-time-series-data/tree/master/exchange_rate, should be put into "baseline_models/datasets/exchange_rate.csv"
# WTH: https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR, should be put into "baseline_models/datasets/WTH.csv"

python3 train.py \
    --seed 22 \
    --loader forecast_csv \
    --dataset ETTh1 \
    --gpu 0 \
    --clustering 1 \
    --max-train-length 201 \
    --num_cluster 8


python3 train.py \
    --seed 12 \
    --loader forecast_csv \
    --dataset ETTh2 \
    --gpu 0 \
    --clustering 1 \
    --max-train-length 201 \
    --num_cluster 10

python3 train.py \
    --seed 42 \
    --loader forecast_csv \
    --dataset ETTm1 \
    --gpu 0 \
    --clustering 1 \
    --max-train-length 201 \
    --num_cluster 8

python3 train.py \
    --seed 32 \
    --loader forecast_csv \
    --dataset exchange_rate \
    --gpu 0 \
    --clustering 1 \
    --max-train-length 201 \
    --num_cluster 8


python3 train.py \
    --seed 42 \
    --loader forecast_csv \
    --dataset WTH \
    --gpu 0 \
    --clustering 1 \
    --max-train-length 201 \
    --num_cluster 6