#!/bin/bash
# Before do classification of UEA, You have to run "baseline_models/TS-TCC/data_preprocessing/UEA/preprocess.py"
# Dataset usage
# Each dataset is available on following page.
# UEA: http://www.timeseriesclassification.com, should be placed at "baseline_models/datasets/UEA/<dataset_name>/<dataset_name>_*.arff`.

folder_path="/baseline_models/datasets/UEA"
dataset=("AtrialFibrillation" "Libras" "ArticularyWordRecognition" "FaceDetection" "NATOPS" "Epilepsy" "EthanolConcentration" "SelfRegulationSCP1" "Heartbeat" "SelfRegulationSCP2" "SpokenArabicDigits" "FingerMovements" "UWaveGestureLibrary" "EigenWorms" "LSST")
cluster=("9" "3" "3" "8" "3" "5" "3" "9" "6" "8" "3" "6" "4" "5" "4" "3" "8" "7" "3" "6" "9" "3" "4" "8" "3" "9" "5" "6" "3")
length=${#dataset[@]}

for ((i=0; i<$length; i++));
do
    for mode in self_supervised train_linear
    do
        CUDA_VISIBLE_DEVICES=0 python3 main.py \
            --selected_dataset UEA \
            --dataset_name ${dataset[$i]} \
            --seed 32 \
            --clustering 1 \
            --training_mode $mode \
            --num_cluster ${cluster[$i]}
    done
done