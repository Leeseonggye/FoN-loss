#!/bin/bash
# Dataset usage
# Each dataset is available on following page.
# UEA: http://www.timeseriesclassification.com, should be placed at "baseline_models/datasets/UEA/<dataset_name>/<dataset_name>_*.arff`.

folder_path="/baseline_models/datasets/UEA"
dataset=("Cricket" "PenDigits" "PhonemeSpectra" "DuckDuckGeese" "CharacterTrajectories" "BasicMotions" "StandWalkJump" "MotorImagery" "HandMovementDirection" "RacketSports" "JapaneseVowels" "Handwriting" "PEMS-SF" "ERing" "AtrialFibrillation" "Libras" "ArticularyWordRecognition" "FaceDetection" "NATOPS" "Epilepsy" "EthanolConcentration" "SelfRegulationSCP1" "Heartbeat" "SelfRegulationSCP2" "SpokenArabicDigits" "FingerMovements" "UWaveGestureLibrary" "EigenWorms" "LSST")
cluster=("3" "7" "5" "3" "4" "5" "3" "8" "7" "9" "7" "7" "7" "4" "7" "8" "4" "8" "8" "3" "4" "9" "7" "8" "8" "8" "8" "3" "7")
length=${#dataset[@]}

for ((i=0; i<$length; i++));
do
    python3 /root/baseline_models/ts2vec/train.py \
        --loader UEA \
        --dataset ${dataset[$i]} \
        --gpu 0 \
        --seed 12 \
        --clustering 1 \
        --num_cluster ${cluster[$i]}
done
