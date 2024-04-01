#!/bin/bash
# Before do classification of UCR, You have to run "baseline_models/TS-TCC/data_preprocessing/UCR/preprocess.py"
# Dataset usage
# Each dataset is available on following page.
# UCR: https://www.cs.ucr.edu/~eamonn/time_series_data_2018) should be put into "baseline_models/datasets/<dataset_name>/<dataset_name>_*.csv" 

folder_path="/baseline_models/datasets/UCR"
dataset=("FreezerRegularTrain" "DistalPhalanxOutlineCorrect" "DiatomSizeReduction" "MelbournePedestrian" "PigCVP" "DodgerLoopWeekend" "Strawberry" "ACSF1" "PhalangesOutlinesCorrect" "AllGestureWiimoteX" "Computers" "MoteStrain" "Haptics" "UMD" "EOGVerticalSignal" "Wine" "SonyAIBORobotSurface2" "Fish" "CricketY" "Adiac" "InlineSkate" "ToeSegmentation2" "SemgHandGenderCh2" "Trace" "DodgerLoopGame" "MixedShapesSmallTrain" "StarLightCurves" "WordSynonyms" "Rock" "OSULeaf" "PigAirwayPressure" "PLAID" "PigArtPressure" "Coffee" "MiddlePhalanxTW" "CricketX" "FaceAll" "EthanolLevel" "InsectEPGSmallTrain" "UWaveGestureLibraryZ" "AllGestureWiimoteZ" "DistalPhalanxOutlineAgeGroup" "ShapesAll" "ChlorineConcentration" "OliveOil" "ProximalPhalanxOutlineAgeGroup" "ToeSegmentation1" "Plane" "UWaveGestureLibraryAll" "Worms" "GestureMidAirD3" "ShakeGestureWiimoteZ" "Fungi" "AllGestureWiimoteY" "GestureMidAirD1" "GunPointMaleVersusFemale" "CricketZ" "SonyAIBORobotSurface1" "FacesUCR" "ECGFiveDays" "FreezerSmallTrain" "SwedishLeaf" "ItalyPowerDemand" "Car" "GestureMidAirD2" "ProximalPhalanxOutlineCorrect" "SmallKitchenAppliances" "Yoga" "UWaveGestureLibraryX" "ScreenType" "InsectWingbeatSound" "Symbols" "FordB" "SemgHandSubjectCh2" "ElectricDevices" "Lightning2" "CBF" "Wafer" "DodgerLoopDay" "GesturePebbleZ2" "NonInvasiveFetalECGThorax1" "HouseTwenty" "EOGHorizontalSignal" "CinCECGTorso" "SemgHandMovementCh2" "Lightning7" "BirdChicken" "GunPoint" "DistalPhalanxTW" "HandOutlines" "FiftyWords" "TwoPatterns" "Herring" "FaceFour" "GunPointAgeSpan" "ArrowHead" "MiddlePhalanxOutlineAgeGroup" "Phoneme" "PowerCons" "Ham" "MedicalImages" "SmoothSubspace" "RefrigerationDevices" "WormsTwoClass" "InsectEPGRegularTrain" "ShapeletSim" "ProximalPhalanxTW" "Meat" "GesturePebbleZ1" "Chinatown" "Beef" "BME" "PickupGestureWiimoteZ" "Crop" "NonInvasiveFetalECGThorax2" "Earthquakes" "GunPointOldVersusYoung" "FordA" "LargeKitchenAppliances" "MiddlePhalanxOutlineCorrect" "ECG5000" "MixedShapesRegularTrain" "SyntheticControl" "Mallat" "TwoLeadECG" "UWaveGestureLibraryY" "ECG200" "BeetleFly")
cluster=("7" "9" "8" "8" "4" "3" "6" "4" "6" "4" "6" "4" "5" "9" "3" "9" "4" "4" "7" "4" "4" "6" "3" "3" "3" "8" "3" "8" "4" "3" "5" "5" "4" "5" "4" "5" "3" "3" "3" "7" "4" "6" "7" "9" "3" "7" "7" "3" "8" "9" "7" "3" "5" "4" "6" "3" "3" "9" "6" "8" "7" "8" "8" "8" "7" "7" "4" "4" "9" "6" "3" "6" "6" "3" "4" "7" "7" "5" "9" "4" "4" "4" "3" "9" "4" "7" "3" "4" "9" "7" "4" "5" "8" "4" "3" "6" "8" "6" "3" "6" "9" "7" "7" "6" "3" "8" "4" "3" "9" "4" "3" "9" "3" "8" "5" "4" "3" "7" "7" "8" "8" "5" "6" "9" "4" "9" "4" "4" )
length=${#dataset[@]}


for ((i=0; i<$length; i++));
do
    for mode in self_supervised train_linear
    do
        CUDA_VISIBLE_DEVICES=0 python3 main.py \
            --selected_dataset UCR \
            --dataset_name ${dataset[$i]} \
            --seed 42 \
            --clustering 1 \
            --training_mode $mode \
            --num_cluster ${cluster[$i]}
    done
done