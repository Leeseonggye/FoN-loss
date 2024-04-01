#!/bin/bash
# Dataset usage
# Each dataset is available on following page.
# UCR: https://www.cs.ucr.edu/~eamonn/time_series_data_2018) should be put into "baseline_models/datasets/<dataset_name>/<dataset_name>_*.csv" 

folder_path="/baseline_models/datasets/UCR"
dataset=("FreezerRegularTrain" "DistalPhalanxOutlineCorrect" "DiatomSizeReduction" "MelbournePedestrian" "PigCVP" "DodgerLoopWeekend" "Strawberry" "ACSF1" "PhalangesOutlinesCorrect" "AllGestureWiimoteX" "Computers" "MoteStrain" "Haptics" "UMD" "EOGVerticalSignal" "Wine" "SonyAIBORobotSurface2" "Fish" "CricketY" "Adiac" "InlineSkate" "ToeSegmentation2" "SemgHandGenderCh2" "Trace" "DodgerLoopGame" "MixedShapesSmallTrain" "StarLightCurves" "WordSynonyms" "Rock" "OSULeaf" "PigAirwayPressure" "PLAID" "PigArtPressure" "Coffee" "MiddlePhalanxTW" "CricketX" "FaceAll" "EthanolLevel" "InsectEPGSmallTrain" "UWaveGestureLibraryZ" "AllGestureWiimoteZ" "DistalPhalanxOutlineAgeGroup" "ShapesAll" "ChlorineConcentration" "OliveOil" "ProximalPhalanxOutlineAgeGroup" "ToeSegmentation1" "Plane" "UWaveGestureLibraryAll" "Worms" "GestureMidAirD3" "ShakeGestureWiimoteZ" "Fungi" "AllGestureWiimoteY" "GestureMidAirD1" "GunPointMaleVersusFemale" "CricketZ" "SonyAIBORobotSurface1" "FacesUCR" "ECGFiveDays" "FreezerSmallTrain" "SwedishLeaf" "ItalyPowerDemand" "Car" "GestureMidAirD2" "ProximalPhalanxOutlineCorrect" "SmallKitchenAppliances" "Yoga" "UWaveGestureLibraryX" "ScreenType" "InsectWingbeatSound" "Symbols" "FordB" "SemgHandSubjectCh2" "ElectricDevices" "Lightning2" "CBF" "Wafer" "DodgerLoopDay" "GesturePebbleZ2" "NonInvasiveFetalECGThorax1" "HouseTwenty" "EOGHorizontalSignal" "CinCECGTorso" "SemgHandMovementCh2" "Lightning7" "BirdChicken" "GunPoint" "DistalPhalanxTW" "HandOutlines" "FiftyWords" "TwoPatterns" "Herring" "FaceFour" "GunPointAgeSpan" "ArrowHead" "MiddlePhalanxOutlineAgeGroup" "Phoneme" "PowerCons" "Ham" "MedicalImages" "SmoothSubspace" "RefrigerationDevices" "WormsTwoClass" "InsectEPGRegularTrain" "ShapeletSim" "ProximalPhalanxTW" "Meat" "GesturePebbleZ1" "Chinatown" "Beef" "BME" "PickupGestureWiimoteZ" "Crop" "NonInvasiveFetalECGThorax2" "Earthquakes" "GunPointOldVersusYoung" "FordA" "LargeKitchenAppliances" "MiddlePhalanxOutlineCorrect" "ECG5000" "MixedShapesRegularTrain" "SyntheticControl" "Mallat" "TwoLeadECG" "UWaveGestureLibraryY" "ECG200" "BeetleFly")
cluster=("3" "7" "7" "7" "9" "7" "6" "8" "5" "9" "9" "7" "9" "9" "5" "5" "5" "7" "4" "8" "4" "7" "8" "9" "8" "5" "6" "3" "9" "5" "5" "7" "3" "9" "5" "4" "9" "3" "9" "4" "3" "4" "6" "6" "5" "9" "4" "8" "5" "6" "3" "6" "5" "8" "6" "3" "9" "8" "4" "9" "5" "7" "3" "7" "3" "8" "5" "3" "3" "9" "8" "5" "7" "6" "5" "4" "9" "8" "4" "3" "5" "6" "7" "7" "6" "9" "6" "9" "3" "6" "5" "9" "6" "5" "9" "8" "3" "5" "8" "3" "8" "6" "9" "9" "9" "4" "9" "8" "6" "8" "9" "9" "7" "4" "5" "9" "9" "7" "3" "3" "4" "4" "7" "6" "5" "3" "7" "9")
length=${#dataset[@]}

for ((i=0; i<$length; i++));
do
    python3 train.py \
        --loader UCR \
        --dataset ${dataset[$i]} \
        --gpu 0 \
        --clustering 1 \
        --seed 22 \
        --num_cluster ${cluster[$i]}
done