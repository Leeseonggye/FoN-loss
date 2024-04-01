# FoN loss

This repository contains the official implementation for the paper [Focus on Your Negative Samples in Time Series Representation Learning]

## Data

The datasets can be obtained and put into `baslines/datasets/` folder in the following way:

* [128 UCR datasets](https://www.cs.ucr.edu/~eamonn/time_series_data_2018)
* [30 UEA datasets](http://www.timeseriesclassification.com)
* [3 ETT datasets](https://github.com/zhouhaoyi/ETDataset)
* [Weather datasets](https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR)
* [Exchange-rate datasets](https://github.com/laiguokun/multivariate-time-series-data/tree/master/exchange_rate)

## Usage

Each basline model folder have UCR.sh, UEA.sh, forecasting.sh
You can just run bashfile and get result of each model when applying FoN_loss