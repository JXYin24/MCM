# Masked Cell Modelling for Anomaly Detection In Tabular Data

This code is the official implementation of the paper: [MCM: Masked Cell Modelling for Anomaly Detection in Tabular Data](https://openreview.net/forum?id=lNZJyEDxy4) published at ICLR 2024 as a conference paper by Jiaxin Yin, Yuanyuan Qiao, Zitang Zhou, Xiangchao Wang, and Jie Yang. The code allows the users to reproduce and extend the results reported in the study. Please cite the above paper when reporting, reproducing or extending the results.


## Prepare dataset
   1) When using your own data, download dataset and move the dataset into `./Data`. 
   2) Add the dataset name to `./Dataset/DataLoader.py` based on the format of your dataset.
   3) Modify `dataset_name` and `data_dim` in `./main.py`

## Run
Run `main.py` to start training and testing the model. Results will be automatically stored in `./results`.


## Requirements
```
- Python 3.6
- PyTorch 1.10.2
- torchvision 0.11.3
- numpy 1.23.5
- pandas 1.5.3
- scipy 1.10.1
```
