# Masked Cell Modelling for Anomaly Detection (MCM)

This code is the official implementation of the paper: MCM: Masked Cell Modelling for Anomaly Detection in Tabular Data published at ICLR 2024 as a conference paper by Jiaxin Yin, Yuanyuan Qiao, Zitang Zhou, Xiangchao Wang, and Jie Yang. The code allows the users to reproduce and extend the results reported in the study. Please cite the above paper when reporting, reproducing or extending the results.

## Requirements
```
- Python 3.6
- PyTorch 1.10.2
- torchvision 0.11.3
- numpy 1.23.5
- pandas 1.5.3
- scipy 1.10.1
```

## Prepare dataset
   1) Download dataset.
   2) Move the dataset into `./Data`. 
   3) The dataset needs to be entered in a specified format to fit the `Dataloader` such as '.npz' and '.mat'.

## Run
Run `main.py` to start training and testing the model. Results will be automatically stored in `./results`.
