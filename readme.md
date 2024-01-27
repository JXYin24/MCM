# MCM

Masked Cell Modeling for Anomaly Detection in Tabular Data


## Requirements
+ Python 3.6
+ PyTorch 1.10.2
+ torchvision 0.11.3
+ numpy 1.23.5
+ pandas 1.5.3
+ scipy 1.10.1

## Usage
1. Install this repository and the required packages.
2. Prepare dataset.
   1) Download dataset.
   2) Move the dataset into `./Data`. 
   3) The dataset needs to be entered in a specified format to fit the `Dataloader` such as '.npz' and '.mat'.
4. Run `main.py` to start training and testing the model. 
