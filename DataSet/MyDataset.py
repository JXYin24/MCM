import os
import csv
import numpy as np
import pandas as pd
from scipy import io
import torch
from torch.utils.data import Dataset


class CsvDataset(Dataset):
    def __init__(self, dataset_name: str, data_dim: int, data_dir: str, mode: str = 'train'):
        super(CsvDataset, self).__init__()
        x = []
        labels = []
        path = os.path.join(data_dir, dataset_name+'.csv')
        with (open(path, 'r')) as data_from:
            csv_reader = csv.reader(data_from)
            for i in csv_reader:
                x.append(i[0:data_dim])
                labels.append(i[data_dim])

        for i in range(len(x)):
            for j in range(data_dim):
                x[i][j] = float(x[i][j])
        for i in range(len(labels)):
            labels[i] = float(labels[i])

        data = np.array(x)
        target = np.array(labels)
        inlier_indices = np.where(target == 0)[0]
        outlier_inices = np.where(target == 1)[0]
        train_data, train_label, test_data, test_label = train_test_split(data[inlier_indices], data[outlier_inices])
        if mode == 'train':
            self.data = torch.Tensor(train_data)
            self.targets = torch.Tensor(train_label)
        else:
            self.data = torch.Tensor(test_data)
            self.targets = torch.Tensor(test_label)
        print(len(self.data))

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return len(self.data)


class MatDataset(Dataset):
    def __init__(self, dataset_name: str, data_dim: int, data_dir: str, mode: str = 'train'):
        super(MatDataset, self).__init__()
        path = os.path.join(data_dir, dataset_name + '.mat')
        data = io.loadmat(path)
        samples = data['X']
        labels = ((data['y']).astype(np.int)).reshape(-1)

        inliers = samples[labels == 0]
        outliers = samples[labels == 1]
        train_data, train_label, test_data, test_label = train_test_split(inliers, outliers)
        if mode == 'train':
            self.data = torch.Tensor(train_data)
            self.targets =torch.Tensor(train_label)
        else:
            self.data = torch.Tensor(test_data)
            self.targets = torch.Tensor(test_label)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return len(self.data)
    

class NpzDataset(Dataset):
    def __init__(self, dataset_name: str, data_dim: int, data_dir: str, mode: str = 'train'):
        super(NpzDataset, self).__init__()
        path = os.path.join(data_dir, dataset_name+'.npz')
        data=np.load(path)  
        samples = data['X']
        labels = ((data['y']).astype(np.int)).reshape(-1)

        inliers = samples[labels == 0]
        outliers = samples[labels == 1]
        train_data, train_label, test_data, test_label = train_test_split(inliers, outliers)
        if mode == 'train':
            self.data = torch.Tensor(train_data)
            self.targets =torch.Tensor(train_label)
        else:
            self.data = torch.Tensor(test_data)
            self.targets = torch.Tensor(test_label)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return len(self.data)

    
def train_test_split(inliers, outliers):
    num_split = len(inliers) // 2
    train_data = inliers[:num_split]
    train_label = np.zeros(num_split)
    test_data = np.concatenate([inliers[num_split:], outliers], 0)

    test_label = np.zeros(test_data.shape[0])
    test_label[num_split:] = 1
    return train_data, train_label, test_data, test_label



if __name__ == '__main__':
    train_data, train_label, test_data, test_label = KDD_train_test_split('../Data/')
    print(train_data[0])