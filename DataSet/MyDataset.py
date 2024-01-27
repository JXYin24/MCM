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


def KDD_preprocessing(path):
    file_names = [path + "kddcup.data_10_percent", path + "kddcup.names"]

    column_name = pd.read_csv(file_names[1], skiprows=1, sep=':', names=['f_names', 'f_types'])
    column_name.loc[column_name.shape[0]] = ['status', ' symbolic.']
    data = pd.read_csv(file_names[0], header=None, names=column_name['f_names'].values)
    data_symbolic = column_name[column_name['f_types'].str.contains('symbolic.')]
    data_continuous = column_name[column_name['f_types'].str.contains('continuous.')]
    samples = pd.get_dummies(data.iloc[:, :-1], columns=data_symbolic['f_names'][:-1])

    sample_keys = samples.keys()
    continuous_idx = []
    for cont_idx in data_continuous['f_names']:
        continuous_idx.append(sample_keys.get_loc(cont_idx))

    labels = np.where(data['status'] == 'normal.', 1, 0)
    return np.array(samples), np.array(labels), continuous_idx


def norm_kdd_data(train_data, test_data, continuous_idx):
    symbolic_idx = np.delete(np.arange(train_data.shape[1]), continuous_idx)
    mu = np.mean(train_data[:, continuous_idx],0,keepdims=True)
    std = np.std(train_data[:, continuous_idx],0,keepdims=True)
    std[std == 0] = 1
    train_continual = (train_data[:, continuous_idx]-mu)/std
    train_normalized = np.concatenate([train_data[:, symbolic_idx], train_continual], 1)
    test_continual = (test_data[:, continuous_idx]-mu)/std
    test_normalized = np.concatenate([test_data[:, symbolic_idx], test_continual], 1)

    return train_normalized, test_normalized


def KDD_train_test_split(path, rev=False):
    samples, labels, continual_idx = KDD_preprocessing(path)
    if rev == False:
        inliers = samples[labels == 0]
        outliers = samples[labels == 1]
        idx_perm = np.random.permutation(inliers.shape[0])
        inliers = inliers[idx_perm]
    else:
        inliers = samples[labels == 1]
        outliers = samples[labels == 0]
        random_cut = np.random.permutation(len(outliers))[:24319]
        outliers = outliers[random_cut]
        idx_perm = np.random.permutation(inliers.shape[0])
        inliers = inliers[idx_perm]

    train_data, train_label, test_data, test_label = train_test_split(inliers, outliers)
    train_data, test_data = norm_kdd_data(train_data, test_data, continual_idx)

    if rev == True:
        train_label = 1. - train_label
        test_label = 1. - test_label

    return train_data, train_label, test_data, test_label


class KDDdataset(Dataset):
    def __init__(self, data, label):
        super(KDDdataset, self).__init__()
        self.data = data
        self.label = label

    def __getitem__(self, item):
        return torch.tensor(self.data[item]).to(torch.float32), \
               torch.tensor(self.label[item]).to(torch.float32)

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = KDD_train_test_split('../Data/')
    print(train_data[0])