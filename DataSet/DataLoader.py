from torch.utils.data import DataLoader
from DataSet.MyDataset import CsvDataset, MatDataset, NpzDataset


def get_dataloader(model_config: dict):
    dataset_name = model_config['dataset_name']


    if dataset_name in ['arrhythmia', 'breastw', 'cardio', 'glass', 'ionosphere', 'mammography', 'pima', 'satellite', 'satimage-2', 'shuttle', 'thyroid', 'wbc']:
        train_set = MatDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], mode='train')
        test_set = MatDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], mode='eval')

    elif dataset_name in ['census', 'campaign', 'cardiotocography', 'fraud', 'nslkdd', 'optdigits', 'pendigits', 'wine']:
        train_set = NpzDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], mode='train')
        test_set = NpzDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], mode='eval')
        
    else:
        train_set = CsvDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], mode='train')
        test_set = CsvDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], mode='eval')

    train_loader = DataLoader(train_set,
                              batch_size=model_config['batch_size'],
                              num_workers=model_config['num_workers'],
                              shuffle=False)
    test_loader = DataLoader(test_set, batch_size=model_config['batch_size'], shuffle=False)
    return train_loader, test_loader