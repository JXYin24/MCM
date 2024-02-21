from torch.utils.data import DataLoader
from DataSet.MyDataset import CsvDataset, MatDataset, NpzDataset


def get_dataloader(model_config: dict):
    dataset_name = model_config['dataset_name']

    if dataset_name in ['thyroid', 'arrhythmia', 'lympho', 'mnist', 'wbc', 'satimage-2', 'breastw', 'shuttle', 'pima', 'cardio', 'ionosphere', 'mammography','glass','musk','letter','vowels','satellite','shuttle']:
        train_set = MatDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], mode='train')
        test_set = MatDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], mode='eval')

    elif dataset_name in ['InternetAds', 'speech', 'backdoor', 'census', 'nslkdd', 'spambase','optdigits','fraud','campaign','cardiotocography', 'donors', 'pendigits', 'wine']:
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
