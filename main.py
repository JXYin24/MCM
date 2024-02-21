import torch
import numpy as np
from Model.Trainer import Trainer
model_config = {
    'dataset_name': 'wbc',
    'data_dim': 30,
    'epochs': 200,
    'learning_rate': 0.05,
    'sche_gamma': 0.98,
    'mask_num': 15,
    'lambda': 5,
    'device': 'cuda:0',
    'data_dir': 'Data/',
    'runs': 1,
    'batch_size': 512, 
    'en_nlayers': 3,
    'de_nlayers': 3,
    'hidden_dim': 256,
    'z_dim': 128,
    'mask_nlayers': 3,
    'random_seed': 42,
    'num_workers': 0
}

if __name__ == "__main__":
    torch.manual_seed(model_config['random_seed'])
    torch.cuda.manual_seed(model_config['random_seed'])
    np.random.seed(model_config['random_seed'])
    if model_config['num_workers'] > 0:
        torch.multiprocessing.set_start_method('spawn')
    result = []
    runs = model_config['runs']
    mse_rauc, mse_ap, mse_f1 = np.zeros(runs), np.zeros(runs), np.zeros(runs)
    for i in range(runs):
        trainer = Trainer(run=i, model_config=model_config)
        trainer.training(model_config['epochs'])
        trainer.evaluate(mse_rauc, mse_ap, mse_f1)
    mean_mse_auc , mean_mse_pr , mean_mse_f1 = np.mean(mse_rauc), np.mean(mse_ap), np.mean(mse_f1)

    print('##########################################################################')
    print("mse: average AUC-ROC: %.4f  average AUC-PR: %.4f"
          % (mean_mse_auc, mean_mse_pr))
    print("mse: average f1: %.4f" % (mean_mse_f1))
    results_name = './results/' + model_config['dataset_name'] + '.txt'

    with open(results_name,'a') as file:
        file.write("epochs: %d lr: %.4f gamma: %.2f masks: %d lambda: %.1f " % (
            model_config['epochs'], model_config['learning_rate'], model_config['sche_gamma'], model_config['mask_num'], model_config['lambda']))
        file.write('\n')
        file.write("de_layer: %d  hidden_dim: %d z_dim: %d mask_layer: %d" % (model_config['de_nlayers'], model_config['hidden_dim'], model_config['z_dim'], model_config['mask_nlayers']))
        file.write('\n')
        file.write("mse: average AUC-ROC: %.4f  average AUC-PR: %.4f average f1: %.4f" % (
            mean_mse_auc, mean_mse_pr, mean_mse_f1))
        file.write('\n')