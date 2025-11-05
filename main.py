import torch
import numpy as np
import argparse
import os
import pandas as pd
from Model.Trainer import Trainer

# 添加命令行参数支持
parser = argparse.ArgumentParser(description='Model Evaluation with Explanation and Calibration')
parser.add_argument('--explain', action='store_true', help='Enable feature contribution explanation')
parser.add_argument('--alpha', type=float, default=0.05, help='Alpha value for calibration (default: 0.05)')
parser.add_argument('--calib_split', type=float, default=0.1, help='Calibration split ratio (default: 0.1)')
parser.add_argument('--batch_eval', nargs='+', help='List of datasets for batch evaluation')
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')

# 解析命令行参数
args = parser.parse_args()

# 基础模型配置
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
    'random_seed': args.seed,
    'num_workers': 0,
    'explain': args.explain,
    'alpha': args.alpha,
    'calib_split': args.calib_split
}

if __name__ == "__main__":
    torch.manual_seed(model_config['random_seed'])
    torch.cuda.manual_seed(model_config['random_seed'])
    np.random.seed(model_config['random_seed'])
    
    if model_config['num_workers'] > 0:
        torch.multiprocessing.set_start_method('spawn')
    
    # 批量评测功能
    if args.batch_eval:
        datasets = args.batch_eval
        summary_data = []
        
        for dataset_name in datasets:
            try:
                # 创建数据集特定的配置
                dataset_config = model_config.copy()
                dataset_config['dataset_name'] = dataset_name
                
                # 根据数据集名称设置正确的数据维度
                # 这里需要根据实际情况补充数据集维度信息
                dataset_dimensions = {
                    'wbc': 30,
                    'arrhythmia': 274,
                    'breastw': 30,
                    'cardio': 21,
                    'glass': 9,
                    'ionosphere': 34,
                    'mammography': 6,
                    'pima': 8,
                    'satellite': 36,
                    'satimage-2': 36,
                    'shuttle': 9,
                    'thyroid': 21
                }
                
                if dataset_name in dataset_dimensions:
                    dataset_config['data_dim'] = dataset_dimensions[dataset_name]
                else:
                    # 默认维度
                    dataset_config['data_dim'] = 30
                
                # 训练和评估模型
                runs = dataset_config['runs']
                mse_rauc, mse_ap, mse_f1 = np.zeros(runs), np.zeros(runs), np.zeros(runs)
                calibration_thresholds = []
                
                for i in range(runs):
                    trainer = Trainer(run=i, model_config=dataset_config)
                    trainer.training(dataset_config['epochs'])
                    calibration_threshold = trainer.evaluate(mse_rauc, mse_ap, mse_f1)
                    calibration_thresholds.append(calibration_threshold)
                
                mean_mse_auc = np.mean(mse_rauc)
                mean_mse_pr = np.mean(mse_ap)
                mean_mse_f1 = np.mean(mse_f1)
                mean_calibration_threshold = np.mean(calibration_thresholds)
                
                # 保存单个数据集的结果
                results_name = f'./results/{dataset_name}.txt'
                with open(results_name, 'a') as file:
                    file.write(f"epochs: {dataset_config['epochs']} lr: {dataset_config['learning_rate']:.4f} gamma: {dataset_config['sche_gamma']:.2f} masks: {dataset_config['mask_num']} lambda: {dataset_config['lambda']:.1f} ")
                    file.write('\n')
                    file.write(f"de_layer: {dataset_config['de_nlayers']}  hidden_dim: {dataset_config['hidden_dim']} z_dim: {dataset_config['z_dim']} mask_layer: {dataset_config['mask_nlayers']}")
                    file.write('\n')
                    file.write(f"mse: average AUC-ROC: {mean_mse_auc:.4f}  average AUC-PR: {mean_mse_pr:.4f} average f1: {mean_mse_f1:.4f}")
                    file.write('\n')
                
                # 添加到汇总数据
                summary_data.append({
                    'Dataset Name': dataset_name,
                    'AUROC': mean_mse_auc,
                    'AUPRC': mean_mse_pr,
                    'Calibration Threshold': mean_calibration_threshold,
                    'Remarks': 'Success'
                })
                
                print(f"\n##########################################################################")
                print(f"Dataset: {dataset_name}")
                print(f"mse: average AUC-ROC: {mean_mse_auc:.4f}  average AUC-PR: {mean_mse_pr:.4f}")
                print(f"mse: average f1: {mean_mse_f1:.4f}")
                print(f"Calibration Threshold: {mean_calibration_threshold:.6f}")
                print(f"##########################################################################")
                
            except Exception as e:
                # 处理异常情况
                summary_data.append({
                    'Dataset Name': dataset_name,
                    'AUROC': 0.0,
                    'AUPRC': 0.0,
                    'Calibration Threshold': 0.0,
                    'Remarks': f'Error: {str(e)}'
                })
                print(f"\n##########################################################################")
                print(f"Dataset: {dataset_name}")
                print(f"Error: {str(e)}")
                print(f"##########################################################################")
        
        # 保存汇总结果
        summary_df = pd.DataFrame(summary_data)
        summary_path = './results/summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n##########################################################################")
        print(f"Batch evaluation completed. Summary saved to {summary_path}")
        print(f"##########################################################################")
    
    else:
        # 单个数据集评测
        runs = model_config['runs']
        mse_rauc, mse_ap, mse_f1 = np.zeros(runs), np.zeros(runs), np.zeros(runs)
        calibration_thresholds = []
        
        for i in range(runs):
            trainer = Trainer(run=i, model_config=model_config)
            trainer.training(model_config['epochs'])
            calibration_threshold = trainer.evaluate(mse_rauc, mse_ap, mse_f1)
            calibration_thresholds.append(calibration_threshold)
        
        mean_mse_auc = np.mean(mse_rauc)
        mean_mse_pr = np.mean(mse_ap)
        mean_mse_f1 = np.mean(mse_f1)
        mean_calibration_threshold = np.mean(calibration_thresholds)

        print('##########################################################################')
        print(f"mse: average AUC-ROC: {mean_mse_auc:.4f}  average AUC-PR: {mean_mse_pr:.4f}")
        print(f"mse: average f1: {mean_mse_f1:.4f}")
        print(f"Calibration Threshold: {mean_calibration_threshold:.6f}")
        print('##########################################################################')
        
        results_name = './results/' + model_config['dataset_name'] + '.txt'

        with open(results_name,'a') as file:
            file.write(f"epochs: {model_config['epochs']} lr: {model_config['learning_rate']:.4f} gamma: {model_config['sche_gamma']:.2f} masks: {model_config['mask_num']} lambda: {model_config['lambda']:.1f} ")
            file.write('\n')
            file.write(f"de_layer: {model_config['de_nlayers']}  hidden_dim: {model_config['hidden_dim']} z_dim: {model_config['z_dim']} mask_layer: {model_config['mask_nlayers']}")
            file.write('\n')
            file.write(f"mse: average AUC-ROC: {mean_mse_auc:.4f}  average AUC-PR: {mean_mse_pr:.4f} average f1: {mean_mse_f1:.4f}")
            file.write('\n')
            file.write(f"Calibration Threshold: {mean_calibration_threshold:.6f}")
            file.write('\n')