import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from DataSet.DataLoader import get_dataloader
from Model.Model import MCM
from Model.Loss import LossFunction
from Model.Score import ScoreFunction
from utils import aucPerformance, get_logger, F1Performance

class Trainer(object):
    def __init__(self, run: int, model_config: dict):
        self.run = run
        self.model_config = model_config
        self.sche_gamma = model_config['sche_gamma']
        self.device = model_config['device']
        self.learning_rate = model_config['learning_rate']
        self.model = MCM(model_config).to(self.device)
        self.loss_fuc = LossFunction(model_config).to(self.device)
        self.score_func = ScoreFunction(model_config).to(self.device)
        self.train_loader, self.test_loader = get_dataloader(model_config)

    def training(self, epochs):
        train_logger = get_logger('train_log.log')
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()
        print("Training Start.")
        min_loss = 100
        for epoch in range(epochs):
            for step, (x_input, y_label) in enumerate(self.train_loader):
                x_input = x_input.to(self.device)
                x_pred, z, masks = self.model(x_input)
                loss, mse, divloss = self.loss_fuc(x_input, x_pred, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            info = 'Epoch:[{}]\t loss={:.4f}\t mse={:.4f}\t divloss={:.4f}\t'
            train_logger.info(info.format(epoch,loss.cpu(),mse.cpu(),divloss.cpu()))
            if loss < min_loss:
                torch.save(self.model, 'model.pth')
                min_loss = loss
        print("Training complete.")
        train_logger.handlers.clear()

    def evaluate(self, mse_rauc, mse_ap, mse_f1):
        model = torch.load('model.pth')
        model.eval()
        mse_score, test_label = [], []
        x_inputs, x_preds = [], []
        
        for step, (x_input, y_label) in enumerate(self.test_loader):
            x_input = x_input.to(self.device)
            x_pred, z, masks = model(x_input)
            mse_batch = self.score_func(x_input, x_pred)
            mse_batch = mse_batch.data.cpu()
            mse_score.append(mse_batch)
            test_label.append(y_label)
            x_inputs.append(x_input.cpu())
            x_preds.append(x_pred.cpu())
        
        mse_score = torch.cat(mse_score, axis=0).numpy()
        test_label = torch.cat(test_label, axis=0).numpy()
        x_inputs = torch.cat(x_inputs, axis=0).numpy()
        x_preds = torch.cat(x_preds, axis=0).detach().numpy()
        
        # 计算AUC-ROC、AUC-PR和F1分数
        mse_rauc[self.run], mse_ap[self.run] = aucPerformance(mse_score, test_label)
        mse_f1[self.run] = F1Performance(mse_score, test_label)
        
        # 解释性功能
        if self.model_config['explain']:
            self.calculate_feature_contributions(x_inputs, x_preds, test_label, mse_score)
        
        # 阈值校准功能
        calibration_threshold = self.calibrate_threshold()
        return calibration_threshold
    
    def calculate_feature_contributions(self, x_inputs, x_preds, test_label, mse_score):
        """计算逐特征异常贡献度"""
        dataset_name = self.model_config['dataset_name']
        
        # 创建结果目录
        result_dir = f'./results/{dataset_name}'
        os.makedirs(result_dir, exist_ok=True)
        
        # 计算每个特征的贡献度
        contributions = []
        num_samples, num_masks, num_features = x_preds.shape
        
        for sample_idx in range(num_samples):
            sample_contributions = []
            for feature_idx in range(num_features):
                # 计算数值特征的贡献度：|x-recon|/MAD
                x_original = x_inputs[sample_idx, feature_idx]
                x_recon = x_preds[sample_idx, :, feature_idx]  # 多个掩码的重构结果
                
                # 计算绝对误差
                abs_error = np.abs(x_original - x_recon)
                mean_abs_error = np.mean(abs_error)  # 平均多个掩码的结果
                
                # 计算MAD（中位数绝对偏差）
                mad = np.median(np.abs(x_inputs[:, feature_idx] - np.median(x_inputs[:, feature_idx])))
                
                # 避免除以零
                if mad == 0:
                    contribution = mean_abs_error
                else:
                    contribution = mean_abs_error / mad
                
                sample_contributions.append((feature_idx, contribution))
            
            # 对贡献度进行排序，取Top-3
            sample_contributions.sort(key=lambda x: x[1], reverse=True)
            top_3 = sample_contributions[:3]
            
            contributions.append((sample_idx, top_3))
        
        # 保存到CSV文件
        csv_path = os.path.join(result_dir, 'explanations.csv')
        with open(csv_path, 'w') as f:
            f.write('Sample ID,Feature Name,Contribution\n')
            for sample_idx, top_3 in contributions:
                for feature_idx, contribution in top_3:
                    f.write(f'{sample_idx},feature_{feature_idx},{contribution:.6f}\n')
    
    def calibrate_threshold(self):
        """基于置信区间分位数的阈值校准"""
        model = torch.load('model.pth')
        model.eval()
        
        # 从训练集中获取校准数据
        calibration_data = []
        for step, (x_input, y_label) in enumerate(self.train_loader):
            x_input = x_input.to(self.device)
            x_pred, z, masks = model(x_input)
            mse_batch = self.score_func(x_input, x_pred)
            mse_batch = mse_batch.data.cpu().numpy()
            calibration_data.append(mse_batch)
        
        calibration_data = np.concatenate(calibration_data, axis=0)
        
        # 计算(1-α)分位数阈值
        alpha = self.model_config['alpha']
        calibration_threshold = np.quantile(calibration_data, 1 - alpha)
        
        # 保存到scores.csv文件
        dataset_name = self.model_config['dataset_name']
        result_dir = f'./results/{dataset_name}'
        os.makedirs(result_dir, exist_ok=True)
        
        csv_path = os.path.join(result_dir, 'scores.csv')
        with open(csv_path, 'w') as f:
            f.write('Metric,Value\n')
            f.write(f'Calibration Threshold,{calibration_threshold:.6f}\n')
        
        return calibration_threshold