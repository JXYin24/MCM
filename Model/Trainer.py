import torch
import torch.optim as optim
from DataSet.DataLoader import get_dataloader
from Model.Model import CSVmodel
from Model.Loss import LossFunction
from Model.Score import ScoreFunction
from utils import aucPerformance, get_logger, F1Performance

class Trainer(object):
    def __init__(self, run: int, model_config: dict):
        self.run = run
        self.sche_gamma = model_config['sche_gamma']
        self.device = model_config['device']
        self.learning_rate = model_config['learning_rate']
        self.model = CSVmodel(model_config).to(self.device)
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
        for step, (x_input, y_label) in enumerate(self.test_loader):
            x_input = x_input.to(self.device)
            x_pred, z, masks = self.model(x_input)
            mse_batch = self.score_func(x_input, x_pred)
            mse_batch = mse_batch.data.cpu()
            mse_score.append(mse_batch)
            test_label.append(y_label)
        mse_score = torch.cat(mse_score, axis=0).numpy()
        test_label = torch.cat(test_label, axis=0).numpy()
        mse_rauc[self.run], mse_ap[self.run] = aucPerformance(mse_score, test_label)
        mse_f1[self.run] = F1Performance(mse_score, test_label)