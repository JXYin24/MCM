import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreFunction(nn.Module):
    def __init__(self, model_config):
        super(ScoreFunction, self).__init__()
        self.mask_num = model_config['mask_num']

    def forward(self, x_input, x_pred):
        x_input = x_input.unsqueeze(1).repeat(1, self.mask_num, 1)
        sub_result = x_pred - x_input
        mse = torch.norm(sub_result, p=2, dim=2)
        mse_score = torch.mean(mse, dim=1,keepdim=True)
        return mse_score



    
