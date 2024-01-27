import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LossFunction(nn.Module):
    def __init__(self, model_config):
        super(LossFunction, self).__init__()
        self.mask_num = model_config['mask_num']
        self.divloss = DiversityMask()
        self.lamb = model_config['lambda']

    def forward(self, x_input, x_pred, masks):
        x_input = x_input.unsqueeze(1).repeat(1, self.mask_num, 1)
        sub_result = x_pred - x_input
        mse = torch.norm(sub_result, p=2, dim=2)
        mse_score = torch.mean(mse, dim=1, keepdim=True)
        e = torch.mean(mse_score)
        divloss = self.divloss(masks)
        loss = torch.mean(e) + self.lamb*torch.mean(divloss)
        return loss, torch.mean(e), torch.mean(divloss)


class DiversityMask(nn.Module):
    def __init__(self,temperature=0.1):
        super(DiversityMask, self).__init__()
        self.temp = temperature

    def forward(self,z,eval=False):
        z = F.normalize(z, p=2, dim=-1)
        batch_size, mask_num, z_dim = z.shape
        sim_matrix = torch.exp(torch.matmul(z, z.permute(0, 2, 1) / self.temp))
        mask = (torch.ones_like(sim_matrix).to(z) - torch.eye(mask_num).unsqueeze(0).to(z)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(batch_size, mask_num, -1)
        trans_matrix = sim_matrix.sum(-1)
        K = mask_num - 1
        scale = 1 / np.abs(K*np.log(1.0 / K))
        loss_tensor = torch.log(trans_matrix) * scale
        if eval:
            score = loss_tensor.sum(1)
            return score
        else:
            loss = loss_tensor.sum(1)
            return loss
