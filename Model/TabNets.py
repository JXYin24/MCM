import torch
import torch.nn as nn
import numpy as np 


class TabNeutralAD(nn.Module):
    def __init__(self, model, config):
        super(TabNeutralAD, self).__init__()
        self.masks = model._make_nets(config['data_dim'], config['mask_nlayers'], config['mask_num'])
        self.mask_num = config['mask_num']
        self.device = config['device']

    def forward(self, x):
        x = x.type(torch.FloatTensor).to(self.device)
        x_T = torch.empty(x.shape[0], self.mask_num, x.shape[-1]).to(x)
        masks = []
        for i in range(self.mask_num):
            mask = self.masks[i](x)
            masks.append(mask.unsqueeze(1))
            mask = torch.sigmoid(mask)
            x_T[:, i] = mask * x
        masks = torch.cat(masks, axis=1)
        return x_T, masks


class TabTransformNet(nn.Module):
    def __init__(self, x_dim, h_dim, num_layers):
        super(TabTransformNet, self).__init__()
        net = []
        input_dim = x_dim
        for _ in range(num_layers-1):
            net.append(nn.Linear(input_dim,h_dim,bias=False))
            net.append(nn.ReLU())
            input_dim= h_dim
        net.append(nn.Linear(input_dim,x_dim,bias=False))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        out = self.net(x)
        return out


class TabNets():
    def _make_nets(self, x_dim, mask_nlayers, mask_num):
        trans = nn.ModuleList(
            [TabTransformNet(x_dim, x_dim, mask_nlayers) for _ in range(mask_num)])
        return trans

