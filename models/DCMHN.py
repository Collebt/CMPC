from models.model import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.transforms.functional import rgb_to_grayscale

    
class Hash_Layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_logit = (x - 0.5)
        x_logit[x_logit > 0] = 1
        x_logit[x_logit <= 0] = -1
        b = (x_logit + 1) / 2
        return b


    
class DCMHN(nn.Module):
    def __init__(self, args):
        super(DCMHN, self).__init__()

        low_dim = args.low_dim
        drop = args.drop
        arch = args.arch

        pool_dim = 2048
        self.gap_layer = nn.AdaptiveMaxPool2d((1, 1))

        self.l2norm = Normalize(2)
        self.latent_layer = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.hash_layer_train = nn.Sigmoid()
        self.hash_layer_test = Hash_Layer()
        # self.bn = nn.BatchNorm1d(low_dim)

        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        layers = []
        layers.extend([model_ft.conv1, model_ft.bn1, model_ft.relu, model_ft.maxpool])
        for i in range(1, 4):
            layers.append(getattr(model_ft, f'layer{i}'))
        model_ft.layer4[0].downsample[0].stride = (1,1)
        model_ft.layer4[0].conv2.stride = (1,1)
        layers.append(model_ft.layer4)
                
        self.common_layers = nn.Sequential(*layers) 
    
    def build_opt(self, args):
        params = []
        for key, value in self.named_parameters():
            if not value.requires_grad:
                continue
            lr = args.lr
            weight_decay = args.weight_decay
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        if args.optim == 'SGD':
            optimizer = getattr(torch.optim, args.optim)(params, momentum=0.9)
        else:
            optimizer = getattr(torch.optim, args.optim)(params)
        return optimizer

    def forward(self, data):

        if self.training:
            x_opt = data['opt']['feat'] #[batch, C, H, W]
            x_sar = data['sar']['feat'] #[batch, C, H, W]
            trans_x1 = data['fil_opt']['feat']
            trans_x2 = data['fil_sar']['feat']
            
            B, C, H, W = x_opt.shape
            x = torch.cat([x_opt, trans_x1, x_sar, trans_x2], dim=0) #[opt, tran1, sar, trans2]
            x = self.common_layers(x)
            x = self.gap_layer(x)
            x_global = x.squeeze(-1).squeeze(-1)
            x_pool = self.latent_layer(x_global)
            # x_hash = self.hash_layer_train(x_pool)
            x_hash = self.l2norm(x_pool)
            x_global = self.l2norm(x_global)
            for i, item in enumerate(['opt', 'fil_opt', 'sar', 'fil_sar']):
                data[item]['out_feat'] = x_global[i*B:(i+1)*B]
                data[item]['out_vec'] = x_hash[i*B:(i+1)*B]
        else: #testing
            for mod in data:
                x = data[mod]['feat']
                x = self.common_layers(x)
                x = self.gap_layer(x)
                x_global = x.squeeze(-1).squeeze(-1)
                
                x_pool = self.latent_layer(x_global)
                # x_hash = self.hash_layer_test(x_pool)
                x_hash = self.l2norm(x_pool)
                x_global = self.l2norm(x_global)
                data[mod]['out_feat'] = x_global
                data[mod]['out_vec'] = x_hash
        return data

            
