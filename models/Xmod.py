import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from models.resnet import ResNet, BasicBlock, Bottleneck
import numpy as np
from models.model import Normalize
from pathlib import Path


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def my_weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.333)
        #nn.init.normal_(m.weight, mean=0.3, std=0.1)
        nn.init.constant_(m.bias, 0.0)
    if isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight, 0.333)
        nn.init.constant_(m.bias, 0.0)

class XmodNet(nn.Module):
    in_planes = 2048

    def __init__(self, args):
        super(XmodNet, self).__init__()

        num_classes = args.num_classes
        last_stride = args.last_stride
        model_path = Path.home() / args.model_path
        neck= args.neck
        model_name= args.arch
        pretrain_choice= args.pretrain_choice

        
        if model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride, block=Bottleneck, layers=[3, 4, 6, 3])
            # self.base = models.resnet50(pretrained=True)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')
            
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
    
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.generate1 = nn.Linear(3, 1)
        self.generate1.apply(my_weights_init)
        self.generate2 = nn.Linear(1, 3)
        self.generate2.apply(my_weights_init)

        self.l2norm = Normalize(2)
    

    def build_opt(self, args):
        params = []
        for key, value in self.named_parameters():
            if not value.requires_grad:
                continue
            lr = args.lr
            weight_decay = args.weight_decay
            if "bias" in key:
                lr = args.lr * args.lr_fastor
                weight_decay = args.weight_decay_bias
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        if args.optim == 'SGD':
            optimizer = getattr(torch.optim, args.optim)(params, momentum=0.9)
        else:
            optimizer = getattr(torch.optim, args.optim)(params)
        return optimizer


    def forward(self, data):
        """
            X-modality net for geerator the optical images. 
        """

        if self.training:

            x_opt = data['opt']['feat'] # [opt, sar]
            x_sar = data['sar']['feat']  #shape =[batch, H, W, channel]
            
            B, C, H, W = x_opt.shape
            #==================================================================================================================#
            gray1 = F.relu(self.generate1(x_opt.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)))
            gray1 = self.generate2(gray1).view(B, -1, 3).permute(0,2,1).contiguous().view(B,3,H,W)
            x = torch.cat((x_sar, x_opt, gray1),dim=0)
            #==================================================================================================================#
        else:
            mod = 'opt' if 'opt' in data else 'sar'
            x = data[mod]['feat']

        global_feat = self.gap(self.base(x))

        global_feat = global_feat.view(global_feat.shape[0], -1)
        feat_tri = self.l2norm(global_feat)

        feat_infer = self.l2norm(self.bottleneck(global_feat))


        if self.training:
            cls_score = self.classifier(global_feat)
            for i, item in enumerate(['sar', 'opt',  'X']):
                if item in data.keys():
                    data[item].update(out_feat=feat_infer[i*B:(i+1)*B],
                                  out_vec = feat_tri[i*B:(i+1)*B],
                                  out_cls = cls_score[i*B:(i+1)*B])
                else:
                    data[item] = dict(out_feat=feat_infer[i*B:(i+1)*B],
                                  out_vec = feat_tri[i*B:(i+1)*B],
                                  out_cls = cls_score[i*B:(i+1)*B])
        else:
            data[mod]['out_feat'] = feat_infer
            data[mod]['out_vec'] = feat_tri
        return data



    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self[i].copy_(param_dict[i])
            #self.state_dict()[i].copy_(param_dict[i])