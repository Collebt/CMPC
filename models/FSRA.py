import torch
import torch.nn as nn
from .vit import vit_small_patch16_224_FSRA
import torch.nn.functional as F
from models.model import Normalize, FeatureBlock
import torch.optim as optim



class Gem_heat(nn.Module):
    def __init__(self, dim = 768, p=3, eps=1e-6):
        super(Gem_heat, self).__init__()
        self.p = nn.Parameter(torch.ones(dim) * p)  # initial p
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)


    def gem(self, x, p=3):
        p = F.softmax(p).unsqueeze(-1)
        x = torch.matmul(x,p)
        x = x.view(x.size(0), x.size(1))
        return x

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.training:
            if self.return_f:
                f = x
                x = self.classifier(x)
                return x,f
            else:
                x = self.classifier(x)
                return x
        else:
            return x


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
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)







class FSRA_emb_net(nn.Module):
    def __init__(self, img_size, num_classes, camera_num=0,block = 4 ,return_f=False):
        super(FSRA_emb_net, self).__init__()
        self.return_f = return_f

        # small
        transformer_name = "vit_small_patch16_224_FSRA"
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(transformer_name))

        self.transformer = vit_small_patch16_224_FSRA(img_size=img_size, stride_size=[12, 12], drop_path_rate=0.1,
                                                        drop_rate= 0.0, attn_drop_rate=0.0)

        self.num_classes = num_classes

        self.classifier1 = ClassBlock(self.in_planes,num_classes,0.5,return_f=return_f)
        self.block = block
        for i in range(self.block):
            name = 'classifier_heat' + str(i+1)
            setattr(self, name, ClassBlock(768, num_classes, 0.5, return_f=self.return_f))


    def forward(self, x):
        features, all_features = self.transformer(x)

        tranformer_feature = self.classifier1(features[:,0])
        if self.block==1:
            return tranformer_feature

        part_features = features[:,1:]

        heat_result = self.get_heartmap_pool(part_features)
        y = self.part_classifier(self.block, heat_result, cls_name='classifier_heat')


        if self.training:
            y = y + [tranformer_feature]
            if self.return_f:
                cls, features = [], []
                for i in y:
                    cls.append(i[0])
                    features.append(i[1])
                return cls, features
            
            y = torch.stack(y, dim=-1)
            y = torch.sum(y, dim=-1)
        else:
            tranformer_feature = tranformer_feature.view(tranformer_feature.size(0),-1,1)
            y = torch.cat([y,tranformer_feature],dim=2)
            y = torch.sum(y, dim=-1)

        

        return y


    def get_heartmap_pool(self, part_features, add_global=False, otherbranch=False):
        heatmap = torch.mean(part_features,dim=-1)
        size = part_features.size(1)
        arg = torch.argsort(heatmap, dim=1, descending=True)
        x_sort = [part_features[i, arg[i], :] for i in range(part_features.size(0))]
        x_sort = torch.stack(x_sort, dim=0)

        split_each = size / self.block
        split_list = [int(split_each) for i in range(self.block - 1)]
        split_list.append(size - sum(split_list))
        split_x = x_sort.split(split_list, dim=1)

        split_list = [torch.mean(split, dim=1) for split in split_x]
        part_featuers_ = torch.stack(split_list, dim=2)
        if add_global:
            global_feat = torch.mean(part_features, dim=1).view(part_features.size(0), -1, 1).expand(-1, -1, self.block)
            part_featuers_ = part_featuers_ + global_feat
        if otherbranch:
            otherbranch_ = torch.mean(torch.stack(split_list[1:], dim=2), dim=-1)
            return part_featuers_, otherbranch_
        return part_featuers_

    def part_classifier(self,block, x, cls_name='classifier_lpn'):
        part = {}
        predict = {}
        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            # part[i] = torch.squeeze(x[:,:,i])
            name = cls_name + str(i+1)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(block):
            y.append(predict[i])
        if not self.training:
            # return torch.cat(y,dim=1)
            return torch.stack(y, dim=2)
        return y

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class two_view_net_FSRA(nn.Module):
    def __init__(self, args):

        class_num = args.class_num
        img_w, img_h = args.img_w, args.img_h
        block = 4
        return_f = False

        super(two_view_net_FSRA, self).__init__()
        self.model_1 = FSRA_emb_net(img_size = (img_w, img_h), num_classes=class_num, block=block,return_f=return_f)
        self.l2norm = Normalize(2)

    def forward(self, data):

        x1 = data['opt']['feat'] if 'opt' in data else None
        x2 = data['sar']['feat'] if 'sar' in data else None
        if x1 is None:
            y1 = None
        else:
            x1 = self.model_1(x1)
            data['opt']['out_feat'] = self.l2norm(x1)
            data['opt']['out_vec'] = self.l2norm(x1)  

        if x2 is None:
            y2 = None
        else:
            x2 = self.model_1(x2)
            # data['sar']['out_feat'] = y2
            data['sar']['out_feat'] = self.l2norm(x2)  
            data['sar']['out_vec'] = self.l2norm(x2)

        return data
    

    def build_opt(self, args):

        ignored_params = list(map(id, self.model_1.transformer.parameters()))
        extra_params = filter(lambda p: id(p) not in ignored_params, self.parameters())
        base_params = filter(lambda p: id(p) in ignored_params, self.parameters())
        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.3 * args.lr},
            {'params': extra_params, 'lr': args.lr}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)

        return optimizer
    