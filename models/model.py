import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler

#
######################################################################
class WDiscriminator(nn.Module):
	def __init__(self, hidden_size, hidden_size2=512):
		super(WDiscriminator, self).__init__()
		self.hidden = torch.nn.Linear(hidden_size, hidden_size2)
		self.hidden2 = torch.nn.Linear(hidden_size2, hidden_size2)
		self.output = torch.nn.Linear(hidden_size2, 1)
	def forward(self, input_embd):
        # return F.sigmoid(self.output(F.leaky_relu(self.hidden2(F.leaky_relu(self.hidden(input_embd), 0.2, inplace=True)))))
		return self.output(F.leaky_relu(self.hidden2(F.leaky_relu(self.hidden(input_embd), 0.2, inplace=True))))
      
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(-1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)]
        feat_block += [nn.BatchNorm1d(low_dim)]

        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block

    def forward(self, x):
        x = self.feat_block(x)
        return x


def MLP(channels: list, do_bn=True): 
    """ Multi-layer perceptron 
    input : [batch, channel, number_feat]
    """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)



# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=False, relu=False, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        # add_block += [nn.Linear(input_dim, num_bottleneck)]
        num_bottleneck = input_dim
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        f = self.add_block(x)
        f_norm = f.norm(p=2, dim=1, keepdim=True) + 1e-8
        f = f.div(f_norm)
        x = self.classifier(f)
        return x, f

def gem(x, p = 3.0):
    b, c, h, w = x.shape
    x = x.view(b, c, -1)
    x_pool = (torch.mean(x ** p, dim=-1) + 1e-12) ** (1 / p)
    return x_pool

# Define the ResNet50-based Model
class opt_resnet(nn.Module):
    def __init__(self, specify_num=3):
        super(opt_resnet, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        layers = []
        for i in range(specify_num):
            if i == 0: #the 1st layer is conv
                layers.extend([model_ft.conv1, model_ft.bn1, model_ft.relu, model_ft.maxpool])
            elif i ==4: # remove the final downsample
                model_ft.layer4[0].downsample[0].stride = (1,1)
                model_ft.layer4[0].conv2.stride = (1,1)
                layers.append(model_ft.layer4)
            else:
                layers.append(getattr(model_ft, f'layer{i}'))
        self.specify_layers = nn.Sequential(*layers) #if specify_num!=0 else lambda x: x

        # remove the final downsample
        # self.model.layer4[0].downsample[0].stride = (1,1)
        # self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
            
        return self.specify_layers(x)

class sar_resnet(nn.Module):
    def __init__(self, specify_num=3):
        super(sar_resnet, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.model = model_ft
        layers = []
        for i in range(specify_num):
            if i == 0: #the 1st layer is conv
                layers.extend([model_ft.conv1, model_ft.bn1, model_ft.relu, model_ft.maxpool])
            elif i ==4: # remove the final downsample
                model_ft.layer4[0].downsample[0].stride = (1,1)
                model_ft.layer4[0].conv2.stride = (1,1)
                layers.append(model_ft.layer4)
            else:
                layers.append(getattr(model_ft, f'layer{i}'))
        self.specify_layers = nn.Sequential(*layers) #if specify_num!=0 else lambda x: x
        # remove the final downsample
        # self.model.layer4[0].downsample[0].stride = (1,1)
        # self.model.layer4[0].conv2.stride = (1,1)
    
    def forward(self, x):
        return self.specify_layers(x)

class embed_share_net(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.l2norm = Normalize(2)
        model_ft = models.resnet50(pretrained=True)
        layers = []
        for i in range(0, 5):
            if i == 0:
                layers.extend([model_ft.conv1, model_ft.bn1, model_ft.relu, model_ft.maxpool])
            elif i ==4:
                model_ft.layer4[0].downsample[0].stride = (1,1)
                model_ft.layer4[0].conv2.stride = (1,1)
                layers.append(model_ft.layer4)
            else:
                layers.append(getattr(model_ft, f'layer{i}'))
        self.common_layers = nn.Sequential(*layers) 
        self.batch_size = args.batch_szie

    def forward(self, data):
        x = torch.cat([data[k]['feat']  for k in data], dim=0)
        x = self.common_layers(x)
        x = self.avgpool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.l2norm(x)
        for i, k in enumerate(data):
            data[k]['out_feat'] = x[i*self.batch_size:(i+1)*self.batch_size]
        return data

        

    def build_opt(self, args):
        optimizer = optim.SGD(self.parameters(), lr=args.lr,weight_decay=5e-4, momentum=0.9, nesterov=True)
        return optimizer
    

class embed_net_my(nn.Module):
    def __init__(self, args):
        super(embed_net_my, self).__init__()

        low_dim = args.low_dim
        drop = args.drop
        arch = args.arch
        specify_num = args.specify_num

        pool_dim = 2048
        self.feature = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature1 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature2 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        # self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.l2norm = Normalize(2)
        # self.bn = nn.BatchNorm1d(low_dim)

        self.opt_net = opt_resnet(specify_num=specify_num)
        self.sar_net = sar_resnet(specify_num=specify_num)
        self.modal_net = {'opt':self.opt_net, 'sar':self.sar_net}
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        layers = []
        for i in range(specify_num, 5):
            if i == 0:
                layers.extend([model_ft.conv1, model_ft.bn1, model_ft.relu, model_ft.maxpool])
            elif i ==4:
                model_ft.layer4[0].downsample[0].stride = (1,1)
                model_ft.layer4[0].conv2.stride = (1,1)
                layers.append(model_ft.layer4)
            else:
                layers.append(getattr(model_ft, f'layer{i}'))
        self.common_layers = nn.Sequential(*layers) 

    def build_opt(self, args):
        ignored_params = list(map(id, self.feature1.parameters())) + \
                 list(map(id, self.feature2.parameters())) + \
                 list(map(id, self.feature.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, self.parameters())
        optimizer = optim.SGD([
            {'params': base_params, 'lr': args.lr},
            {'params': self.feature.parameters(), 'lr': args.lr_mlp},
            {'params': self.feature1.parameters(), 'lr': args.lr_mlp},
            {'params': self.feature2.parameters(), 'lr': args.lr_mlp}],
                weight_decay=5e-4, momentum=0.9, nesterov=True)
        return optimizer

    def forward(self, data):
        # batch_size = x1.shape[0]
        for key in self.modal_net:
            # concatnate feat in each modality
            valid_i = [i for i in data if key in i]
            if len(valid_i) == 0:
                continue
            x = torch.cat([data[i]['feat']  for i in valid_i], dim=0)
            x = self.modal_net[key](x)

            batch_num = data[valid_i[0]]['feat'].shape[0]
            for i, k in enumerate(valid_i): # separate feature
                data[k]['emb_feat'] = x[i*batch_num:(i+1)*batch_num]

        x = torch.cat([data[k]['emb_feat']  for k in data], dim=0)
        x = self.common_layers(x)
        x = self.avgpool(x)
        x = x.squeeze(-1).squeeze(-1)
        # x = self.l2norm(x) #multi l2norm
        x_pool = self.feature(x)
        x_pool = self.l2norm(x_pool)
        x = self.l2norm(x)
        for i, k in enumerate(data):
            data[k]['out_feat'] = x[i*batch_num:(i+1)*batch_num]
            data[k]['out_vec'] = x_pool[i*batch_num:(i+1)*batch_num]
        return data

class embed_net_MLP(nn.Module):
    def __init__(self, args):
        super(embed_net_MLP, self).__init__()

        low_dim = args.low_dim
        drop = args.drop
        arch = args.arch
        specify_num = args.specify_num
        
        pool_dim = 2048
        # self.feature = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature = MLP(channels=[2048, 1024, 512])
        self.feature1 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature2 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        # self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.l2norm = Normalize(2)
        # self.bn = nn.BatchNorm1d(low_dim)

        self.opt_net = opt_resnet(specify_num=specify_num)
        self.sar_net = sar_resnet(specify_num=specify_num)
        self.modal_net = {'opt':self.opt_net, 'sar':self.sar_net}
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        layers = []
        for i in range(specify_num, 5):
            if i == 0:
                layers.extend([model_ft.conv1, model_ft.bn1, model_ft.relu, model_ft.maxpool])
            elif i ==4:
                model_ft.layer4[0].downsample[0].stride = (1,1)
                model_ft.layer4[0].conv2.stride = (1,1)
                layers.append(model_ft.layer4)
            else:
                layers.append(getattr(model_ft, f'layer{i}'))
        self.common_layers = nn.Sequential(*layers) 

    def build_opt(self, args):
        ignored_params = list(map(id, self.feature1.parameters())) + \
                 list(map(id, self.feature2.parameters())) + \
                 list(map(id, self.feature.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, self.parameters())
        optimizer = optim.SGD([
            {'params': base_params, 'lr': args.lr},
            {'params': self.feature.parameters(), 'lr': args.lr_mlp},
            {'params': self.feature1.parameters(), 'lr': args.lr_mlp},
            {'params': self.feature2.parameters(), 'lr': args.lr_mlp}],
            weight_decay=5e-4, momentum=0.9, nesterov=True)

        return optimizer

    def forward(self, data):
        # batch_size = x1.shape[0]
        for key in self.modal_net:
            # concatnate feat in each modality
            valid_i = [i for i in data if key in i]
            if len(valid_i) == 0:
                continue
            x = torch.cat([data[i]['feat']  for i in valid_i], dim=0)
            x = self.modal_net[key](x)

            batch_num = data[valid_i[0]]['feat'].shape[0]
            for i, k in enumerate(valid_i): # separate feature
                data[k]['emb_feat'] = x[i*batch_num:(i+1)*batch_num]

        x = torch.cat([data[k]['emb_feat']  for k in data], dim=0)
        x = self.common_layers(x)
        x = self.avgpool(x)
        x = x.squeeze(-1).squeeze(-1)
        x_pool = self.feature(x)
        x_pool = self.l2norm(x_pool)
        x = self.l2norm(x)
        for i, k in enumerate(data):
            data[k]['out_feat'] = x[i*batch_num:(i+1)*batch_num]
            data[k]['out_vec'] = x_pool[i*batch_num:(i+1)*batch_num]
        return data




class embed_net(nn.Module):
    def __init__(self, low_dim, drop=0.5, arch='resnet50'):
        super(embed_net, self).__init__()
        self.opt_net = opt_resnet()
        self.sar_net = sar_resnet()
        pool_dim = 2048
        self.feature = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature1 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature2 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.model = models.resnet50(pretrained=True)
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.l2norm = Normalize(2)
        self.bn = nn.BatchNorm1d(low_dim)

    def forward(self, x1, x2, x3, x4, modal=0):
        batch_size = x1.shape[0]
        x1 = torch.cat((x1, x3), 0)
        x2 = torch.cat((x2, x4), 0)
        x1 = self.opt_net(x1)
        x2 = self.sar_net(x2)
        x = torch.cat((x1, x2), dim=0)

        #layer0
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        # x = self.model.layer1(x)
        x = self.model.layer1(x) #layer1
        x = self.model.layer2(x) #layer2
        x = self.model.layer3(x) #layer3
        x = self.model.layer4(x) #layer4

        x = self.avgpool(x)
        x = torch.squeeze(x)
        opt_feat = x[:batch_size]
        opt_feat_filted = x[batch_size : 2 * batch_size]
        sar_feat = x[2 * batch_size : 3 * batch_size]
        sar_feat_filted = x[3 * batch_size:]

        opt_feat_pool = self.feature(opt_feat)
        sar_feat_pool = self.feature(sar_feat)
        opt_filted_pool = self.feature(opt_feat_filted)
        sar_filted_pool = self.feature(sar_feat_filted)

        if modal == 0:
            return self.l2norm(opt_feat_pool), self.l2norm(sar_feat_pool), self.l2norm(opt_filted_pool), self.l2norm(sar_filted_pool)

        elif modal == 1:
            return self.l2norm(opt_feat), self.l2norm(opt_feat_pool)

        else:
            return self.l2norm(sar_feat), self.l2norm(sar_feat_pool)
        
class Discriminator(nn.Module):
    def __init__(self, input_dim = 2048, class_num = 2, dropout=0.5):
        super(Discriminator, self).__init__()
        classifier = []
        # classifier += [nn.Linear(input_dim, input_dim)]
        # classifier += [nn.LeakyReLU(0.1)]
        # classifier += [nn.Dropout(p=dropout)]
        classifier += [nn.Linear(input_dim, input_dim)]
        classifier += [nn.BatchNorm1d(input_dim)]
        classifier += [nn.LeakyReLU(0.1)]
        classifier += [nn.Dropout(p=dropout)]

        classifier += [nn.Linear(input_dim, input_dim)]
        classifier += [nn.BatchNorm1d(input_dim)]
        classifier += [nn.LeakyReLU(0.1)]
        classifier += [nn.Dropout(p=dropout)]

        classifier += [nn.Linear(input_dim, class_num)] #no softmax

        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier

    def forward(self, x):
        x = self.classifier(x)
        x = F.softmax(x, dim=-1) # add softmax for classifier
        return x

class embed_net4CAM(nn.Module):
    def __init__(self, low_dim, drop=0.5, arch='resnet50', specify_num=3):
        super(embed_net_my, self).__init__()
        
        pool_dim = 2048
        self.feature = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature1 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature2 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        # self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.l2norm = Normalize(2)
        # self.bn = nn.BatchNorm1d(low_dim)

        self.opt_net = opt_resnet(specify_num=specify_num)
        self.sar_net = sar_resnet(specify_num=specify_num)
        self.modal_net = {'opt':self.opt_net, 'sar':self.sar_net}
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        layers = []
        for i in range(specify_num, 5):
            if i == 0:
                layers.extend([model_ft.conv1, model_ft.bn1, model_ft.relu, model_ft.maxpool])
            elif i == 4:
                model_ft.layer4[0].downsample[0].stride = (1,1)
                model_ft.layer4[0].conv2.stride = (1,1)
                layers.append(model_ft.layer4)
            else:
                layers.append(getattr(model_ft, f'layer{i}'))
        self.common_layers = nn.Sequential(*layers) 

    def forward(self, data):
        # batch_size = x1.shape[0]
        for key in self.modal_net:
            # concatnate feat in each modality
            valid_i = [i for i in data if key in i]
            if len(valid_i) == 0:
                continue
            x = torch.cat([data[i]['feat']  for i in valid_i], dim=0)
            x = self.modal_net[key](x)

            batch_num = data[valid_i[0]]['feat'].shape[0]
            for i, k in enumerate(valid_i): # separate feature
                data[k]['emb_feat'] = x[i*batch_num:(i+1)*batch_num]

        x = torch.cat([data[k]['emb_feat']  for k in data], dim=0)
        x = self.common_layers(x)
        x = self.avgpool(x)
        x = x.squeeze(-1).squeeze(-1)
        x_pool = self.feature(x)
        x_pool = self.l2norm(x_pool)
        x = self.l2norm(x)
        for i, k in enumerate(data):
            data[k]['out_feat'] = x[i*batch_num:(i+1)*batch_num]
            data[k]['out_vec'] = x_pool[i*batch_num:(i+1)*batch_num]
        return x


