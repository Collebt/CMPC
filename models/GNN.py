import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn import init
from torch.optim import lr_scheduler

from torchvision import models
from losses import compute_dist_my, compute_dist

from models.GANet import ATEM

try:
    from torch_geometric.nn import GCNConv, GATConv
    from torch_geometric.data import Batch, Data
    from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse
except:
    pass
######################################################################


"""
2-class classification gnn will lead to imbalance of positive and negative samples, change to the 1D score-based prediction GNN.

"""

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
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Linear(channels[i - 1], channels[i]))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
                # layers.append(nn.InstanceNorm1d(channels[i]))
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



class embed_GNN(nn.Module):
    def __init__(self, args):
        super(embed_GNN, self).__init__()

        self.args = args

        low_dim = args.low_dim
        drop = args.drop
        arch = args.arch
        specify_num = args.specify_num

        self.graph_in = args.graph_in
        self.sigma = args.graphedge_sigma
        self.node_topk = args.node_topk
        self.edge_topk = args.edge_topk
        self.gnn_softmax = args.gnn_softmax
        self.node_func = args.node_func
        self.gnn_detach = args.gnn_detach

        self.end2end = args.end2end
        self.shortcut = args.shortcut
        
        pool_dim = 2048

        #build gnn
        self.train_refine = False
        self.edge_weight = args.edge_weight
        self.gnn = eval(args.graph_model)(args)

        # build network
        self.feature = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))

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
        ignored_params = list(map(id, self.feature.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, self.parameters())
        optimizer = optim.SGD([
            {'params': base_params, 'lr': args.lr},
            {'params': self.feature.parameters(), 'lr': args.lr_mlp}],
                weight_decay=5e-4, momentum=0.9, nesterov=True)
        return optimizer

    def build_graph(self, query_feat, ref_feat, ref_pos, gt_id=None):
        # data['opt']['feat'] 
        # data['sar']['feat']
        # data['opt']['pos'] edge weight

        #sim array (l2)
        # l2_dist = compute_dist(query_feat, ref_feat)
        
        sim_mat = torch.einsum('md,nd->mn', query_feat, ref_feat)
        # sim_mat = torch.softmax(similarity, dim=-1)
        if self.training: 
            if gt_id is None:  #mini_batch training
                feat_num = query_feat.shape[0]
                gt_id = torch.arange(0, query_feat.shape[0]).to(query_feat) #ground truth ID is aligned to samples
                sim_mat = sim_mat + torch.eye(feat_num).to(sim_mat)#set similarity of gt pairs to max
            else: #acces the whole training set
                feat_num = query_feat.shape[0]
                sim_mat[(torch.arange(feat_num) , gt_id)] += 1 #set similarity of gt pairs to max

        # sim_mat = torch.exp(-l2_dist)

        graph_list = []
        batch_pos = []

        for b, (sar_feat, sim) in enumerate(zip(query_feat, sim_mat)):

            # find GT + top-10(w/0 gt)
            _, top_idx = sim.topk(k=self.node_topk) #gt is the 1st index, because the gt has max similarity
            # gt_idx = torch.tensor([b]).to(top_idx)
            # node_idx = torch.cat([gt_idx, top_idx]) #gt is the 1st index
            node_idx = top_idx
            opt_feat = ref_feat[node_idx] #

            assert opt_feat.shape[0] == self.node_topk #

            #concatencate feature
            # node_feat = torch.cat(sar_feat.repeat(), opt_feat, dim=-1)
            # element product of feature
            # node_feat = opt_feat + sar_feat

            if self.node_func == 'product':
                node_feat = opt_feat * sar_feat
            elif self.node_func == 'sum':
                node_feat = opt_feat + sar_feat
            elif self.node_func == 'cat':
                node_feat = torch.cat((sar_feat.repeat(self.node_topk, 1), opt_feat), dim=-1)
            else:
                raise('node function is not provide!') 

            

            
            #build edge weight via reference GPS distance 
            pos_ingraph = ref_pos[node_idx] #[N,2]
            norm_pos = F.normalize(pos_ingraph, p=2)
            dist_mat = compute_dist_my(norm_pos, norm_pos, squred=False)

            #build edge weight via node feature 
            if self.args.KNN_inner:
                norm_feat = F.normalize(opt_feat, p=2) #[N,C]
                dist_mat = compute_dist_my(norm_feat, norm_feat, squred=False)

            # knn dist
            edge_weight = torch.exp(- dist_mat /  self.sigma)
            edge_weight = edge_weight - (edge_weight.diag()+0.1).diag()

            _, nn_idx = torch.topk(edge_weight, k=self.edge_topk, dim=-1)
            mask = torch.zeros(self.node_topk, self.node_topk).to(opt_feat)
            # build mask

            center_idx = torch.arange(0, self.node_topk).repeat(self.edge_topk, 1).transpose(1,0).to(nn_idx)
            edge_idx = torch.stack((center_idx, nn_idx), dim=0) # [2, edge_topk, node_num]
            edge_idx_flat = edge_idx.reshape(2,-1) #[2, edege_topk * node_num]

            #check if edge index is correct
            mask[(center_idx, nn_idx)] = edge_weight[(center_idx, nn_idx)]
            edge_attr = edge_weight[(center_idx, nn_idx)].reshape(-1,1)

            graph_list.append(Data(x=node_feat, y=(opt_feat * sar_feat).sum(-1), edge_attr=edge_attr, edge_index=edge_idx_flat, pos=norm_pos)) 

            batch_pos.append(norm_pos)

        #adjancancy
        batch_data = Batch.from_data_list(graph_list)
        batch_mask = to_dense_adj(batch_data.edge_index, batch_data.batch)
        batch_edge_attr = to_dense_adj(batch_data.edge_index, batch_data.batch,batch_data.edge_attr).squeeze()
        batch_pos = torch.stack(batch_pos, dim=0)

        return batch_data, batch_mask, batch_edge_attr, batch_pos, graph_list


    def forward(self, data):
        
        if not self.train_refine or not self.training: # only not run when not end2end in training
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

        if self.training and self.train_refine: #refine result via GNN
            batch_num = data['sar']['out_feat'].shape[0]
            gt_id = data['sar']['id'] if self.end2end is False else None
            
            graph, batch_adj, batch_edge_attr, batch_pos, graphs = self.build_graph(
                data['sar'][self.graph_in], 
                data['opt'][self.graph_in], 
                data['opt']['pos'], gt_id=gt_id)
            # input_node = graph.x
            if self.edge_weight is False: # edge_weight is only in 0 or 1
                batch_edge_attr = batch_adj

            batch_idx = torch.arange(0, batch_num).repeat_interleave(self.node_topk).long().cuda()
            x_batch = to_dense_batch(graph.x, batch_idx, fill_value=0)[0] # get batch feat [batch, node_num, dim]
            x_batch = x_batch.detach() if self.gnn_detach else x_batch
            score, node_feat = self.gnn(x_batch, batch_adj, batch_edge_attr, batch_pos)
            # graph.x_pred = graph_pred

            norm_node_feat = self.l2norm(node_feat) #[batch, C, N]
            sim_mat = torch.einsum('md,nd->mn', data['sar'][self.graph_in], data['opt'][self.graph_in])
            ori_scores_topk, _ = torch.topk(sim_mat, k=self.node_topk, dim=-1) #[M, N]
            refine_sim_mat = torch.einsum('bd, bnd->bn', data['sar'][self.graph_in], norm_node_feat)


            # softmax
            score = F.softmax(score, dim=-1) if self.gnn_softmax else score

            # shortcut
            if self.shortcut:
                score = score + ori_scores_topk
                refine_sim_mat = refine_sim_mat + ori_scores_topk

            graph.pred_mat = score
            graph.sim_mat = refine_sim_mat


            data.update(graph=graph)

        return data

    def refine_inference(self, query, ref_feat, ref_pos):
        """
            refine the coarsh-retrieved topk candidate from the whole database
            query: input query [Batch, channel]
            ref_feat: all reference feature [Number, Channel]
            ref_pos: all reference 2-d position [Number, 2]
            gt_id: the ground gruth id of query [Number]

            return: 
                graph: the output graph 
                refine_idx_in_topk: index along the top-K [Batch, K], e.g. [1,0,2,4,3] in top-5

        """
        batch_size = query.shape[0]
        graph_batch, batch_adj, batch_edge_attr, batch_pos, graph_list = self.build_graph(query, ref_feat, ref_pos)
        batch_idx = torch.arange(0, batch_size).repeat_interleave(self.node_topk).long().cuda()
        batch_x = to_dense_batch(graph_batch.x, batch_idx, fill_value=0)[0] # get batch feat [batch, node_num, dim]
        x_batch_pred, node_feat = self.gnn(batch_x, batch_adj, batch_edge_attr, batch_pos) 

        # use similarity as node score
        norm_node_feat = self.l2norm(node_feat)
        x_batch_pred = torch.einsum('bd, bnd->bn', query, norm_node_feat)


        #batch
        refine_idx_in_topk = torch.argsort(x_batch_pred, dim=-1, descending=True)

        return graph_batch, x_batch_pred, refine_idx_in_topk, graph_list
    



