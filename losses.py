import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from utils.visual import wd_vis, cls_vis, tri_vis
import matplotlib.pyplot as plt

from models.model import Discriminator, FeatureBlock, Normalize, WDiscriminator

def compute_dist(opt_feat_embed, sar_feat_embed, squred=True):
    # get the distance matrix between optical features and sar features
    m, n = opt_feat_embed.shape[0], sar_feat_embed.shape[0]

    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    emb1_pow = torch.pow(opt_feat_embed, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(sar_feat_embed, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # mat_sum = torch.einsum('md,nd->mn', opt_feat_embed,sar_feat_embed )
    # dist_mat = dist_mat.addmm(1, -2, opt_feat_embed, sar_feat_embed.t()) #ori admm
    dist_mat = emb1_pow + emb2_pow - 2*(opt_feat_embed @ sar_feat_embed.t())

    dist_mat[dist_mat < 0] = 0
    if not squred:
        dist_mat = dist_mat.clamp(min=1e-12).sqrt()
    return dist_mat


def compute_dist_my(src_feat, tgt_feat, squred=True):
    """
    L2 distance between two features,
    
    """
    # get the distance matrix between optical features and sar features
    m, n, d = src_feat.shape[0], tgt_feat.shape[0], src_feat.shape[-1]

    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    emb1_pow = src_feat.unsqueeze(1).expand(m, n, d)
    emb2_pow = tgt_feat.unsqueeze(0).expand(m, n, d)
    dist_mat = (emb1_pow - emb2_pow).pow(2).sum(-1)
    if not squred:
        dist_mat = dist_mat.clamp(min=1e-12).sqrt()
    return dist_mat



class Triplet_Hard_Mining(nn.Module):
    def __init__(self, margin = 0.3):
        super(Triplet_Hard_Mining, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin = margin)

    def forward(self, src_feat, tgt_feat, labels, recorder=None):

        # dist_mat = 2 - 2 * torch.einsum('nd, md->nm', src_feat, tgt_feat)
        dist_mat = compute_dist(src_feat, tgt_feat)
        n = src_feat.shape[0]
        mask = labels.expand(n, n).eq(labels.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist_mat[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist_mat[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        if recorder is not None: #reocrd
            tri_data = recorder.get('tri_vis')
            tri_data['feat'].append(torch.stack([dist_ap, dist_an]))
            recorder.save(tri_vis=tri_data)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        # accuracy = torch.ge(dist_an, dist_ap).sum() / n
        # recorder.update('acc_tri', accuracy.item(), n, type='%')
        return loss

def triplet_loss(data, criterion_tri, feat_in, self_mode=False, recorder=None):
    """
    compute the loss between different modality (not include self domain feature)
    """"""
    Note: 
    supervise the  "out_vec" is much better than supervise the "out_feat"
    """
    loss = 0.
    if self_mode:
        for key_i in data:
            if 'opt' in key_i or 'sar' in key_i:
                for key_j in data:
                    if 'opt' in key_j or 'sar' in key_j:
                        loss = loss + criterion_tri(data[key_i][feat_in], data[key_j][feat_in], data[key_i]['id'], recorder) 
    else:
        for key_i in data:
            if 'opt' in key_i or 'sar' in key_i:
                for key_j in data:
                    if 'opt' in key_j or 'sar' in key_j:
                        loss = loss + criterion_tri(data[key_i][feat_in], data[key_j][feat_in], data[key_i]['id'], recorder ) if key_i != key_j else loss
    return loss


class Project_triplet(nn.Module):
    def __init__(self, args, recorder, device) -> None:
        super().__init__()
        self.criterion_tri = Triplet_Hard_Mining(margin=0.5)
        self.self_mode = args.self_mode
        self.projector = FeatureBlock(2048, args.low_dim, dropout=args.drop)
        self.optimize = optim.SGD(self.projector.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
        self.l2norm = Normalize(2)
        self.recorder = recorder

 

    def __call__(self, data, args):
        vec_sar = self.projector(data['sar']['out_feat'])
        vec_opt = self.projector(data['opt']['out_feat'])
        pass


class Triplet_Loss(nn.Module):

    def __init__(self, args, recorder, device) -> None:
        super().__init__()
        self.criterion_tri = Triplet_Hard_Mining(margin=0.5)
        self.self_mode = args.self_mode

        self.output_path = f'{args.result_path}/{args.exp_name}'
        self.record_n = 0
        self.recorder = recorder
        self.recorder.register_vis(tri_vis)
        

    def __call__(self, data, args=None) :

        if self.record_n % 962 == 0: #clean record per 1 epoch
            record_data = dict(name=f'{self.output_path}/tri{self.record_n}', feat=[]) #init record distance
            self.recorder.save(tri_vis=record_data)
        recorder = self.recorder if self.record_n % 100 == 0 else None #get recorder per 100 iteration
        self.record_n += 1
        

        loss = triplet_loss(data, self.criterion_tri, feat_in=args.tri_in, self_mode=self.self_mode, recorder=recorder)
        
        sim_mat = torch.einsum('md, nd-> mn', data['sar'][args.tri_in], data['opt'][args.tri_in])
        retri_idx = sim_mat.argmax(-1)
        acc = (retri_idx == torch.arange(0, retri_idx.shape[0]).to(retri_idx)).sum() / args.batch_size
        self.recorder.update('loss_tri', loss.item(), args.batch_size, type='f')
        self.recorder.update('acc_tri', acc.item(), args.batch_size, type='%')
        return loss 

class Hash_Triplet(nn.Module):
    def __init__(self, args, recorder, device) -> None:
        super().__init__()
        self.criterion_tri = Triplet_Hard_Mining(margin=0.5)
        self.output_path = f'{args.result_path}/{args.exp_name}'
        self.record_n = 0
        self.recorder = recorder
        self.recorder.register_vis(tri_vis)
        

    def __call__(self, data, args=None) :

        if self.record_n % 962 == 0: #clean record per 1 epoch
            record_data = dict(name=f'{self.output_path}/tri{self.record_n}', feat=[]) #init record distance
            self.recorder.save(tri_vis=record_data)
        recorder = self.recorder if self.record_n % 100 == 0 else None #get recorder per 100 iteration
        self.record_n += 1

        loss_tri = 0
        loss_reg = 0
        loss_bal = 0
        for mod in data:
            gt_id = data[mod]['id']
            ori_f, tran_f = data[mod]['out_feat'], data[mod]['out_feat_tran']
            loss_tri = loss_tri + self.criterion_tri(ori_f, tran_f, gt_id)
            loss_reg = loss_reg - torch.mean(((ori_f-0.5)**2).sum(-1)) - torch.mean(((tran_f-0.5)**2).sum(dim=-1))
            loss_bal = loss_bal + torch.mean((ori_f.mean(-1) - 0.5)**2) + torch.mean((tran_f.mean(-1) - 0.5)**2)

        loss = loss_tri + args.beta * loss_reg + args.gamma * loss_bal
        
        sim_mat = torch.einsum('md, nd-> mn', data['sar'][args.tri_in], data['opt'][args.tri_in])
        retri_idx = sim_mat.argmax(-1)
        acc = (retri_idx == torch.arange(0, retri_idx.shape[0]).to(retri_idx)).sum() / args.batch_size
        self.recorder.update('loss_tri', loss_tri.item(), args.batch_size, type='f')
        self.recorder.update('loss_reg', loss_reg.item(), args.batch_size, type='f')
        self.recorder.update('loss_bal', loss_bal.item(), args.batch_size, type='f')
        self.recorder.update('acc_tri', acc.item(), args.batch_size, type='%')
        return loss 



class Xmod_Loss(nn.Module):
    """
    CMG: [cross modality gap]
        Triplet between SAR-X, SAR-RGB,
    MRG: [modality respective gap]

        Class: idfentify the ID label of patches,
        Triplet [X] :(there is no sample in the same modality, there the triplet within modality is set as 0)

    """
    def __init__(self, args, recorder, device) -> None:
        super().__init__()
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_tri = Triplet_Hard_Mining(margin=0.5)

        self.output_path = f'{args.result_path}/{args.exp_name}'
        self.recorder = recorder
        self.tri_in = args.tri_in
        

    def __call__(self, data, args=None) :

        ##### cross-modal triplet loss######
        gt_id = data['sar']['id']
        sar_feat, opt_feat, x_feat = data['sar'][self.tri_in], data['opt'][self.tri_in], data['X'][self.tri_in]
        loss1 = self.criterion_tri(sar_feat, opt_feat, gt_id)
        loss2 = self.criterion_tri(sar_feat, x_feat, gt_id)
        loss_tri = loss1 + loss2
        ###### class identify ######
        sar_score, opt_score, x_score = data['sar']['out_cls'], data['opt']['out_cls'], data['X']['out_cls']
        
        loss_cls = self.criterion_cls(sar_score, gt_id) + self.criterion_cls(opt_score, gt_id) + self.criterion_cls(x_score, gt_id)

        loss = loss_tri + args.cls_lambda * loss_cls
        
        self.recorder.update('loss_tri', loss_tri.item(), args.batch_size, type='f')
        self.recorder.update('loss_cls', loss_cls.item(), args.batch_size, type='f')
        return loss 



class Adversarial_Loss(nn.Module):

    def __init__(self, args, recorder, device) -> None:
        super().__init__()
        # self.criterion_tri = Triplet_Hard_Mining(margin=0.5)
        in_dim = args.feat_dim if 'feat' in args.D_input else args.low_dim
        self.discriminator = Discriminator(input_dim=in_dim)
        self.criterion_cls = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999)) #optimize for disriminator
        
        self.discriminator.to(device)
        self.criterion_cls.to(device)
        self.lambda_cls = args.lambda_cls

        self.D_input = args.D_input

        self.self_mode = args.self_mode
        self.recorder = recorder
        self.recorder.register_vis(cls_vis)
        self.record_n = 0
        self.output_path = f'{args.result_path}/{args.exp_name}'

    
    def train_discriminator(self, data):
        batch_size = data['opt'][self.D_input].shape[0]
        data['opt']['mod_label'] = torch.zeros(batch_size).cuda().long()
        data['sar']['mod_label'] = torch.ones(batch_size).cuda().long()
        if len(data) == 4:
            data['fil_opt']['mod_label'] = torch.zeros(batch_size).cuda().long()
            data['fil_sar']['mod_label'] = torch.ones(batch_size).cuda().long()
        
        loss = 0.
        correct = 0.
        feat_list = []
        for key in['sar', 'opt']:
            item = data[key]
            cls_res = self.discriminator(item[self.D_input].detach())
            loss = loss + self.criterion_cls(cls_res, item['mod_label'])

            _, predicted = cls_res.max(1)
            correct = correct + predicted.eq(item['mod_label']).sum().item()
            if key == 'opt' or key == 'sar':
                feat_list.append(cls_res)
        self.recorder.save(wd=dict(name=f'{self.output_path}/cls{self.record_n}', feat=feat_list))
        self.record_n += 1

        loss_avg = loss/len(data) #avgerage of modal feature
        accuracy_D = correct/batch_size/len(data)  #classify predict accuracy of discriminator

        self.optimizer.zero_grad()
        loss_avg.backward()
        self.optimizer.step()

        return loss_avg , accuracy_D

    def train_G_iter(self, data):
        loss_cls = 0
        for key in ['sar', 'opt']:
            cls_res = self.discriminator(data[key][self.D_input])
            reverse_label = 1 - data[key]['mod_label']
            loss_cls = loss_cls + self.criterion_cls(cls_res, reverse_label)

        return loss_cls


    def __call__(self, data, args):
        # loss = triplet_loss(data, self.criterion_tri, self_mode=self.self_mode)
        loss_D, cls_acc_D = self.train_discriminator(data)
        loss_cls = self.train_G_iter(data)
        loss = self.lambda_cls * loss_cls #update loss

        #record loss & acc
        # recorder.update('loss_tri',loss.item(), args.batch_size)
        self.recorder.update('acc_D', cls_acc_D, args.batch_size, type='%')
        self.recorder.update('loss_D', loss_D.item(), args.batch_size, type='f')
        self.recorder.update('loss_G', loss.item(), args.batch_size, type='f')

        return loss  






class Wasserstein_Loss(nn.Module):

    def __init__(self, args, recorder, device) -> None:
        super().__init__()
        # self.criterion_tri = Triplet_Hard_Mining(margin=0.5)
        in_dim = args.feat_dim if 'feat' in args.D_input else args.low_dim
        self.discriminator = WDiscriminator(in_dim) # 2048 for out_feat, 512 for out_vec 3 layers [2048, 512, 1] 
        self.opt_wd = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr_d, weight_decay=5e-4)
        
        self.discriminator.to(device)
        self.lambda_cls = args.lambda_cls
        self.self_mode = args.self_mode
        self.recorder = recorder
        self.recorder.register_vis(wd_vis)
        self.record_n = 0
        self.D_epoch = args.D_epoch
        self.loss_weight = args.lambda_cls
        self.output_path = f'{args.result_path}/{args.exp_name}'
        self.batch_size = args.batch_size


    def __call__(self, data, args):
        embd0 = data['sar'][args.D_input] #
        embd1 = data['opt'][args.D_input]
        anchor_size = embd1.size(0) // 2 #original: embd1.size(0)
        self.discriminator.train()
        record_feat = []

        for j in range(self.D_epoch):
            w0 = self.discriminator(embd0)
            w1 = self.discriminator(embd1)
            anchor1 = w1.view(-1).argsort(descending=True)[: anchor_size]
            anchor0 = w0.view(-1).argsort(descending=False)[: anchor_size]
            embd0_anchor = embd0[anchor0, :].clone().detach()
            embd1_anchor = embd1[anchor1, :].clone().detach()
            self.opt_wd.zero_grad()
            loss = -torch.mean(self.discriminator(embd0_anchor)) + torch.mean(self.discriminator(embd1_anchor))
            loss.backward()
            self.opt_wd.step()
            for p in self.discriminator.parameters():
                p.data.clamp_(-0.1, 0.1)
            if j % (self.D_epoch//3+1) == 0:
                record_feat.append([w0, w1])
        record_feat.append([w0, w1])
        self.recorder.update("loss_D", loss.item(), self.batch_size, type='f')
        self.recorder.save(wd=dict(name=f'{self.output_path}/wd_vis{self.record_n}', feat=record_feat))
        self.record_n += 1
        w0 = self.discriminator(embd0)
        w1 = self.discriminator(embd1)
        anchor1 = w1.view(-1).argsort(descending=True)[: anchor_size]
        anchor0 = w0.view(-1).argsort(descending=False)[: anchor_size]
        embd0_anchor = embd0[anchor0, :]
        embd1_anchor = embd1[anchor1, :]
        loss = -torch.mean(self.discriminator(embd1_anchor))
        output = self.loss_weight * loss
        self.recorder.update("loss_G", output.item(), args.batch_size, type='f')
        return output


class Instance_Loss(nn.Module):
    def __init__(self, args, recorder, device) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = args.batch_size

    def __call__(self, data, args):
        loss = 0
        for key, item in data.items():
            pred = item['out_vec']
            gt_idx = item['id']
            loss = loss + self.criterion(pred, gt_idx)

        return loss

class Hybrid_Loss(nn.Module):
    """
        VIGOR hybrid loss including triplet loss, IOU_loss, adn offset loss
    """
    def __init__(self, args, recorder, device) -> None:
        super().__init__()
        self.criterion_tri = Triplet_Hard_Mining(margin=0.5)
        self.recorder = recorder

    def __call__(self, data, args) :
        '''
        修改用out_vec作为输出监督loss，用out_feat的效果不太好。'''
        # loss_tri = triplet_loss(data, self.criterion_tri, self_mode=self.self_mode)
        feat = args.tri_in
        sar_feat, opt_feat, semi_feat = data['sar'][feat], data['opt'][feat], data['semi_opt'][feat]

        loss1 = self.criterion_tri(sar_feat, opt_feat,  data['sar']['id'])
        loss2 = self.criterion_tri(sar_feat, semi_feat, data['sar']['id'])
        sim_1 = (sar_feat * opt_feat).sum(-1)
        sim_2 = (sar_feat * semi_feat).sum(-1)

        loss_tri = loss1 + loss2
        
        delta_1 = data['opt']['pos'] - data['sar']['pos']
        delta_2 = data['semi_opt']['pos'] - data['sar']['pos']
        ratio = distance_score(delta_1, delta_2, L=200)

        error = (sim_2/sim_1) - ratio
        
        loss_3 = torch.mean(error*error) #/10.
        loss = loss_tri + args.IOU_weight * loss_3

        self.recorder.update('loss_tri', loss_tri.item(), args.batch_size)
        self.recorder.update('loss_IOU', loss_3.item(), args.batch_size)

        return loss 

def distance_score(delta_1, delta_2, mode = 'IOU', L=640.):
    if mode == 'distance':
        distance_1 = torch.sqrt(delta_1[:, 0] * delta_1[:, 0] + delta_1[:, 1] * delta_1[:, 1])
        distance_2 = torch.sqrt(delta_2[:, 0] * delta_2[:, 0] + delta_2[:, 1] * delta_2[:, 1])
        ratio = distance_1 / distance_2
    elif mode == 'IOU':
        IOU_1 = 1. / (1. - (1 - torch.abs(delta_1[:, 0]) / L) * (1. - torch.abs(delta_1[:, 1]) / L) / 2.) - 1
        IOU_2 = 1. / (1. - (1 - torch.abs(delta_2[:, 0]) / L) * (1. - torch.abs(delta_2[:, 1]) / L) / 2.) - 1
        ratio = IOU_2/ IOU_1
    return ratio



def compute_loss(sat_global, grd_global, batch_size = 14, loss_weight = 10.0):
    '''
    Compute the weighted soft-margin triplet loss
    :param grd_global: the ground image global descriptor
    :param batch_hard_count: the number of top hard pairs within a batch. If 0, no in-batch hard negative mining
    :return: the loss
    '''

    # dist_array = 2 - 2 * torch.einsum('nd, md->nm', sat_global, grd_global)
    dist_array = compute_dist(sat_global, grd_global)
    pos_dist = dist_array.diag() #bn (n=m)

    
    pair_n = batch_size * (batch_size - 1.0)

    # ground to satellite
    triplet_dist_g2s = pos_dist.unsqueeze(1) - dist_array
    loss_g2s = torch.sum(torch.log(1 + torch.exp(triplet_dist_g2s * loss_weight))) / pair_n

    # satellite to ground
    triplet_dist_s2g = pos_dist.unsqueeze(0) - dist_array
    loss_s2g = torch.sum(torch.log(1 + torch.exp(triplet_dist_s2g * loss_weight))) / pair_n

    loss = (loss_g2s + loss_s2g) / 2.0
    return loss

class Graph_cls_loss(nn.Module):

    def __init__(self, args, recorder, device) -> None:
        super(Graph_cls_loss, self).__init__()
        self.criterion_cls = nn.CrossEntropyLoss()

        # self.output_path = f'{args.result_path}/{args.exp_name}'
        # self.record_n = 0
        self.recorder = recorder
        # self.recorder.register_vis(tri_vis)

        #two classify node
        #[inlier. outlier] [0, 1, ..., 1]
        # gt_idx_per_batch = torch.cat((torch.tensor([0]), torch.ones(args.node_topk))) # set the 0-th as inlier(0) class
        # self.gt_idx = gt_idx_per_batch.repeat(args.batch_size).long().to(device)

        #1-D node feature, inlier idx is always 0
        self.gt_idx = torch.zeros(args.batch_size).long().to(device) #[batch_size]


    def __call__(self, data, args=None) :
        
        # pred = data['graph'].x_pred #extract feat from graph # [batch, node_size, 2]
        if 'graph' in data:
            pred = data['graph'].pred_mat #extract feat from graph # [batch, node_size]
            loss = self.criterion_cls(pred, self.gt_idx)
            hits = pred.argmax(-1)

            #debug vis
            # plt.matshow(pred.detach().cpu())
            # plt.savefig('temp')
            # plt.close()
            
            acc = hits.eq(self.gt_idx).sum() / hits.shape[0]
            self.recorder.update('loss_gnn', loss.item(), args.batch_size, type='f')
            self.recorder.update('acc_gnn', acc.item(), args.batch_size, type='%')
        else:
            loss = 0

        return args.gnn_loss_weight * loss 
    

class Graph_sim_loss(nn.Module):

    def __init__(self, args, recorder, device) -> None:
        super(Graph_sim_loss, self).__init__()
        self.criterion_cls = nn.CrossEntropyLoss()

        # self.output_path = f'{args.result_path}/{args.exp_name}'
        # self.record_n = 0
        self.recorder = recorder
        # self.recorder.register_vis(tri_vis)

        #two classify node
        #[inlier. outlier] [0, 1, ..., 1]
        # gt_idx_per_batch = torch.cat((torch.tensor([0]), torch.ones(args.node_topk))) # set the 0-th as inlier(0) class
        # self.gt_idx = gt_idx_per_batch.repeat(args.batch_size).long().to(device)

        #1-D node feature, inlier idx is always 0
        self.gt_idx = torch.zeros(args.batch_size).long().to(device) #[batch_size]


    def __call__(self, data, args=None) :
        
        # pred = data['graph'].x_pred #extract feat from graph # [batch, node_size, 2]

        pred = data['graph'].sim_mat #extract feat from graph # [batch, node_size]
        loss = self.criterion_cls(pred, self.gt_idx)
        hits = pred.argmax(-1)

        #debug vis
        # plt.matshow(pred.detach().cpu())
        # plt.savefig('temp')
        # plt.close()
        
        acc = hits.eq(self.gt_idx).sum() / hits.shape[0]
        self.recorder.update('loss_gnn', loss.item(), args.batch_size, type='f')
        self.recorder.update('acc_gnn', acc.item(), args.batch_size, type='%')
        return args.gnn_loss_weight * loss 
    

class Graph_sim_mining_loss(nn.Module):

    def __init__(self, args, recorder, device) -> None:
        super(Graph_sim_mining_loss, self).__init__()
        self.criterion_cls = nn.CrossEntropyLoss()

        # self.output_path = f'{args.result_path}/{args.exp_name}'
        # self.record_n = 0
        self.recorder = recorder

        self.ranking_loss = nn.MarginRankingLoss(margin = 0.3)
        # self.recorder.register_vis(tri_vis)

        #two classify node
        #[inlier. outlier] [0, 1, ..., 1]
        # gt_idx_per_batch = torch.cat((torch.tensor([0]), torch.ones(args.node_topk))) # set the 0-th as inlier(0) class
        # self.gt_idx = gt_idx_per_batch.repeat(args.batch_size).long().to(device)

        #1-D node feature, inlier idx is always 0
        self.gt_idx = torch.zeros(args.batch_size).long().to(device) #[batch_size]


    def __call__(self, data, args=None) :
        
        # pred = data['graph'].x_pred #extract feat from graph # [batch, node_size, 2]
        if 'graph' in data:
            pred = data['graph'].sim_mat #extract feat from graph # [batch, node_size]
            loss = self.criterion_cls(pred, self.gt_idx)
            B,N = pred.shape
            positive = pred[:,0].unsqueeze(1)
            negitive = torch.max(pred[:,1:], dim=-1)[0]
            # loss = torch.mean(negitive - positive) 

            y = - torch.ones_like(positive)
            loss = self.ranking_loss(negitive, positive, y)

            #debug vis
            # plt.matshow(pred.detach().cpu())
            # plt.savefig('temp')
            # plt.close()
            hits = pred.argmax(-1)
            acc = hits.eq(self.gt_idx).sum() / hits.shape[0]
            self.recorder.update('loss_gnn', loss.item(), args.batch_size, type='f')
            self.recorder.update('acc_gnn', acc.item(), args.batch_size, type='%')
        else:
            loss = 0
        return args.gnn_loss_weight * loss 
    


class Birank_Loss(nn.Module):

    def __init__(self, args, recorder, device) -> None:
        super().__init__()
        self.criterion_tri = Triplet_Hard_Mining(margin=0.5)
        self.self_mode = args.self_mode

        self.recorder = recorder


    def compute_birank(self, feat1, feat2, pos_mask, margin=0.5, lambda_intra=0.1):
        """from 1 to 2 distance"""
        # print(feat1)
        pos_mask = pos_mask.to(feat1)
        dist_array = compute_dist(feat1, feat2)

        # inter-modality constraints
        furthest_pos, position_pos = torch.max(dist_array*pos_mask, dim=1)
        closest_neg, position_neg = torch.min(dist_array + 1e5*pos_mask, dim=1)

        cross_loss = torch.clamp(furthest_pos - closest_neg + margin, 0).mean()

        # intra-modality constraints
        intra_dist = compute_dist(feat2, feat2)
        intrsa_loss = torch.mean(intra_dist[(position_pos, position_neg)])

        return cross_loss + intrsa_loss * lambda_intra

        

    def __call__(self, data, args=None) :

        labels = data['sar']['id']
        feat = args.tri_in
        sar_feat = data['sar'][feat]
        opt_feat = data['opt'][feat]

        batch = opt_feat.shape[0]
        pos_mask = torch.tensor(labels.expand(batch, batch).eq(labels.expand(batch, batch).t()))
        # print(pos_mask)
        loss1 = self.compute_birank(sar_feat, opt_feat, pos_mask)
        loss2 = self.compute_birank(opt_feat, sar_feat, pos_mask)
        loss = loss1 + loss2

        sim_mat = torch.einsum('md, nd-> mn', sar_feat, opt_feat)
        retri_idx = sim_mat.argmax(-1)
        acc = (retri_idx == torch.arange(0, retri_idx.shape[0]).to(retri_idx)).sum() / args.batch_size

        self.recorder.update('loss_birank', loss.item(), args.batch_size, type='f')
        self.recorder.update('acc_tri', acc.item(), args.batch_size, type='%')

        return loss

def filter_InfoNCE(sim_mat, sim_mat2, logit_scale, loss_fn, label1, label2):

    
    logits_per_image1 = logit_scale * sim_mat
    
    logits_per_image2 = logit_scale * sim_mat2
    
    loss = (loss_fn(logits_per_image1, label1) + loss_fn(logits_per_image2, label2))/2
    
    return loss

