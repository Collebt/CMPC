import argparse
import sys
import torch
import torch.nn as nn
import torch.utils.data as data
from data.dataset import * #TestData, SN6TestData, SN6TestPosAlign
from data.dataset_sn6loc import *
from data.data_os import *

from utils.utils import *
import time
import random
from pathlib import Path
from scipy.io import savemat
##################################visual
# from sklearn.manifold import TSNE  #debug
# from sklearn.decomposition import PCA


import matplotlib.pyplot as plt
from models.model import embed_net, embed_net_my
from models.rk_net import rk_two_view_net
from models.GNN import *
from models.RSB import *
from models.Xmod import * 
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from losses import compute_dist
from utils.draw_graph import vis_graph, statstic_edge_weigt

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(-1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Feat_Arch:
    def __init__(self, args, save_path):
        #####################build reference image-base ######
        #init the gallery dataloader
        self.batch_size = args.test_batch
        gallset = FeatEmbTrain(args, mode='opt', dataset='train')
        self.gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        
        #here the test index is not started from 0
        # init the reference gallery 
        self.ref_feat = None
        self.ref_id = None
        self.ref_pos = None
        self.ngall = None
        self.pos_eval = args.pos_eval
        self.refine_eval = args.refine_eval
        self.args = args
        self.dist = args.dist_type
        self.norm = Normalize(2)

        self.refine_batch = 107
        self.save_path = save_path

    def extract_feat(self, net, dataloader, mode='opt'):
        """extract feature of reference images
        Returns:
            ref_fear: n-dimension extracted features of the reference images
            ref_id: the id of the location
            ref_pos: GPS position of the correspondent reference images
        """
        net.eval()   #evaluate the network
        feat = []
        feat_vec = []
        ids = []
        pos = []
        img_name = []
        with torch.no_grad():
            for data in dataloader:
                data = to_cuda(data) if 'cuda' in next(net.parameters()).device.type else data#defaule to 1 cuda device
                data = net(data)
                feat.append(data[mode]['out_feat'])
                feat_vec.append(data[mode]['out_vec'])
                ids.append(data[mode]['id'])
                if self.pos_eval:
                    pos.append(data[mode]['pos'])
                if mode == 'sar':
                    img_name.extend(data[mode]['img_name'])
        ref_feat = torch.cat(feat, dim=0)
        ref_vec = torch.cat(feat_vec, dim=0)
        ref_id = torch.cat(ids, dim=0)
        ref_pos = torch.cat(pos, dim=0) if self.pos_eval else None

        if len(img_name) > 0:
            return ref_feat, ref_vec, ref_id, ref_pos, img_name
        else:
            return ref_feat, ref_vec, ref_id, ref_pos

    def retrieve(self, query_feat):
        if self.dist == 'sim':
            # query_feat = self.norm(query_feat)
            # ref_feat = self.norm(self.ref_feat)
            similarity = torch.einsum('nd,md->nm', query_feat, self.ref_feat) #inner-prodict
        elif self.dist == 'l2':
            similarity = - compute_dist(self.ref_feat, query_feat).T
        # retrieval_sim = torch.softmax(similarity, dim=-1)
        top1_idx = similarity.argmax(dim=-1)
        pred_pos = self.ref_pos[top1_idx] if self.pos_eval else None
        return similarity, pred_pos

    def eval_retrieval(self, sim_mat, gt_qid):
        # compute distmat
        mAP = []

        retrieval_idx = torch.argsort(sim_mat, dim=1, descending=True)
        matches = (self.ref_id[retrieval_idx] == gt_qid.reshape(-1, 1)) #get binary matching mat along the reference
        hit_pos = torch.where(matches == 1)[1]

        cmc = matches.sum(0).cumsum(-1) 
        cmc = cmc.to(torch.float)/ gt_qid.shape[0]

        mAP = torch.mean(1 / (hit_pos.to(torch.float) + 1))
        return cmc, mAP, matches            


    def feat_embed(self, net, query_loader,dataset='train'):

        self.ref_feat, self.ref_vec, self.ref_id, self.ref_pos = self.extract_feat(net, self.gall_loader, mode='opt')
        query_feat, query_vec, gt_id, gt_pos, query_name = self.extract_feat(net, query_loader, mode='sar')
        
        sim_mat, pred_pos = self.retrieve(query_feat)
   
        opt_data = dict(feat=self.ref_feat, id=self.ref_id, pos=self.ref_pos)
        sar_data = dict(feat=query_feat, id=gt_id, pos=gt_pos, name=query_name)
        data = dict(opt=opt_data, sar=sar_data, sim_mat=sim_mat)
        torch.save(data, self.save_path/'embed_feats')

class sn6loc_test:
    def __init__(self, args):
        #####################build reference image-base ######
        #init the gallery dataloader
        self.batch_size = args.test_batch
        gallset = eval(args.test_dataset)(args, mode='opt')
        self.gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        
        #here the test index is not started from 0
        # init the reference gallery 
        self.ref_feat = None
        self.ref_id = None
        self.ref_pos = None
        self.ngall = None
        self.pos_eval = args.pos_eval
        self.refine_eval = args.refine_eval
        self.args = args
        # self.dist = args.dist_type
        self.norm = Normalize(2)

        self.refine_batch = args.test_batch

    def update_gall_feature(self, net):
        """update reference feature with latest embedding network
        Args:
            net (nn.module):
        """
        print('==> Building Image-Base Gallery...')
        start = time.time()
        self.ref_feat, self.ref_vec, self.ref_id, self.ref_pos = self.extract_feat(net, self.gall_loader, mode='opt')
        self.ngall = self.ref_feat.shape[0]
        print(f'Gallery Loading Time:\t {time.time() - start:.3f}')

    def extract_feat(self, net, dataloader, mode='opt'):
        """extract feature of reference images
        Returns:
            ref_fear: n-dimension extracted features of the reference images
            ref_id: the id of the location
            ref_pos: GPS position of the correspondent reference images
        """
        net.eval()   #evaluate the network
        feat = []
        feat_vec = []
        ids = []
        pos = []
        img_name = []
        with torch.no_grad():
            for data in dataloader:
                data = to_cuda(data) if 'cuda' in next(net.parameters()).device.type else data#defaule to 1 cuda device
                data = net(data)
                feat.append(data[mode]['out_feat'])
                feat_vec.append(data[mode]['out_vec'])
                ids.append(data[mode]['id'])
                if self.pos_eval:
                    pos.append(data[mode]['pos'])
                if mode == 'sar':
                    img_name.extend(data[mode]['img_name'])
        ref_feat = torch.cat(feat, dim=0)
        ref_vec = torch.cat(feat_vec, dim=0)
        ref_id = torch.cat(ids, dim=0)
        ref_pos = torch.cat(pos, dim=0) if self.pos_eval else None

        if len(img_name) > 0:
            return ref_feat, ref_vec, ref_id, ref_pos, img_name
        else:
            return ref_feat, ref_vec, ref_id, ref_pos

    def retrieve(self, query_feat):
        # if self.dist == 'sim':
        #     # query_feat = self.norm(query_feat)
        #     # ref_feat = self.norm(self.ref_feat)
        #     similarity = torch.einsum('nd,md->nm', query_feat, self.ref_feat) #inner-prodict
        # elif self.dist == 'l2':
        similarity = - compute_dist(query_feat, self.ref_feat)
        # retrieval_sim = torch.softmax(similarity, dim=-1)
        top1_idx = similarity.argmax(dim=-1)
        pred_pos = self.ref_pos[top1_idx] if self.pos_eval else None
        return similarity, pred_pos

    def eval_retrieval(self, sim_mat, gt_qid):
        # compute distmat
        mAP = []
        retrieval_idx = torch.argsort(sim_mat, dim=1, descending=True)
        matches = (self.ref_id[retrieval_idx] == gt_qid.reshape(-1, 1)) #get binary matching mat along the reference
        hit_pos = torch.where(matches == 1)[1]

        cmc = matches.sum(0).cumsum(-1) 
        cmc = cmc.to(torch.float)/ gt_qid.shape[0]

        mAP = torch.mean(1 / (hit_pos.to(torch.float) + 1))
        return cmc, mAP, matches

    def eval_refine(self, refine_idx, gt_qid, coarse_matches):
        # mAP = []

        fine_matches = (refine_idx == gt_qid.reshape(-1, 1)) #get matching mat by descending sorting
        

        rank_top = fine_matches.shape[-1]
        matches = torch.cat([fine_matches, coarse_matches[:, rank_top:]], dim=-1)
        hit_pos = torch.where(matches == 1)[1]


        cmc = matches.sum(0).cumsum(-1) 
        cmc = cmc.to(torch.float)/ gt_qid.shape[0]
        mAP = torch.mean(1 / (hit_pos.to(torch.float) + 1))
        return cmc, mAP, matches

    def eval_localizarion(self, pred_pos, gt_pos, max_meter=40, vis_path=None):
        # evaluate localization error 

        err = torch.sqrt(((pred_pos - gt_pos) ** 2).sum(-1))
        cmc = []
        for r in torch.linspace(0, max_meter ,20): #40m threshold error
            cmc.append(torch.sum(err<=r))
        
        cmc = torch.tensor(cmc).to(float)  / err.shape[0] 
        merr = err.mean()
       
        return cmc, merr



    def refine(self, model, query_feat, sim_mat):
        # sim_mat = torch.einsum('md,nd->mn', query_feat, self.ref_feat)
        node_topk = self.args.node_topk
        _, topk_idx = sim_mat.topk(k=node_topk)
        query_len = query_feat.shape[0]#, query_feat[-1]

        refine_idx_in_topk_all = []
        graph_all = []
        score_topk = []
        with torch.no_grad(): 
            for b_idx in range(0, query_len, self.refine_batch): #batch_query;[batch, num, channel]
                if b_idx + self.refine_batch < query_len:
                    end_idx = b_idx+self.refine_batch
                else:
                    end_idx = query_len
                query_feat_batch = query_feat[b_idx:end_idx]

                graph_batch, x_batch_topk, refine_idx_in_topk, graph_list = model.refine_inference(query_feat_batch, self.ref_feat, self.ref_pos)
                refine_idx_in_topk_all.append(refine_idx_in_topk)
                graph_all.extend(graph_list)
                score_topk.append(x_batch_topk)


        refine_idx_in_topk_all = torch.cat(refine_idx_in_topk_all, dim=0)
        score_topk = torch.cat(score_topk, dim=0)

        refine_id = self.ref_id[topk_idx[(torch.arange(query_len).unsqueeze(1).expand(-1, node_topk), refine_idx_in_topk_all)]]
        refine_pos= self.ref_pos[topk_idx[(torch.arange(query_len), refine_idx_in_topk_all[:,0])]] #[Num_query, 1]

        # reassign score to the sim_mat
        refine_sim_mat = sim_mat
        for rank in range(0, node_topk):
            refine_sim_mat[(torch.arange(query_len), topk_idx[(torch.arange(query_len), refine_idx_in_topk_all[:,rank])])] = node_topk - rank

        return score_topk, refine_sim_mat, refine_pos, graph_all #refine_id, 
                

    
    def eval_cmc(self, net, query_loader, vis_path=None, vis_full=False, vis_retrieval_imgs=False):

        self.update_gall_feature(net)
        query_feat, query_vec, gt_id, gt_pos, query_img = self.extract_feat(net, query_loader, mode='sar')

        print(f'-----------------------')
        print(f'dim:{query_feat.shape[-1]}|#Ids\t| #Img ')
        print(f'-----------------------')
        print(f'Query\t|{torch.unique(gt_id).shape[0]}\t|{query_feat.shape[0]} ')
        print(f'Gallery\t|{self.ngall}\t|{self.ngall}')
        print(f'-----------------------')

        # evaluate retrieval accuracy
        sim_mat, pred_pos = self.retrieve(query_feat)
        cmc, mAP1, coarse_match = self.eval_retrieval(sim_mat, gt_id)
        print(f'Retrieval: top-1:{cmc[0]:.2%} | top-5:{cmc[4]:.2%} | top-10:{cmc[9]:.2%} | top-20:{cmc[19]:.2%} | mAP:{mAP1:.2%}')
        
        # self.save_search_res(sim_mat, gt_id, self.ref_id, query_img) # save the coarse search results
        
        if self.pos_eval: #evaluate meter-level localization accuracy
            max_meter = 200
            loc_cmc, mer = self.eval_localizarion(pred_pos, gt_pos, max_meter)
            print(f'Meter-level: {max_meter/4:.0f}m:{loc_cmc[4]:.2%} | {max_meter/2:.0f}m:{loc_cmc[9]:.2%} | {max_meter/4*3:.0f}m:{loc_cmc[14]:.2%} | {max_meter:.0f}m:{loc_cmc[19]:.2%} | merr:{mer:.2f}m')
            
            if self.refine_eval:

                score_topk, refine_sim_mat, refine_pos, graph_list  = self.refine(net, query_feat, sim_mat)
                del sim_mat
                refine_cmc, refine_mAP, refine_match = self.eval_retrieval(refine_sim_mat, gt_id)
                # self.visual_score_eval(score_topk, sim_mat, gt_id)
                loc_cmc, mer = self.eval_localizarion(refine_pos, gt_pos, max_meter)
                
                print(f'Refine retrieval: top-1:{refine_cmc[0]:.2%} | top-5:{refine_cmc[4]:.2%} | top-10:{refine_cmc[9]:.2%}| top-20:{refine_cmc[19]:.2%} | mAP:{refine_mAP:.2%}')
                print(f'Meter-level: {max_meter/4:.0f}m:{loc_cmc[4]:.2%} | {max_meter/2:.0f}m:{loc_cmc[9]:.2%} | {max_meter/4*3:.0f}m:{loc_cmc[14]:.2%} | {max_meter:.0f}m:{loc_cmc[19]:.2%} | merr:{mer:.2f}m')
        
                del score_topk, refine_sim_mat

    def visual_score_eval(self, score_topk, sim_mat, gt_id):
        node_topk = self.args.node_topk
        _, topk_idx = sim_mat.topk(k=node_topk)
        cand_id = self.ref_id[topk_idx]

        gt_masks = cand_id == gt_id.unsqueeze(-1)
        gt_mask = gt_masks[:100].detach().cpu().T
        scores = score_topk[:100].detach().cpu().T
        # Create a plot of the weighted matrix
        fig, ax = plt.subplots()
        im = ax.imshow(scores)
        for i in range(gt_mask.shape[0]):
            for j in range(gt_mask.shape[1]):
                if gt_mask[i, j]:
                    rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='red', linewidth=0.5)
                    ax.add_patch(rect)

        # Add a colorbar to the plot
        # cbar = ax.figure.colorbar(im, ax=ax)

        # Set the axis labels and title
        ax.axis('off')
        # ax.set_xticks(np.arange(scores.shape[1]))
        # ax.set_yticks(np.arange(scores.shape[0]))
        # ax.set_xticklabels(np.arange(1, scores.shape[1]+1))
        # ax.set_yticklabels(np.arange(1, scores.shape[0]+1))
        ax.set_ylabel('TopK Candidate')
        ax.set_xlabel('Queries')
        ax.set_title('Topk Score with GT Highlight')

        # Show the plot
        plt.savefig('weighted_matrix.png')
        plt.close()


    def vis_retrieval_imgs(self, pred_retrieval, gt_pos, gt_id, query_img, show_num=4, gall_num=5, log_path=None, fix_id=None, f_name='base'):
        
        def draw_imgs(ids_mat, dis_mat, gt_ranks, f_name):
            rows, cols = ids_mat.shape

            for b_idx in range(0, rows//4*4, 4):
                vis_mat = ids_mat[b_idx:b_idx+4]
                vis_dis = dis_mat[b_idx:b_idx+4]
                vis_gt = gt_ranks[b_idx:b_idx+4]

                # init img 
                fig, axs = plt.subplots(show_num, cols, figsize=(cols, show_num*1.2))
                for r in range(show_num):
                    gt_rank = vis_gt[r].item()+1
                    for c in range(cols):
                        if c == 0:
                            imgs  = plt.imread(str(self.args.data_path / f'test/imgs/sar/{query_img[vis_mat[r, c]]}.png'))
                        else:
                            imgs  = plt.imread(str(self.args.data_path / f'test/imgs/opt/{self.ref_id[vis_mat[r, c]]}.png'))
                            if c > gall_num:
                                axs[0, c].set_title('GT-Ref')
                                axs[r, c].set_xlabel(f'rank:{gt_rank:.0f}')
                            else:
                                axs[0, c].set_title(f'Top-{c}') 
                                axs[r, c].set_xlabel(f'{vis_dis[r, c-1]:.2f}')
                                if c == gt_rank:
                                    green_box = np.zeros_like(imgs)
                                    green_box[:,:,1] = 1
                                    green_box[10:-10, 10:-10] = imgs[10:-10, 10:-10]
                                    imgs = green_box
                        axs[r, c].imshow(imgs)
                        axs[r, c].set_xticks([])
                        axs[r, c].set_yticks([]) 
                        axs[r, c].spines['top'].set_visible(False)
                        axs[r, c].spines['right'].set_visible(False)
                        axs[r, c].spines['bottom'].set_visible(False)
                        axs[r, c].spines['left'].set_visible(False)
                axs[0,0].set_title('Query')
                plt.subplots_adjust(wspace=0.1, hspace=0.3)
                plt.savefig(log_path / f'{f_name}_{b_idx//4}')
                plt.close()

        # find good rssult
        qids = gt_id
        top5_idx = pred_retrieval.argsort(dim=-1, descending=True)[:,:gall_num]
        query_img_idx = torch.arange(0, len(qids)).to(top5_idx)

        gids = self.ref_id
        retrieval_idx = torch.argsort(pred_retrieval, dim=1, descending=True)
        matches = (gids[retrieval_idx] == qids.reshape(-1, 1)) #get matching mat with absenc sort
        hit_pos = torch.where(matches == 1)[1]

        match_mask = gids[top5_idx[:,0]] == qids
        
        all_sample_good = torch.cat([query_img_idx[match_mask].reshape(-1,1),top5_idx[match_mask,:]], dim=-1)
        all_sample = torch.cat([query_img_idx.reshape(-1,1),top5_idx], dim=-1)
        
        sample_good = all_sample_good[:show_num]
        
        
        sample_bad =  torch.cat([query_img_idx[~match_mask].reshape(-1,1),top5_idx[~match_mask,:]], dim=-1)[:show_num] 
        sample_bad_hit =  hit_pos[~match_mask][:show_num].squeeze()
        sample_bad_gt_id =   retrieval_idx[~match_mask] [torch.arange(show_num), sample_bad_hit].reshape(-1,1)  

        dis_good = ((gt_pos[match_mask][:show_num].unsqueeze(1) - self.ref_pos[top5_idx[match_mask,:]][:show_num]) **2).sum(-1).sqrt()
        dis_bad = ((gt_pos[~match_mask][:show_num].unsqueeze(1) - self.ref_pos[top5_idx[~match_mask,:][:show_num]]) **2).sum(-1).sqrt()

        draw_imgs(sample_good, dis_good, torch.zeros(gall_num), 'good_vis'+f_name)
        draw_imgs(torch.cat([sample_bad, sample_bad_gt_id],dim=-1), dis_bad, sample_bad_hit, 'bad_vis'+f_name)
        
        ids_show = query_img_idx[match_mask].flatten()
        # print('good ID are: '+','.join([f'{ids}' for ids in ids_show]))
        # if f_name == 'gnn':
        #     fix_id = ids_show
        if fix_id is not None:
            # fix_id is a list, i.e. [1,2,4,6,7]
            sample_fix = all_sample[fix_id]
            dis_fix = ((gt_pos[fix_id].unsqueeze(1) - self.ref_pos[top5_idx][fix_id]) **2).sum(-1).sqrt()
            draw_imgs(sample_fix, dis_fix, hit_pos[fix_id], 'fix_vis'+f_name)
            print('fixed ID are: '+','.join([f'{ids}' for ids in qids[fix_id]]))
        return ids_show

    
    def save_search_res(self, sim_mat, gt_id, ref_id, query_img):
        
        qimgs = []
        _, topk_idx = sim_mat.topk(k=10)
        topk_gids = ref_id[topk_idx]

        retrieval_idx = torch.argsort(sim_mat, dim=1, descending=True)
        matches = (self.ref_id[retrieval_idx] == gt_id.reshape(-1, 1)) #get binary matching mat along the reference
        hit_pos = torch.where(matches == 1)[1]
        hit_topk = hit_pos < 10
        
        for this_qid in gt_id:

            q_name  = str(self.args.data_path / f'test/imgs/sar/{query_img[this_qid]}.png')#get image name
            qimgs.append(q_name)
        mdic = {'q_name': qimgs, 'q_ids':gt_id.cpu().numpy(), 'in_topk':hit_topk.cpu().numpy(),
                'topk_ids':topk_gids.cpu().numpy(), 'r_ids':ref_id.cpu().numpy(), 'sim_mat':sim_mat.cpu().numpy()}

        savemat("coarse_search.mat", mdic)
        




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
    parser.add_argument('param_path', default='results/aux_adv/45_lr2.5e-2_margin.t', type=str, help='the relative path of model parameters for testing')
    parser.add_argument('cfg', default='experiments/base.json', type=str, help='json config path')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
    parser.add_argument('--arch', default='resnet50', type=str, help='network baseline')
    parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--model_path', default='results/', type=str, help='model save path')
    parser.add_argument('--log_path', default='results/test_log/', type=str, help='log save path')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--low_dim', default=512, type=int,
                        metavar='D', help='feature dimension')
    parser.add_argument('--img_w', default=256, type=int,
                        metavar='imgw', help='img width')
    parser.add_argument('--img_h', default=256, type=int,
                        metavar='imgh', help='img height')
    parser.add_argument('--batch_size', default=24, type=int,
                        metavar='B', help='training batch size')
    parser.add_argument('--test_batch', default=16, type=int,
                        metavar='tb', help='testing batch size')
    parser.add_argument('--method', default='id', type=str,
                        metavar='m', help='Method type')
    parser.add_argument('--drop', default=0.0, type=float,
                        metavar='drop', help='dropout ratio')
    parser.add_argument('--trial', default=1, type=int,
                        metavar='t', help='trial')
    parser.add_argument('--gpu', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--mode', default='all', type=str, help='all or indoor')
    parser.add_argument('--per_img', default=1, type=int,
                        help='number of samples of an id in every batch')
    parser.add_argument('--w_hc', default=0.5, type=float,
                        help='weight of Hetero-Center Loss')
    parser.add_argument('--thd', default=0, type=float,
                        help='threshold of Hetero-Center Loss')
    parser.add_argument('--gall-mode', default='single', type=str, help='single or multi')

    args = parser.parse_args()
    args = update_args(args.cfg, args=args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)  # 为所有GPU设置随机种子
    np.random.seed(1)
    random.seed(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # set data path
    args.data_path = Path.home() / args.data_path
    load_param = Path(os.getcwd()) / args.param_path
    log_path = Path(os.getcwd()) / Path(args.param_path).parent

    ## set log output
    sys.stdout = Logger(Path(log_path) /  f'Test_{args.test_dataset}_{time.asctime()}.log')
    print(f"==========\nArgs:{args}\n==========")

    # if args.method == 'id':
    #     criterion = nn.CrossEntropyLoss()
    #     criterion.to(device)

    # loading the parameter of net
    # net = embed_net_my(args.low_dim, drop=args.drop, arch=args.arch, specify_num=args.specify_num) #init the net framework
    # net = rk_two_view_net(6328, droprate = args.drop, stride = args.stride, pool = args.pool, share_weight = args.share)
    model = eval(args.model)(args)
    print('==> Building model..')
    if load_param is not None:
        print('==> Resuming from checkpoint..')
        if os.path.isfile(load_param):
            print(f'==> loading checkpoint from {load_param}')
            checkpoint = torch.load(load_param)
            start_epoch = checkpoint['epoch']
            # try:
            model.load_state_dict(checkpoint['model'])
            # except:
            #     model.load_state_dict(checkpoint['net'])
            print(f'==> loaded checkpoint in epoch {checkpoint["epoch"]}')
        else:
            print(f'==> no checkpoint found at {load_param}')
    model.to(device)
   
    result_test_obj = sn6loc_test(args)
    queryset = eval(args.test_dataset)(args, mode='sar')
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    
    result_test_obj.eval_cmc(model, query_loader, vis_path=log_path, vis_retrieval_imgs=True)
    print('test finished')


    