
import torch
import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt

def wd_vis(record_dict, *args):
    """visualize the w distance of the features

    Args:
        record_dict(dict)--wd(dict)--  name(str): output file name
                                    |- wd_list(list): list of the output 1-d values
                                    
        [src_feat (tensor): w-value of the source feature output from Discriminator. [b]
        tgt_feat (tensor): w-value of the target feature output from Discriminator. [b]]
    """
    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()

    f_name = record_dict['wd']['name']
    f_name = record_dict['wd']['name']
    for i, feats in enumerate(record_dict['wd']['feat']):
        src_feat = feats[0].squeeze().detach().cpu()
        tgt_feat = feats[1].squeeze().detach().cpu()

        src_label = torch.ones_like(src_feat)
        tgt_label = torch.zeros_like(tgt_feat)
        src_pos = torch.stack([src_feat, src_label])
        tgt_pos = torch.stack([tgt_feat, tgt_label])
        axs[i].scatter(src_pos[0], src_pos[1], alpha=0.2, c='blue', label='SAR')
        axs[i].scatter(tgt_pos[0], tgt_pos[1], alpha=0.2, c='red', label='optical')
        axs[i].legend()
        for src_p, tgt_p in zip(src_pos.T, tgt_pos.T):
            X = [src_p[0], tgt_p[0]]
            Y = [src_p[1], tgt_p[1]]
            axs[i].plot(X, Y, c='black', alpha=0.22, linewidth=0.5)
        axs[i].set_yticks([]) 
        axs[i].set_xlabel(f'epoch{i*171}')
        

    plt.subplots_adjust(hspace=0.5)
    plt.savefig(f_name, dpi=200)
    plt.close()


def cls_vis(record_dict, *args):
    """visualize the class result of the features

    Args:
        record_dict(dict)--wd(dict)--  name(str): output file name
                                    |- feat(list): list of the output 1-d values
                                    
        [src_feat (tensor): classifcation result of the source feature output from Discriminator. [b]
        tgt_feat (tensor): classifcation result of the target feature output from Discriminator. [b]
    """

    f_name = record_dict['wd']['name']

    src_pos = record_dict['wd']['feat'][0].T.squeeze().detach().cpu() #shape =[2, b]
    tgt_pos = record_dict['wd']['feat'][1].T.squeeze().detach().cpu()


    plt.scatter(src_pos[0], src_pos[1], alpha=0.2, c='blue', label='SAR')
    plt.scatter(tgt_pos[0], tgt_pos[1], alpha=0.2, c='red', label='optical')
    plt.legend()
    for src_p, tgt_p in zip(src_pos.T, tgt_pos.T):
        X = [src_p[0], tgt_p[0]]
        Y = [src_p[1], tgt_p[1]]
        plt.plot(X, Y, c='black', alpha=0.22, linewidth=0.5)
    plt.xlabel(f'optical')
    plt.ylabel(f'SAR')   
    plt.savefig(f_name, dpi=200)
    plt.close()

def tri_vis(record_dict, *args):
    """visualize the distance between positive& negative samples.
    Args:
        record_dict(dict)--tri_vis(dict)--  name(str): output file name
                                        |- feat(list): distance of the output 1-D samples
                                    
        [pos_distance (tensor): the positive distance of samples from a batch. [b]
        neg_distance (tensor): the negative distance of samples from a batch. [b]
    """
    batch_idx = args[0]
    if batch_idx < 900:
        return 0
    f_name = record_dict['tri_vis']['name']

    feat = torch.stack(record_dict['tri_vis']['feat']) #shape = [Nx2, 2, batch_size]
    iters = feat.shape[0] // 2

    feat = feat.permute(1, 0, 2) #shape = [2, 2N, batch_size]
    

    pos_d = feat[0].reshape(iters, -1).detach().cpu() #[N, batch_size]
    neg_d = feat[1].reshape(iters, -1).detach().cpu() #[N, batch_size]

    

    iters_x = torch.arange(0, iters).reshape(-1, 1).repeat(1, pos_d.shape[-1]).detach().cpu()


    plt.scatter(iters_x.flatten(), pos_d.flatten(), alpha=0.22, c='green', label='positive dist')
    plt.scatter(iters_x.flatten(), neg_d.flatten(), alpha=0.22, c='red', label='negative dist')
    plt.legend()

    pos_dis_avg = pos_d.mean(-1)
    neg_dis_avg = neg_d.mean(-1)

    plt.plot(torch.arange(0, iters), pos_dis_avg, c='green')
    plt.plot(torch.arange(0, iters), neg_dis_avg, c='red')

    plt.xlabel(f'iterations')
    plt.ylabel(f'distance')   
    plt.savefig(f_name, dpi=200)
    plt.close()