import torch.utils.data as data
from pathlib import Path
import torch
import argparse

from models.model import embed_net_my
from utils.utils import *
from utils.cam import GradCAM, show_cam_on_image, InnerProductTarget, ScoreCAM, GradCAMPlusPlus
from data.dataset_sn6loc import *




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
    parser.add_argument('param_path', default='results/aux_adv/45_lr2.5e-2_margin.t', type=str, help='the relative path of model parameters for testing')
    parser.add_argument('cfg', default='experiments/base.json', type=str, help='json config path')

    #vis config 
    parser.add_argument('--idx', '-i', default=10, type=int, help='number of show image pair')
    #net architacure
    parser.add_argument('--arch', default='resnet50', type=str,help='network baseline:resnet18 or resnet50')
    parser.add_argument('--low_dim', default=512, type=int,metavar='D', help='feature dimension')
    parser.add_argument('--drop', default=0.0, type=float,metavar='drop', help='dropout ratio')

    
    
    args = parser.parse_args()
    args = update_args(args.cfg, args=args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # set data path
    args.data_path = Path.home() / args.data_path
    load_param = Path(os.getcwd()) / args.param_path
    log_path = Path(os.getcwd()) / Path(args.param_path).parent

    model = embed_net_my(args.low_dim, drop=args.drop, arch=args.arch, specify_num=args.specify_num) #init the net framework
    print('==> Building model..')
    if load_param is not None:
        print('==> Resuming from checkpoint..')
        if os.path.isfile(load_param):
            print(f'==> loading checkpoint from {load_param}')
            checkpoint = torch.load(load_param)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['net'])
            print(f'==> loaded checkpoint in epoch {checkpoint["epoch"]}')
        else:
            print(f'==> no checkpoint found at {load_param}')
    model.to(device)
    
    # Note: change layer
    if args.specify_num == 5:
        target_layers_opt = [model.modal_net['opt'].specify_layers[-1][-1]] #last layer of network
        target_layers_sar = [model.modal_net['sar'].specify_layers[-1][-1]]
    else:
        target_layers_opt = [model.common_layers[-1][-1]]
        target_layers_sar = [model.common_layers[-1][-1]]


    # get CAM sample
    trainset = eval(args.train_dataset)(args)
    sampler = SortIDSampler(trainset.train_sar_label)
    trainloader = data.DataLoader(trainset, batch_size=1, sampler=sampler)
    for i, data_dict in enumerate(trainloader):
        if i > args.idx:
        # if data_dict['sar']['id'] == args.idx:
            break

        data_dict = to_cuda(data_dict)
        output = model(data_dict)
        
        input_opt = dict(opt=output['opt'])
        input_sar = dict(sar=output['sar'])

        img_opt = plt.imread(args.data_path/f"train/imgs/opt/{data_dict['opt']['img_name'].item()}.png")
        img_sar = plt.imread(args.data_path/f"train/imgs/sar/{data_dict['sar']['img_name'].item()}.png")

        # get metric target
        opt_feat = output['opt']['out_feat']
        sar_feat = output['sar']['out_feat']    
        dist = torch.sum((opt_feat - sar_feat) ** 2, dim=-1).sqrt()
        print(f"id:{data_dict['sar']['id'].item()}, sar_img_name: {data_dict['sar']['img_name'].item()}.png, inner product: {dist.item():.2f}")

        targets_opt = InnerProductTarget(sar_feat, 'opt')
        targets_sar = InnerProductTarget(opt_feat, 'sar')

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        # Construct the CAM object once, and then re-use it on many images:
        cam_opt = GradCAMPlusPlus(model=model, target_layers=target_layers_opt, use_cuda=True)
        cam_sar = GradCAMPlusPlus(model=model, target_layers=target_layers_sar, use_cuda=True)

        #ScoreCAM, GradCAM
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam_opt = cam_opt(input_tensor=input_opt, targets=targets_opt)[0]
        grayscale_cam_sar = cam_sar(input_tensor=input_sar, targets=targets_sar)[0]

        vis_opt = show_cam_on_image(img_opt, grayscale_cam_opt, use_rgb=True)
        vis_sar = show_cam_on_image(img_sar, grayscale_cam_sar, use_rgb=True)

        fig, axs =  plt.subplots(2, 2)
        axs = axs.flatten()
        axs[0].imshow(vis_sar)
        axs[0].set_title('SAR query image')
        axs[1].imshow(vis_opt)
        axs[1].set_title('Optical reference image')
        axs[2].imshow(img_sar)
        axs[3].imshow(img_opt)
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([]) 
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        plt.savefig(log_path/f"CAM_vis_{data_dict['sar']['id'].item()}.png")