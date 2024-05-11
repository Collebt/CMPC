import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.scheduler import Scheduler
from utils.show_log import show_loss

import torch.backends.cudnn as cudnn

import torch.utils.data as data

import os, sys, time, argparse, re, random
from pathlib import Path

from data.dataset import * #OSData, TestData, SN6AlignData, SN6TestData
from data.dataset_sn6loc import *
from data.data_os import *

# Models 
from models.model import * #embed_net, embed_net_my, Discriminator 
from models.rk_net import * #rk_two_view_net
from models.GNN import * #embed_net_GNN
from models.Xmod import * 
from models.RSB import * 
from models.DCMHN import *
from models.FSRA import * 

from test import sn6loc_test, Feat_Arch
from utils.utils import *

from losses import *


def train(model, dataloader, criterions, epoch, recorder, args, optimizer, scheduler, warm_up=1):

    recorder.reset() #init record
    current_lr = scheduler.get_last_lr()
    model.train() 
    
    for batch_idx, data in enumerate(dataloader):
        start_t = time.time()

        data = to_cuda(data) if 'cuda' in next(model.parameters()).device.type else data#defaule to 1 cuda device
        
        data = model(data)
        loss = sum([cri(data, args) for cri in criterions])
        loss = loss * warm_up

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() #optimize

        # record maodel parameter& grad
        params_list = list(zip(*model.named_parameters()))[-1]
        grads = torch.cat([x.grad.flatten() if x.grad is not None else x.new_zeros(1) for x in params_list]) #for debug
        params =  torch.cat([x.data.flatten() if x.grad is not None else x.new_zeros(1) for x in params_list]) #for debug

        if torch.isnan(params).sum() >0 or torch.isnan(grads).sum() >0:
            print(params)
            print(grads)

        #record metric
        recorder.update('Loss', loss.item(), args.batch_size)
        recorder.update('FPS', args.batch_size/(time.time() - start_t))

        if batch_idx % 20 == 0: #show record
            rcd_list = [f'Epoch[{epoch}][{batch_idx}/{len(trainloader)}]',
                        f'lr:{current_lr[0]:.3f}']
            rcd_list.extend(recorder.display())
            print(' '.join([rcd for rcd in rcd_list])) 
            recorder.reset()
        
        #for visualization
        if vis_path is not None and batch_idx % 100 == 0 and epoch % 5 == 0:
            recorder.param_vis(batch_idx)






parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('exp_name', type=str, help='name of this experiment')
parser.add_argument('cfg', default='experiments/rk_base_arb.json', type=str, help='json config path')
parser.add_argument('--output_feat', action='store_true', 
                        help='output the embedding feature of the images')
parser.add_argument('--gpu', '-g', default='0', type=str,
                      help='gpu device ids for CUDA_VISIBLE_DEVICES')

parser.add_argument('--lr', default=0.025, type=float, help='learning rate')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', default='', type=str,
                    help='load model to resume from checkpoint')
parser.add_argument('--result_path', default='results/debug', type=str,
                    help='model save path')
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--low_dim', default=512, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--img_w', default=256, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=256, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch_size', default=16, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test_batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--drop', default=0.0, type=float,
                    metavar='drop', help='dropout ratio')

parser.add_argument('--per_img', default=1 , type=int,
                    help='number of samples of an id in every batch')
parser.add_argument('--epochs', default=60, type=int,
                    help='weight of Hetero-Center Loss')
parser.add_argument('--dist_type', default='l2', type=str,
                    help='type of distance')


args = parser.parse_args()
args = update_args(args.cfg, args=args)

#set train environments
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)   
np.random.seed(1)
random.seed(1)
device = f'cuda' if torch.cuda.is_available() else 'cpu' 
cudnn.benchmark = True #acclerate convolution

#set paths
args.data_path = Path.home() / args.data_path #get absolute path
log_path = Path(args.result_path) / args.exp_name
vis_path = None
checkpoint_path = log_path

os.makedirs(log_path) if not os.path.isdir(log_path) else None
os.makedirs(checkpoint_path) if not os.path.isdir(checkpoint_path) else None


#output path & print options
sys.stdout = Logger(log_path /  f'{args.train_dataset}_{time.asctime()}.log')
print(f"==========\nArgs:{args}\n==========")

# dataloader
print('==> Loading data..')
trainset = eval(args.train_dataset)(args)
queryset = eval(args.test_dataset)(args, mode='sar')
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
#test loader
evaluation = sn6loc_test(args)

#build model
start_epoch = 0
print('==> Building model..')
model = eval(args.model)(args)
# load the previous model
if len(args.resume) > 0 :
    load_param = args.resume
    print('==> Resuming from checkpoint..')
    if os.path.isfile(load_param):
        print(f'==> loading checkpoint from {load_param}')
        checkpoint = torch.load(load_param)
        start_epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['model'])
        print(f'==> loaded checkpoint in epoch {checkpoint["epoch"]}')
    else:
        print(f'==> no checkpoint found at {load_param}')
model.to(device)

#record
recorder = Recorder({"FPS":'d', "Loss":'f'})

#loss function
criterions = [eval(l)(args, recorder, device) for l in args.loss]
print("criterion:", ", ".join([type(cri).__name__ for cri in criterions]))

#Optimizer & LR_scheduler 
optimizer = model.build_opt(args) #optimizer for main model
optimizers = [optimizer]
for c in criterions:  #optimizer for learnable loss
    if "optimizer" in c.__dict__:
        optimizers.append(c.optimizer)


exp_lr_scheduler = Scheduler(optimizers, step_size=20, gamma=0.1)

#warm up initial
warm_up = 0.


# training
print('==> Start Training...')
for epoch in range(start_epoch, args.epochs):
    print(f'===== Experiment: {args.exp_name} ====')
    # identity sampler
    sampler = SARSampler(trainset.train_sar_label, args.batch_size, args.per_img)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=args.workers, drop_last=True)


    # change warm up
    warm_up = min(1.0, warm_up + 1 / args.warm_epoch) if epoch<args.warm_epoch else 1

    # training in 1 epoch
    train(model, trainloader, criterions, epoch, recorder, args, optimizer, exp_lr_scheduler, warm_up=warm_up)
    exp_lr_scheduler.step() #adjust the optimizers learning rate

    # evaluation 
    if epoch % 2 == 0 or epoch == args.epochs-1: 
        evaluation.eval_cmc(model, query_loader, vis_path=vis_path)
    # save model parameters
    if ((epoch+1) % args.save_epoch == 0) or (epoch+1 == args.epochs):
        state = {
            'model': model.state_dict(),
            'epoch': epoch+1,
        }
        torch.save(state, checkpoint_path / f'{epoch+1}_param.t')

print(f'======= Experiment: {args.exp_name} is finished! ======')
show_loss(log_path, param='top-1', file_type='log', keyword='Retrieval', vis=False)

# save output feat for refinement
if args.output_feat:
    result_temp = Feat_Arch(args, checkpoint_path)
    queryset = FeatEmbTrain(args, mode='sar', dataset='train')
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    result_temp.feat_embed(model, query_loader, dataset='train')


