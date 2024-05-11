""" Dataset of the OSDataset.

"""

import numpy as np
from PIL import Image
import torch.utils.data as data
import random
import torch 
import matplotlib.pyplot as plt
import json
import os, re
from utils.transform import build_transform
from pathlib import Path

class OSDataTrain(data.Dataset):

    def __init__(self, args, mode='opt') -> None:
        super().__init__()
        print('build the OSDataset')
        data_path = Path(args.data_path)
        self.img_dir = data_path / 'train'
        self.mode = mode
        self.transform = build_transform(args.transform_test)
        self.sar_infos = []
        self.opt_infos = []

        sar_ids = []
        opt_ids = []

        for imgs in os.listdir(self.img_dir):
            # fliter the modality
            img_name = imgs.split('.')[0]
            img_id=int(img_name[3:])
            anno = dict(file_path=imgs, id=img_id)
            if img_name[:3] == 'sar':
                self.sar_infos.append(anno)
                sar_ids.append(img_id)
            elif img_name[:3] == 'opt':
                self.opt_infos.append(anno)
                opt_ids.append(img_id)
            else:
                raise TypeError
            
        try: #transform sar different from optical
            self.transform_sar = build_transform(args.transform_train_sar)
        except:
            self.transform_sar = self.transform


        try:
            self.transform_filter = build_transform(args.transform_train_aux)
            self.use_aux = True
        except:
            self.transform_filter = None
            self.use_aux = False
            
        self.nclass = len(opt_ids)
        self.train_sar_label = torch.tensor(sar_ids)


    def __getitem__(self, index):
        sar_anna = self.sar_infos[index] # load scene infos
        aligned_id = sar_anna['id'] 
        opt_name = f'opt{aligned_id}.png'
        
        img1 = Image.open(self.img_dir / opt_name).convert('RGB')
        img2 = Image.open(self.img_dir / sar_anna['file_path']).convert('RGB')
        

        opt_img = self.transform(img1)
        sar_img = self.transform_sar(img2) 

        opt_data = dict(feat=opt_img, id=aligned_id, pos=0, img_name=aligned_id)
        sar_data = dict(feat=sar_img, id=aligned_id, pos=0, img_name=aligned_id)
        data_dict = dict(opt=opt_data, sar=sar_data)

        if self.use_aux:
            fil_opt = self.transform_filter(img1)
            fil_sar = self.transform_filter(img2)
            fil_opt_data = dict(feat=fil_opt, id=aligned_id, pos=0)
            fil_sar_data = dict(feat=fil_sar, id=aligned_id, pos=0)
            data_dict.update(fil_opt=fil_opt_data, fil_sar=fil_sar_data)


        return data_dict
    
    def __len__(self):
        return len(self.sar_infos)

    

class OSDataTest(data.Dataset):

    def __init__(self, args, mode='opt') -> None:
        super().__init__()
        print('build the OSDataset')
        data_path = Path(args.data_path)
        self.img_dir = data_path / 'test'
        self.mode = mode
        self.transform = build_transform(args.transform_test)
        self.obj_infos = []
        for imgs in os.listdir(self.img_dir):
            # fliter the modality
            img_name = imgs.split('.')[0]
            if img_name[:3] == mode:
                anno = dict(file_path=imgs, 
                            id=int(img_name[3:]))
                self.obj_infos.append(anno)


    def __getitem__(self, index):
        # img, target = self.test_image[index], self.test_label[index]
        anna = self.obj_infos[index] # load scene infos
        img = Image.open(self.img_dir / anna['file_path']).convert('RGB')
        target = anna['id']
        pos = 0 # no pose in this OSdataset
        img = self.transform(img)
        return {self.mode: dict(feat=img, id=target, pos=pos, img_name=target)}
    
    def __len__(self):
        return len(self.obj_infos)