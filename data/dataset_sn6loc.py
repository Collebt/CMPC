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


class SN6LocMultiAlignData(data.Dataset):
    """training dataset of  the sar-rgb retrieval task
    multi-id SAR image would aligned to 1 optical image.
    random sample the sar image, and find the aligned optical image via the same id

    """
    def __init__(self, args):
        print("begin to load training data!")
        # Load training images (path) and labels
        data_path = Path(args.data_path)
        opt_anno_dir = data_path / f'train/annotations/opt'
        sar_anno_dir = data_path / f'train/annotations/sar'

        sar_ids = []
        opt_ids = [] 
        for fp in os.listdir(sar_anno_dir):
            bbox_info = json.load(open( sar_anno_dir / fp)) #get the sar id from json
            sar_ids.append(bbox_info['id'])
            # print(f"add id: {bbox_info['id']}")

        for fp in os.listdir(opt_anno_dir):
            bbox_info = json.load(open( opt_anno_dir / fp)) #get the optical id from json
            opt_ids.append(bbox_info['id'])
            # print(f"add id: {bbox_info['id']}")

        opt_json_files = os.listdir(opt_anno_dir)
        sar_json_files = os.listdir(sar_anno_dir)
                    
        self.opt_anno_dir = opt_anno_dir
        self.sar_anno_dir = sar_anno_dir
        self.opt_img_dir = data_path / f'train/imgs/opt'
        self.sar_img_dir = data_path / f'train/imgs/sar'
        self.opt_json_files = opt_json_files #named x.json, x is start from 0, indicate the id of the optical images
        self.sar_json_files = sar_json_files

        
        self.transform = build_transform(args.transform_train_base)

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
        
        sar_anno = json.load(open(self.sar_anno_dir / self.sar_json_files[index]))  #get sar sample json 
        aligned_id = sar_anno['id'] #in aligned dataset, optical ids are as same as sar ids
        opt_anno = json.load(open(self.opt_anno_dir / f'{aligned_id}.json'))#fing the opt_json with the sar id 
        
        opt_pos = torch.tensor([opt_anno['pos']['x'], opt_anno['pos']['y']])
        sar_pos = torch.tensor([sar_anno['pos']['x'], sar_anno['pos']['y']])

        img1 = Image.open(self.opt_img_dir / opt_anno['file_path']) #optical image
        img2 = Image.open(self.sar_img_dir / sar_anno['file_path']) #SAR image

        opt_img = self.transform(img1)
        sar_img = self.transform_sar(img2) 
        opt_data = dict(feat=opt_img, id=aligned_id, pos=opt_pos, img_name=int(opt_anno['file_path'].split('.')[0]))
        sar_data = dict(feat=sar_img, id=aligned_id, pos=sar_pos, img_name=int(sar_anno['file_path'].split('.')[0]))
        data_dict = dict(opt=opt_data, sar=sar_data)

        if self.use_aux:
            fil_opt = self.transform_filter(img1)
            fil_sar = self.transform_filter(img2)
            fil_opt_data = dict(feat=fil_opt, id=aligned_id, pos=opt_pos)
            fil_sar_data = dict(feat=fil_sar, id=aligned_id, pos=sar_pos)
            data_dict.update(fil_opt=fil_opt_data, fil_sar=fil_sar_data)

        return data_dict

    def __len__(self):
        return len(self.sar_json_files)


class SN6LocNeighborData(data.Dataset):
    """training dataset of  the sar-rgb retrieval task
    SAR images in arbitrary location, optical images are overlaped with 50%.
    random sample the sar image, and find the aligned optical image via the same id

    """
    def __init__(self, args):
        print("begin to load training data!")
        # Load training images (path) and labels
        data_path = Path(args.data_path)
        opt_anno_dir = data_path / f'train/annotations/opt'
        sar_anno_dir = data_path / f'train/annotations/sar'

        sar_ids = []
        opt_ids = [] 
        for fp in os.listdir(sar_anno_dir):
            bbox_info = json.load(open( sar_anno_dir / fp)) #get the sar id from json
            sar_ids.append(bbox_info['id'])
            # print(f"add id: {bbox_info['id']}")

        for fp in os.listdir(opt_anno_dir):
            bbox_info = json.load(open( opt_anno_dir / fp)) #get the optical id from json
            opt_ids.append(bbox_info['id'])
            # print(f"add id: {bbox_info['id']}")

        opt_json_files = os.listdir(opt_anno_dir)
        sar_json_files = os.listdir(sar_anno_dir)
                    
        self.opt_anno_dir = opt_anno_dir
        self.sar_anno_dir = sar_anno_dir
        self.opt_img_dir = data_path / f'train/imgs/opt'
        self.sar_img_dir = data_path / f'train/imgs/sar'
        self.opt_json_files = opt_json_files #named x.json, x is start from 0, indicate the id of the optical images
        self.sar_json_files = sar_json_files

        
        self.transform = build_transform(args.transform_train_base)

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
        
        sar_anno = json.load(open(self.sar_anno_dir / self.sar_json_files[index]))  #get sar sample json 
        match_id = sar_anno['id'] #In arbitrary dataset, the nearest optical ids are as same as SAR ids, but not aligned.
        opt_anno = json.load(open(self.opt_anno_dir / f'{match_id}.json'))#fing the opt_json with the sar id 
        
        opt_pos = torch.tensor([opt_anno['pos']['x'], opt_anno['pos']['y']])
        sar_pos = torch.tensor([sar_anno['pos']['x'], sar_anno['pos']['y']])

        img1 = Image.open(self.opt_img_dir / opt_anno['file_path']) #optical image
        img2 = Image.open(self.sar_img_dir / sar_anno['file_path']) #SAR image

        opt_img = self.transform(img1)
        sar_img = self.transform_sar(img2) 
        opt_data = dict(feat=opt_img, id=match_id, pos=opt_pos, img_name=int(opt_anno['file_path'].split('.')[0]))
        sar_data = dict(feat=sar_img, id=match_id, pos=sar_pos, img_name=int(sar_anno['file_path'].split('.')[0]))
        data_dict = dict(opt=opt_data, sar=sar_data)

        # semi optical 
        neighbor_id = random.choice(sar_anno['near_id'][1:4])
        anno = json.load(open(self.opt_anno_dir / f'{neighbor_id}.json'))#fing the opt_json with the sar id 
        img = Image.open(self.opt_img_dir / anno['file_path'])
        img = self.transform(img)
        pos = torch.tensor([anno['pos']['x'], anno['pos']['y']])
        
        semi_data = dict(feat=img, id=neighbor_id, pos=pos, img_name=int(anno['file_path'].split('.')[0]))
        data_dict.update(semi_opt=semi_data)

        # auxiliary augnment
        if self.use_aux:
            fil_opt = self.transform_filter(img1)
            fil_sar = self.transform_filter(img2)
            fil_opt_data = dict(feat=fil_opt, id=match_id, pos=opt_pos)
            fil_sar_data = dict(feat=fil_sar, id=match_id, pos=sar_pos)
            data_dict.update(fil_opt=fil_opt_data, fil_sar=fil_sar_data)

        return data_dict

    def __len__(self):
        return len(self.sar_json_files)



class SN6LocTestData(data.Dataset):
    """test the sar-rgb localization accuracy with the align data with the same id.
    """
    def __init__(self, args, mode='opt'):
        # print("begin to load testing data!")
        # Load training images (path) and labels
        data_path = Path(args.data_path)

        anno_dir = data_path / 'test/annotations' / mode
        json_files =  os.listdir(anno_dir)

        self.anno_dir = anno_dir
        self.img_dir = data_path /'test/imgs'/ mode
        self.json_files = json_files

        self.transform = build_transform(args.transform_test)
        
        mode = 'opt' if mode == 'rgb' else mode #switch all rgb name to opt
        self.mode = mode

    def __getitem__(self, index):
        # img, target = self.test_image[index], self.test_label[index]
        anna = json.load(open(self.anno_dir / self.json_files[index]))
        img = Image.open(self.img_dir / anna['file_path'])
        target = anna['id']
        pos = torch.tensor([anna['pos']['x'], anna['pos']['y']])
        img = self.transform(img)
        return {self.mode : dict(feat=img, id=target, pos=pos, img_name=int(anna['file_path'].split('.')[0]))}

    def __len__(self):
        return len(self.json_files)




class FeatEmbTrain(data.Dataset):
    """test the sar-rgb localization accuracy with the same area data
    """
    def __init__(self, args, mode='opt', dataset='train'):
        # print("begin to load testing data!")
        # Load training images (path) and labels
        data_path = Path(args.data_path)
        anno_dir = data_path / dataset / 'annotations' / mode
        
        #sort the data by file name (index of opt, name of sar)
        json_files =  os.listdir(anno_dir)
        ids = [int(re.findall('\d+', f)[0])  for f in  json_files]
        sort_ids = np.argsort(ids)
        json_files = [json_files[i] for i in sort_ids]

        self.anno_dir = anno_dir
        self.img_dir = data_path / dataset/ 'imgs' / mode
        self.json_files = json_files

        self.transform = build_transform(args.transform_test)
        self.mode = mode

    def __getitem__(self, index):
        # img, target = self.test_image[index], self.test_label[index]
        anna = json.load(open(self.anno_dir / self.json_files[index]))
        img = Image.open(self.img_dir / anna['file_path'])
        target = anna['id']
        pos = torch.tensor([anna['pos']['x'], anna['pos']['y']])
        img = self.transform(img)
        return {self.mode : dict(feat=img, id=target, pos=pos, img_name=int(anna['file_path'].split('.')[0]))}

    def __len__(self):
        return len(self.json_files)
    


class GNN_PostTrainDataset(data.Dataset):

    def __init__(self, args):

        #NOTE: make sure the id of ref_id is sorted abscending
        data_path = Path(args.emb_feat_path) 
        data = torch.load(data_path / 'embed_feats' )

        opt_data, sar_data, sim_mat = data['opt'], data['sar'], data['sim_mat']

        _, self.topk_idx = sim_mat.topk(k=args.node_topk)

        self.ref_feat, self.ref_id, self.ref_pos = opt_data['feat'], opt_data['id'], opt_data['pos']
        self.que_feat, self.que_id, self.que_pos = sar_data['feat'], sar_data['id'], sar_data['pos']
        self.que_name = sar_data['name']

        if 'test' in args.emb_feat: # set the id of test set to 0
            self.ref_id -= 6328
            self.que_id -= 6328
            

    def __len__(self):
        return len(self.que_id)
        
    def __getitem__(self, index):

        out_feat = self.que_feat[index]
        que_pos = self.que_pos[index]
        gt_id = self.que_id[index]

        return {'sar': dict(out_feat=out_feat, id=gt_id, pos=que_pos)}