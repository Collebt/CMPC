import numpy as np
from PIL import Image
import torch.utils.data as data
import torch 
import matplotlib.pyplot as plt
import json
import os
from utils.transform import build_transform
from pathlib import Path



class OSData(data.Dataset):
    """Osdataset aligned data, with normalization transform
    """
    def __init__(self, args, opt_index=None, sar_index=None):
        data_dir = args.data_path
        print("begin to load training data!")
        # Load training images (path) and labels
        train_opt_image = np.load(data_dir + 'new_train_opt_img.npy')
        self.train_opt_label = np.load(data_dir + 'new_train_opt_label.npy')

        train_sar_image = np.load(data_dir + 'new_train_sar_img.npy')
        self.train_sar_label = np.load(data_dir + 'new_train_sar_label.npy')
        print("done")

        # BGR to RGB
        self.train_opt_image = train_opt_image
        self.train_sar_image = train_sar_image

        self.transform_filter = build_transform(args.transform_train_aux)
        self.transform_original = build_transform(args.transform_train_base)
               
    def __getitem__(self, index):
        img1, target1 = self.train_opt_image[index], self.train_opt_label[index]
        img2, target2 = self.train_sar_image[index], self.train_sar_label[index]

        img_opt = self.transform_original(img1)
        img_sar = self.transform_original(img2)

        fil_opt = self.transform_filter(img1)
        fil_sar = self.transform_filter(img2)

        opt_data = dict(feat=img_opt, id=target1)
        sar_data = dict(feat=img_sar, id=target2)
        fil_opt_data = dict(feat=fil_opt, id=target1)
        fil_sar_data = dict(feat=fil_sar, id=target2)
        return dict(opt=opt_data, sar=sar_data, fil_opt=fil_opt_data, fil_sar=fil_sar_data)

    def __len__(self):
        return len(self.train_opt_label)

class SN6AlignBase(data.Dataset):
    """sn6 aligned data, base normalization
    """
    def __init__(self, args, opt_index=None, sar_index=None):
        print("begin to load training data!")
        # Load training images (path) and labels
        data_dir = Path(args.data_path)
        self.train_opt_image = torch.load(data_dir / 'opt_ori_train.pt')
        self.train_sar_image = torch.load(data_dir / 'sar_ori_train.pt')

        self.train_sar_label = torch.arange(self.train_opt_image.shape[0])
        self.train_opt_label = torch.arange(self.train_sar_image.shape[0])

        print("done")
        self.transform = build_transform(args.transform_train_base)

    def __getitem__(self, index):
        img1, target1 = self.train_opt_image[index], self.train_opt_label[index]
        img2, target2 = self.train_sar_image[index], self.train_sar_label[index]
        
        img_opt = self.transform(img1) #normalize the image
        img_sar = self.transform(img2)
        
        opt_data = dict(feat=img_opt, id=target1)
        sar_data = dict(feat=img_sar, id=target2)
        return dict(opt=opt_data, sar=sar_data)

    def __len__(self):
        return len(self.train_opt_label)


class SN6AlignAuxData(data.Dataset):
    """sn6 aligned data, with auxillary transformation 
    """
    def __init__(self, args, opt_index=None, sar_index=None):
        print("begin to load training data!")
        data_dir = Path(args.data_path)
        # Load training images (path) and labels
        self.train_opt_image = torch.load(data_dir / 'opt_ori_train.pt')
        self.train_sar_image = torch.load(data_dir / 'sar_ori_train.pt')


        # self.train_fil_opt = torch.load(data_dir / 'opt_cha_exc_train.pt')
        # self.train_fil_sar = torch.load(data_dir / 'sar_cha_exc_train.pt')


        self.train_sar_label = torch.arange(self.train_opt_image.shape[0])
        self.train_opt_label = torch.arange(self.train_sar_image.shape[0])

        print("done")

        self.transform_original = build_transform(args.transform_train_base)
        self.transform_filter = build_transform(args.transform_train_aux)

    def __getitem__(self, index):
        img1, target1 = self.train_opt_image[index], self.train_opt_label[index]
        img2, target2 = self.train_sar_image[index], self.train_sar_label[index]

        # fil_opt = self.train_fil_opt[index]  #pre transform in preprocess 
        # fil_sar = self.train_fil_sar[index]
        #1``

        img_opt = self.transform_original(img1)
        img_sar = self.transform_original(img2)
        fil_opt = self.transform_filter(img1)
        fil_sar = self.transform_filter(img2)

        opt_data = dict(feat=img_opt, id=target1)
        sar_data = dict(feat=img_sar, id=target2)
        fil_opt_data = dict(feat=fil_opt, id=target1)
        fil_sar_data = dict(feat=fil_sar, id=target2)
        return dict(opt=opt_data, sar=sar_data, fil_opt=fil_opt_data, fil_sar=fil_sar_data)

    def __len__(self):
        return len(self.train_opt_label)

class SN6PosAlignData(data.Dataset):
    """provide the sar-optical localization accuracy with the align data with the same id.
    """
    def __init__(self, args):
        print("begin to load testing data!")
        # Load training images (path) and labels
        data_dir = Path(args.data_path)
        json_paths = []
        ids = []
        sar_json_paths = []
        opt_anno_dir = data_dir/ f'annotations/opt'
        sar_anno_dir = data_dir/ f'annotations/sar'
        for fp in os.listdir(opt_anno_dir):
                bbox_info = json.load(open( opt_anno_dir / fp))
                if bbox_info['id'] < 1500: # get the train set < 1500 (1500/2171)
                    json_paths.append(fp)
                    sar_json_paths.append(f'0_{fp}')
                    ids.append(bbox_info['id'])
                    
        self.opt_anno = opt_anno_dir
        self.sar_anno = sar_anno_dir
        self.data_dir = data_dir
        self.json_paths = json_paths
        self.sar_json_paths = sar_json_paths

        self.transform = build_transform(args.transform_train_base)
        self.transform_filter = build_transform(args.transform_train_aux)

        self.train_opt_label = torch.tensor(ids)
        self.train_sar_label = torch.tensor(ids)


    def __getitem__(self, index):

        opt_anna = json.load(open(self.opt_anno / self.json_paths[index]))
        sar_anna = json.load(open(self.sar_anno / self.sar_json_paths[index])) 

        img1 = Image.open(self.data_dir / opt_anna['file_path'])
        img2 = Image.open(self.data_dir / sar_anna['file_path'])

        opt_img = self.transform(img1)
        sar_img = self.transform(img2) 
        fil_opt = self.transform_filter(img1)
        fil_sar = self.transform_filter(img2)

        target = opt_anna['id'] #in aligned dataset, optical ids are as same as sar ids
        pos = torch.tensor([opt_anna['pos']['x'], opt_anna['pos']['x']])

        opt_data = dict(feat=opt_img, id=target, pos=pos)
        sar_data = dict(feat=sar_img, id=target, pos=pos)
        fil_opt_data = dict(feat=fil_opt, id=target, pos=pos)
        fil_sar_data = dict(feat=fil_sar, id=target, pos=pos)

        return dict(opt=opt_data, sar=sar_data, fil_opt=fil_opt_data, fil_sar=fil_sar_data)

    def __len__(self):
        return len(self.json_paths)


#####################################Test Dataset####################################
class TestData(data.Dataset):
    def __init__(self, args, mode = 'sar', img_size = (256,256)):
        print("begin to load testing data!")
        data_dir = args.data_path
        # Load training images (path) and labels
        self.mode = mode
        if self.mode == 'sar':
            self.test_image = np.load(data_dir + 'new_test_sar_img.npy')
            self.test_label = np.load(data_dir + 'new_test_sar_label.npy')
        else:
            self.test_image = np.load(data_dir + 'new_test_opt_img.npy')
            self.test_label = np.load(data_dir + 'new_test_opt_label.npy')
        # print("Dataset statistics:")
        # print("  ------------------------------")
        # print("  subset   | # ids | # images")
        # print("  ------------------------------")
        # print("   q/g     | {:5d} | {:8d}".format(self.test_image.shape[0], self.test_label.shape[0]))
        # print("  ------------------------------")
        # print("done")
        self.transform = build_transform(args.transform_test)

    def __getitem__(self, index):
        img, target = self.test_image[index], self.test_label[index]
        img = self.transform(img)

        return {self.mode : dict(feat=img, id=target)}

    def __len__(self):
        return self.test_label.shape[0]

class SN6TestData(data.Dataset):
    def __init__(self, args, mode = 'sar', img_size = (256,256)):
        print("begin to load testing data!")
        # Load training images (path) and labels
        data_dir = Path(args.data_path)
        self.mode = mode
        if self.mode == 'sar':
            self.test_image = torch.load(data_dir / 'sar_ori_test.pt')
            self.test_label = torch.arange(self.test_image.shape[0])
        else:
            self.test_image = torch.load(data_dir / 'opt_ori_test.pt')
            self.test_label = torch.arange(self.test_image.shape[0])
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("   q/g     | {:5d} | {:8d}".format(self.test_image.shape[0], self.test_label.shape[0]))
        print("  ------------------------------")
        print("done")
        self.transform = build_transform(args.transform_test)

    def __getitem__(self, index):
        img, target = self.test_image[index], self.test_label[index]
        img = self.transform(img)
        return {self.mode : dict(feat=img, id=target)}

    def __len__(self):
        return self.test_label.shape[0] 

class SN6TestPosAlign(data.Dataset):
    """test the sar-optical localization accuracy with the align data with the same id.
    """
    def __init__(self, args, mode='opt'):
        print("begin to load testing data!")
        # Load training images (path) and labels
        data_dir = Path(args.data_path)
        json_paths = []
        anno_dir = data_dir/ f'annotations/{mode}'
        for fp in os.listdir(anno_dir):
                bbox_info = json.load(open( anno_dir / fp))
                if bbox_info['id'] > 1500: # get the test set > 1500 (1500/2171)
                    json_paths.append(fp)
        self.anno_dir = anno_dir
        self.data_dir = data_dir
        self.json_paths = json_paths
        self.transform = build_transform(args.transform_test)
        self.mode = mode

    def __getitem__(self, index):
        # img, target = self.test_image[index], self.test_label[index]
        anna = json.load(open(self.anno_dir / self.json_paths[index]))
        img = Image.open(self.data_dir / anna['file_path'])
        target = anna['id']
        pos = torch.tensor([anna['pos']['x'], anna['pos']['x']])
        img = self.transform(img)
        return {self.mode : dict(feat=img, id=target, pos=pos)}

    def __len__(self):
        return len(self.json_paths)


