# from cv2.ximgproc import guidedFilter, rollingGuidanceFilter
from torchvision import transforms
from PIL import Image
import random
import math
import numpy as np





class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation 

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)



class ChannelExchange(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self):
        self.choice = 4
        
    def __call__(self, img):
        index = random.randint(0, 3)
        if index == 0:
            # random select R Channel
            img[1, :, :] = img[0, :, :]
            img[2, :, :] = img[0, :, :]
        elif index == 1:
            # random select B Channel
            img[0, :, :] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif index == 2:
            # random select G Channel
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
        else:
            tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
            img[0, :, :] = tmp_img
            img[1, :, :] = tmp_img
            img[2, :, :] = tmp_img
        return img


class ChannelChoice(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self):
        self.choice = 4
        
    def __call__(self, img):
        index = random.randint(0, 3)

        if index <  3:
            # random select R Channel
            tmp_img = img[index]
        else:
            tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
        img[0, :, :] = tmp_img
        img[1, :, :] = tmp_img
        img[2, :, :] = tmp_img
        return img
    

class GuideFilter(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, r = 4, eps = 0.6 * 0.6):
        self.r = r
        self.eps = eps * 255 * 255

    def __call__(self, img):
        img = guidedFilter(np.float32(img), np.float32(img), self.r, self.eps)
        img  = img / img.max(0).max(0) #renormalize each channel to maxize value =1
        return img

class RollingGuidenceFilter(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, sigma = 150, r1 = 3, r2 = 4):
        self.sigma = sigma
        self.r1 = r1
        self.r2 = r2

    def __call__(self, img):
        rollingRes = img
        rollingRes = rollingGuidanceFilter(img, rollingRes, self.sigma, self.r1, self.r2)
        return rollingRes

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

TRANSFORM_DICT = {
    'RandomCrop200': transforms.RandomCrop((200, 200)),
    'RandomCrop384': transforms.RandomCrop((384, 384)),

    'Resize200': transforms.Resize((200, 200)),
    'Resize384': transforms.Resize((384,384), interpolation=3),
    'ResizePCB': transforms.Resize((384,192), interpolation=3),
    'RandomHorizontalFlip':transforms.RandomHorizontalFlip(),
    'RandomAffine90': transforms.RandomAffine(90),
    'Pad10': transforms.Pad(10, padding_mode='edge'),
    'ToTensor': transforms.ToTensor(),
    'ToPILImage': transforms.ToPILImage(),
    'normalize': normalize,
    'GuideFilter': GuideFilter(),
    # 'Pad10': Pad(10),
    'ChannelExchange': ChannelExchange(),
    "ChannelChoice": ChannelChoice(),
    # 'RandomErasing10':transforms.RandomErasing(probability = 10, mean=[0.0, 0.0, 0.0])


}

def build_transform(t_name_list):
    t_list = []
    for t_name in t_name_list:
        t_list.append(TRANSFORM_DICT[t_name])
    return transforms.Compose(t_list)