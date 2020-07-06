import random
import torch
from PIL import Image
from glob import glob
import os
import numpy as np

class CirclesLoad(torch.utils.data.Dataset):
    def __init__(self, img_root, img_transform,
                 split='train'):
        super(CirclesLoad, self).__init__()
        self.img_transform = img_transform
        # use about 8M images in the challenge dataset
        self.paths = glob('{:s}*.png'.format(img_root))
        if len(self.paths )== 0:
            assert False
        if split == 'train' :
            self.paths = self.paths[:(int)(len(self.paths)*.8)]
        else:
            self.paths = self.paths[-1*(int)(len(self.paths)*.2):]
    def getIndex(self,path):
        ind = -1
        for p in self.paths:
            ind+=1
            if p == path:
                return ind
        return -1
    

    def __getitem__(self, index):
        path = self.paths[index]
        gt_img = Image.open(path)
        gt_img = self.img_transform(gt_img.convert('RGB'))
        return gt_img, 0

    def __len__(self):
        return len(self.paths)
