import os
import numpy as np
import random
from PIL import Image
from torchvision import transforms
# import cv2
# from pywt import dwt2, idwt2
# import torch_dct
# import torch
# from data_utils import get_dct_matrix
# import torchsnooper
# @torchsnooper.snoop()

class RobotPush(object):
    
    """Data Handler that loads robot pushing data."""

    def __init__(self, data_root, train=True, seq_len=20, image_size=(64, 64)):
        self.root_dir = data_root
        self.train_pahse = train
        if train:
            self.data_dir = '%s/processed_data/train' % self.root_dir
            self.ordered = False
            self.ite_len = len(self.data_dir)*15
        else:
            self.data_dir = '%s/processed_data/test' % self.root_dir
            self.ordered = True
            self.ite_len = len(self.data_dir) * 30
        self.dirs = []
        for d1 in os.listdir(self.data_dir):
            for d2 in os.listdir('%s/%s' % (self.data_dir, d1)):
                self.dirs.append('%s/%s/%s' % (self.data_dir, d1, d2))
        self.seq_len = seq_len
        self.image_size = image_size[0]
        self.seed_is_set = False # multi threaded loading
        self.d = 0

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return self.ite_len


    def get_seq(self, index):
        if self.ordered:
            d = self.dirs[self.d]
            if self.d == len(self.dirs) - 1:
                self.d = 0
            else:
                self.d+=1
        else:
            d = self.dirs[np.random.randint(len(self.dirs))]
        image_seq = []
        P1 = 1 if self.train_pahse and random.random()>0.5 else 0
        P2 = 1 if self.train_pahse and random.random()>0.5 else 0
        for i in range(self.seq_len):
            fname = '%s/%d.png' % (d, i)

            im = Image.open(fname)
            #
            im = transforms.RandomVerticalFlip(p=P1)(im)
            im = transforms.RandomHorizontalFlip(p=P2)(im)

            im = im.resize((self.image_size, self.image_size), Image.ANTIALIAS)

            im = np.array(im)
            im = im.reshape(1, self.image_size, self.image_size, 3)
            # R = im[:,:,:,0]
            # G = im[:,:,:,1]
            # B = im[:,:,:,2]
            # im[:,:,:,0] = 0.299 * R + 0.587 * G + 0.114 * B
            # im = im[:,:,:,0].reshape(1, 64, 64, 1)
            image_seq.append((im/255.).astype('float64'))

            # im = np.array(im)/255.
            # im = im.astype('float32')
            # # im_ycrcb = cv2.cvtColor(im, cv2.COLOR_RGB2YCR_CB)
            # im_ycrcb = im
            # im_ycrcb_dct = []
            # for i in range(3):
            #     im_ycrcb_dct.append(cv2.dct(im_ycrcb[:,:,i]))
            #
            #     # # 对img进行haar小波变换：
            #     # cA, (cH, cV, cD) = dwt2(im_ycrcb[:,:,i], 'haar')
            #     # # 将各个子图拼接(低频cA取值范围[0,510],高频[-255,255])
            #     # AH = np.concatenate([cA, cH ], axis=1)  # axis=1表示列拼接
            #     # VD = np.concatenate([cV , cD], axis=1)
            #     # img = np.concatenate([AH, VD], axis=0)
            #
            #     # im_ycrcb_dct.append(img)
            # im_ycrcb_dct = np.array(im_ycrcb_dct).transpose(1,2,0)
            #
            # # im_ycrcb = np.transpose(im_ycrcb.reshape(1, 64, 64, 3), (0, 2, 3, 1))
            # # dct_m, idct_m = get_dct_matrix(N = 64)
            # # im_ycrcb_dct = np.dot(np.dot(dct_m, im_ycrcb), np.transpose(dct_m))
            # # im_ycrcb_dct = np.transpose(im_ycrcb_dct, (0, 3, 1, 2))
            # im_ycrcb_dct = np.expand_dims(im_ycrcb_dct, 0)
            # image_seq.append(im_ycrcb_dct)

        image_seq = np.concatenate(image_seq, axis=0)

        # image_seq = [index, image_seq[:10,:,:,:], image_seq[10:20,:,:,:]]

        return image_seq


    def __getitem__(self, index):
        self.set_seed(index)
        return self.get_seq(index)

