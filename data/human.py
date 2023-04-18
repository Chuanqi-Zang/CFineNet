import random
import os
import numpy as np
import cv2
# import socket
# import torch
# from scipy import misc
# import torchfile
from PIL import Image
# from torchvision import transforms

class Human(object):

    def __init__(self, train, data_root, n_frames_input=4, n_frames_output=4, image_size=128):
        self.data_root = data_root
        self.n_input = n_frames_input
        # self.n_output = n_frames_output
        self.seq_len = n_frames_input+n_frames_output

        self.image_size = image_size[0]
        self.classes = ['Walking']
        # self.samples = list(range(1,5))
        self.dirs = os.listdir(self.data_root)
        #papers using 1-16 for training, 17-25 for testing
        if train:
            self.train = True
            self.data_type = 'train'
            self.subjects = ['s1', 's5', 's6', 's7', 's8']
        else:
            self.train = False
            self.data_type = 'test'
            self.subjects = ['s9', 's11']
        self.seed_set = False
        self.dirs= []
        # self.file = '%s/%s/%s_meta%dx%d.t7' % (self.data_root, c, self.data_type, image_size, image_size)
        for subject in self.subjects:
            print ('load data...', self.data_root+'/'+subject)
            filenames = os.listdir(self.data_root+'/'+subject)
            filenames.sort()
            seq = []
            frames = []
            episode = 0
            for filename in filenames:
                # S1_Directions.54138969_000001.jpg
                fix, nums, format = filename.split('.')
                scenario = fix.split('_')[1]
                epi, num = nums.split('_')

                if scenario not in self.classes:
                    continue
                if episode == 0:
                    episode = epi
                if epi == episode:
                    file_path = os.path.join(self.data_root, subject, filename)
                    frames.append(file_path) # record images in one sequence
                else:
                    seq.append(frames) # record multi sequence
                    frames = []
                    episode = epi
            self.dirs.append(seq) #recoed multi subjects


    def get_sequence(self, index):
        t = self.seq_len

        c_idx = np.random.randint(len(self.subjects))
        seq_idx = np.random.randint(len(self.dirs[c_idx]))
        vid = self.dirs[c_idx][seq_idx]
        frame_idx = np.random.randint(0, len(vid)-t)
        seq = []
        for i in range(frame_idx, frame_idx+t):
            im = Image.open(vid[i])
            #[1000,1000,3] -> [250:750,250:750,3]
            point_lu = im.size[0] // 4
            point_rl = im.size[0] - im.size[0] // 4
            im = im.crop((point_lu, point_lu, point_rl, point_rl))
            im = im.resize((self.image_size, self.image_size), Image.ANTIALIAS)
            im = (np.array(im)/255.).astype('float64')
            seq.append(im)
            # seq.append(im[:, :, :].reshape(self.image_size, self.image_size, 3))
        seq = np.array(seq)

        return seq

    def __getitem__(self, index):
        if not self.seed_set:
            self.seed_set = True
            random.seed(index)
            np.random.seed(index)
        return self.get_sequence(index)

    def __len__(self):
        return len(self.dirs) * 300# arbitrary


