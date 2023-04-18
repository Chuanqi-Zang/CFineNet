import random
import os
import numpy as np
import socket
import torch
from scipy import misc
# import torchfile
from PIL import Image
from torchvision import transforms

class KTH(object):

    def __init__(self, train, data_root, n_frames_input=10, n_frames_output=10, image_size=(128, 128)):
        self.data_root = data_root + '/processed'
        self.n_input = n_frames_input
        # self.n_output = n_frames_output
        self.seq_len = n_frames_input+n_frames_output
        self.image_size = image_size[0]
        self.classes = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
        self.samples = list(range(1,5))
        #papers using 1-16 for training, 17-25 for testing
        if train:
            self.train = True
            self.data_type = 'train'
            self.persons = list(range(1, 17))
        else:
            self.train = False
            self.persons = list(range(17, 26))
            self.data_type = 'test'

        self.dirs= []
        # self.file = '%s/%s/%s_meta%dx%d.t7' % (self.data_root, c, self.data_type, image_size, image_size)
        for j in range(len(self.classes)):
            p = []
            for person in self.persons:
                q = []
                for sample in self.samples:
                    c = self.classes[j]
                    if 'person%02d_%s_d%d' % (person, c, sample)=='person13_handclapping_d3': # dataset missing
                        break
                    ad = '%s/%s/person%02d_%s_d%d' % (self.data_root, c, person, c, sample)
                    for i in range(1, len(os.listdir(ad))):
                        q.append('%s/%s/person%02d_%s_d%d/image-%03d_%dx%d.png' % \
                                                  (self.data_root, c, person, c, sample, i, self.image_size, self.image_size))
                p.append(q)
            self.dirs.append(p)

        self.seed_set = False

    def get_sequence(self, index):
        t = self.seq_len
        c_idx = np.random.randint(len(self.classes))
        vid_idx = np.random.randint(len(self.dirs[c_idx]))
        vid = self.dirs[c_idx][vid_idx]
        seq_idx = np.random.randint(10, len(vid)-t) # skip seqeunces that are too short

        seq = [] 
        for i in range(seq_idx, seq_idx+t):
            # fname = '%s/%s/%s/%s' % (self.data_root, c, vid, vid[i])
            im = Image.open(vid[i])
            im = (np.array(im)/255.).astype('float64')
            seq.append(im[:, :, :].reshape(self.image_size, self.image_size, 3))
        seq = np.array(seq)
        # image_seq = [index, seq[:self.n_input, :, :, :], seq[self.n_input:, :, :, :]]
        return seq

    def __getitem__(self, index):
        if not self.seed_set:
            self.seed_set = True
            random.seed(index)
            np.random.seed(index)
        return self.get_sequence(index)

    def __len__(self):
        # print(len(self.persons))
        return len(self.persons) * 10# arbitrary

