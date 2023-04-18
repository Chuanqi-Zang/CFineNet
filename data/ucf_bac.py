import random
import os
import numpy as np
import torch
import os.path
from PIL import Image

class UCF(object):

    def __init__(self, train, data_root, n_frames_input=8, n_frames_output=10, image_size=(120, 160)):
        self.data_root = data_root + '/preprocess'
        self.n_input = n_frames_input
        # self.n_output = n_frames_output
        self.seq_len = n_frames_input+n_frames_output
        self.image_size = image_size
        self.classes = os.listdir(self.data_root)

        if train:
            self.train = True
            self.data_type = 'train'
            self.persons = list(range(1, 23))
            self.ite_len=500
        else:
            self.train = False
            self.persons = list(range(23, 26))
            self.data_type = 'test'
            self.ite_len=1000
        self.dirs= []
        # self.file = '%s/%s/%s_meta%dx%d.t7' % (self.data_root, c, self.data_type, image_size, image_size)
        for c in self.classes:
            p = []
            for person in self.persons:
                q = []
                SAMPLE_LEN = 10
                for sample in range(1, SAMPLE_LEN):
                    video_dir = '%s/%s/v_%s_g%02d_c%02d' % (self.data_root, c, c, person, sample)
                    if not os.path.exists(video_dir):
                        break
                    for i in range(len(os.listdir(video_dir))):
                        q.append(video_dir+'/image-%03d_%dx%d.png' % (i, self.image_size[1], self.image_size[0]))
                p.append(q)
            self.dirs.append(p)
        self.seed_set = False
        print("success load ucf-101")

    def get_sequence(self):
        t = self.seq_len
        c_idx = np.random.randint(len(self.classes))
        vid_idx = np.random.randint(len(self.dirs[c_idx]))
        vid = self.dirs[c_idx][vid_idx]
        seq_idx = np.random.randint(len(vid)-t*2) # skip seqeunces that are too short

        seq = []
        for i in range(seq_idx, seq_idx+t*2, 2):
            # fname = '%s/%s/%s/%s' % (self.data_root, c, vid, vid[i])
            im = Image.open(vid[i])
            im = (np.array(im)/255.).astype('float64')
            seq.append(im)
        seq = np.array(seq)
        # image_seq = [index, seq[:self.n_input, :, :, :], seq[self.n_input:, :, :, :]]
        return seq

    def __getitem__(self, index):
        if not self.seed_set:
            self.seed_set = True
            random.seed(index)
            np.random.seed(index)
            # torch.manual_seed(index)
        return torch.from_numpy(self.get_sequence())

    def __len__(self):
        return self.ite_len



import random
import os
import numpy as np
import torch
import os.path
from PIL import Image

class UCF(object):

    def __init__(self, train, data_root, n_frames_input=8, n_frames_output=10, image_size=(120, 160)):
        self.data_root = data_root + '/preprocess'
        self.n_input = n_frames_input
        # self.n_output = n_frames_output
        self.seq_len = n_frames_input+n_frames_output
        self.image_size = image_size
        self.classes = os.listdir(self.data_root)

        if train:
            self.train = True
            self.data_type = 'train'
            self.persons = list(range(1, 26))
            self.ite_len=500
            f = open(data_root + '/ucf_train.txt', 'r')
            train_list = f.readlines()
            self.img_list = train_list
            print("success load ucf-101 train set, length: %d"%len(train_list))
        else:
            self.train = False
            self.persons = list(range(1, 26))
            self.data_type = 'test'
            self.ite_len=1000
            f = open(data_root + '/ucf_test.txt', 'r')
            test_list = f.readlines()
            self.img_list = test_list
            print("success load ucf-101 test set, length: %d"%len(test_list))
        self.dirs= []

        for s in self.img_list:
            q=[]
            c = s.split('_')[1]
            video_dir = '%s/%s/%s' % (self.data_root, c, s.replace('\n', ''))
            for i in range(len(os.listdir(video_dir))):
                # q.append(video_dir + '/image-%03d_%dx%d.png' % (i, 160, 120))
                q.append(video_dir + '/image-%03d_%dx%d.png' % (i, 64, 85))
            self.dirs.append(q)
        self.seed_set = False


    def get_sequence(self):
        t = self.seq_len
        c_idx = np.random.randint(len(self.dirs))
        vid = self.dirs[c_idx]
        while(len(vid)<=t*2):
            c_idx = np.random.randint(len(self.dirs))
            vid = self.dirs[c_idx]
        seq_idx = np.random.randint(len(vid)-t*2) # skip seqeunces that are too short

        seq = []
        for i in range(seq_idx, seq_idx+t*2, 2):
            # fname = '%s/%s/%s/%s' % (self.data_root, c, vid, vid[i])
            im = Image.open(vid[i])
            im = im.resize((88, 64), Image.ANTIALIAS)
            im = (np.array(im)/255.).astype('float64')
            # print(im.shape)
            seq.append(im)
        seq = np.array(seq)
        # image_seq = [index, seq[:self.n_input, :, :, :], seq[self.n_input:, :, :, :]]
        return seq

    def __getitem__(self, index):
        if not self.seed_set:
            self.seed_set = True
            random.seed(index)
            np.random.seed(index)
            # torch.manual_seed(index)
        return torch.from_numpy(self.get_sequence())

    def __len__(self):
        return self.ite_len
