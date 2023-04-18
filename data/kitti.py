import hickle as hkl
import numpy as np
import os

# Data generator that creates sequences for input into PredNet.
class KITTI():
    def __init__(self, train, data_root, n_frames_input=10, n_frames_output=1, image_size=128):
        if train:
            data_file = os.path.join(data_root, 'X_train.hkl')
            source_file = os.path.join(data_root, 'sources_train.hkl')
        else:
            data_file = os.path.join(data_root, 'X_test.hkl')
            source_file = os.path.join(data_root, 'sources_test.hkl')
        print(data_file)
        print(source_file)
        image_size = (128, 160)
        self.train = train
        self.X = hkl.load(data_file)  # X will be like (n_images, nb_cols, nb_rows, nb_channels)
        self.sources = hkl.load(source_file) # source for each image so when creating sequences can assure that consecutive frames are from same video
        self.seq_len = n_frames_input + n_frames_output
        self.possible_starts = np.array([i for i in range(self.X.shape[0] - self.seq_len) if self.sources[i] == self.sources[i + self.seq_len - 1]])
        self.N_sequences = len(self.possible_starts)
        self.seed_set = False


    def get_sequence(self, index):
        idx = np.random.randint(self.N_sequences)
        seq = (self.X[idx:idx+self.seq_len]/255.).astype('float64')
        # print(seq.size)
        return seq

    def __getitem__(self, index):
        if not self.seed_set:
            self.seed_set = True
            np.random.seed(index)
        return self.get_sequence(index)

    def __len__(self):
        if self.train: return int(self.N_sequences/100)# arbitrary
        else: return int(self.N_sequences)# arbitrary
