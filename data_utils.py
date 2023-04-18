import torch
import socket
import numpy as np
import cv2
import matplotlib
matplotlib.use('agg')
hostname = socket.gethostname()


def load_dataset(opt):
    global train_data, test_data
    if opt.dataset == 'ucf':
        from data.ucf import UCF
        train_data = UCF(
                train=True,
                data_root='%s/UCF-101' % opt.data_root,
                n_frames_input=opt.n_past+2,
                n_frames_output=opt.n_future,
                image_size=opt.image_width)
        test_data =UCF(
                train=False,
                data_root='%s/UCF-101' % opt.data_root,
                n_frames_input=opt.n_past+2,
                n_frames_output=opt.n_eval,
                image_size=opt.image_width)

    elif opt.dataset == 'human':
        from data.human import Human
        train_data = Human(
            train=True,
            data_root='%s/human' % opt.data_root,
            n_frames_input=opt.n_past+2,
            n_frames_output=opt.n_future,
            image_size=opt.image_width)
        test_data = Human(
            train=False,
            data_root='%s/human' % opt.data_root,
            n_frames_input=opt.n_past+2,
            n_frames_output=opt.n_eval,
            image_size=opt.image_width)

    return train_data, test_data


def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
             hasattr(arg, "__iter__")))


def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images) - 1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding:
                      (i + 1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images) - 1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding:
                         (i + 1) * y_dim + i * padding].copy_(image)
        return result


def make_image(tensor):
    tensor = tensor.cpu().clamp(0, 1)
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
    # pdb.set_trace()
    # return scipy.misc.toimage(tensor.numpy(), high=255., channel_axis=0)
    return tensor.numpy()


def save_image(filename, tensor):
    img = make_image(tensor.permute(1,2,0))
    img = img[...,::-1] #RGB to BGR
    if img.shape[2] == 2:
        img = img[:, :, :1]
    cv2.imwrite(filename, img*255.)


def save_tensors_image(filename, inputs, padding=1):
    images = image_tensor(inputs, padding)
    return save_image(filename, images)

def plot(gt_seq, gen_seq, fname):
    """x: list:T, [BS, C, 3, W, H]"""
    to_plot = []
    batchs = gt_seq.shape[0]
    nrow = min(batchs, 10)
    length = gt_seq.shape[1]
    for i in range(nrow):
        row = []
        for t in range(length):
            row.append(gt_seq[i, t])
        to_plot.append(row)
        s_list = [0]
        for ss in range(len(s_list)):
            row = []
            for t in range(length):
                row.append(gen_seq[i,t])
            to_plot.append(row)
    save_tensors_image(fname, to_plot)