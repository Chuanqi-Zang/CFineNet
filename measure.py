from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
from lpips import lpips
import torch


def f_mse(video_true, video_test, total_mse):
    for i in range(video_true.shape[1]):
        total_mse[i] += np.mean((video_true[:,i] - video_test[:,i]) ** 2, axis=0).sum()
    return total_mse

def f_mae(video_true, video_test, total_mae):
    for i in range(video_true.shape[1]):
        total_mae[i] += np.mean(np.abs(video_true[:,i] - video_test[:,i]), axis=0).sum()
    return total_mae


def f_ssim(video_true, video_test, total_ssim):
    for i in range(0, video_true.shape[1]):
        for a in range(0, video_true.shape[0]):
            for b in range(0, video_true.shape[2]):
                total_ssim[i] += ssim(video_true[a, i, b], video_test[a, i, b]) / \
                              (video_true.shape[0] * video_true.shape[2])
    return total_ssim


def f_psnr(video_true, video_test, total_psnr):
    video_test = np.minimum(np.maximum(video_test, 0), 1)
    video_true = np.minimum(np.maximum(video_true, 0), 1)
    for i in range(0, video_true.shape[1]):
        for a in range(0, video_true.shape[0]):
            total_psnr[i] += psnr(video_true[a, i], video_test[a, i]) / video_true.shape[0]
    return total_psnr


def f_fvd(video_true, video_test):
    #https://github.com/google-research/google-research/tree/master/frechet_video_distance
    return


def f_lpips(video_true, video_test, total_fpips):
    # functional call
    if video_true.shape[3] == 1 or video_true.shape[3] == 3:
        lipip_func = lpips.LPIPS(net='alex').cuda()
        for i in range(0, video_true.shape[2]):
            for a in range(0, video_true.shape[1]):
                l= lipip_func.forward(video_true[:,a,i,:,:,:], video_test[:,a,i,:,:,:])
                l = torch.mean(l, dim=(0, 1, 2, 3))/ video_true.shape[1]
                total_fpips[i] += l.cpu().numpy()
    else: # channel==2
        total_fpips=0
    return total_fpips
