import warnings
warnings.filterwarnings('ignore')
import os
import time
import random
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_utils import plot, load_dataset
from losses import l2_loss, area_loss, Class_net
from measure import f_mse, f_ssim, f_psnr, f_mae, f_lpips
import layers_3d as model
import layers_D

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
# optimizer setting
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
# experiment setting
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--niter', type=int, default=120, help='number of epochs to train for')
parser.add_argument('--epoch_size', type=int, default=500, help='epoch size') #500
parser.add_argument('--reload', type=int, default=0, help='reload model')
parser.add_argument('--just_sample', type=bool, default=False, help='reload model')
parser.add_argument('--seed', default=1234, type=int, help='manual seed')
# dress setting
parser.add_argument('--log_dir', default='logs', help='base directory to save logs')
parser.add_argument('--data_root', default='/home/zangchuanqi/dataset', help='root directory for data')
# dataset and auxiliary setting
parser.add_argument('--dataset', default='human', help='dataset to train with')
parser.add_argument('--image_width', type=int, default=(128, 128), help='the height / width of the input image to network')
parser.add_argument('--channel', type=int, default=3, help='the channel of the input image to network')
parser.add_argument('--n_past', type=int, default=8, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
parser.add_argument('--n_eval', type=int, default=20, help='number of frames to predict at eval time')
parser.add_argument('--predictor_rnn_layers', type=int, default=6, help='number of layers')
parser.add_argument('--rnn_size', type=int, default=32, help='dimensionality of hidden layer')
parser.add_argument('--dir_sign', default='default', help='dataset to train with')

opt = parser.parse_args()

if opt.dataset == "ucf":
    opt.image_width=(120,160)
    # opt.image_width=(64,88)
    opt.n_past=2
    opt.n_future=10
    opt.n_eval=10
    opt.batch_size=8
    opt.predictor_rnn_layers = 6
if opt.dataset == "human":
    opt.image_width=(128, 128)
    opt.n_past=2
    opt.n_future=4
    opt.n_eval=4
    opt.batch_size=4
    opt.predictor_rnn_layers = 6
print(opt)

# ----------------create dir for model-------------------
name = 'layers=%s_seq_len=%s_image_size=%s_dir_sign=%s' % (
opt.predictor_rnn_layers, opt.n_future + opt.n_past, opt.image_width, opt.dir_sign)
opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)
os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)

#---------tensorboard----------------------
# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

#---------random seed----------------------
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

# -------- optimizers ----------------
channels = opt.channel
encoder         = model.autoencoder(nBlocks=[4,5,3], nStrides=[1, 2, 2], nChannels=None, init_ds=2, dropout_rate=0.,
                            affineBN=True, in_shape=[channels, opt.image_width[0], opt.image_width[1]], mult=2)
frame_predictor = model.zig_rev_predictor(opt.rnn_size,  opt.rnn_size, opt.rnn_size, opt.predictor_rnn_layers,
                                          opt.batch_size, n_past=opt.n_past, channels= channels, dataset=opt.dataset)
discriminator   = layers_D.define_D(3*(opt.n_past + opt.n_future-1), 64, 3, 2)
Class_Loss = Class_net(use_lsgan=True)

frame_predictor_optimizer = optim.Adam(frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_optimizer = optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_D_T = torch.optim.Adam(discriminator.parameters(), lr=0.00005, betas=(opt.beta1, 0.999))

scheduler1 = torch.optim.lr_scheduler.StepLR(frame_predictor_optimizer, step_size=50, gamma=0.2)
scheduler2 = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=50, gamma=0.2)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer_D_T, step_size=50, gamma=0.2)

# --------- transfer to gpu ------------------------------------
encoder.cuda()
frame_predictor.cuda()
discriminator.cuda()

# --------- load a dataset ------------------------------------
train_data, test_data = load_dataset(opt)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True,
                                           num_workers=8, drop_last=True, pin_memory=False)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False,
                                          num_workers=8, drop_last=True, pin_memory=False)


# --------- training funtions ------------------------------------
def train(x, epoch):
    frame_predictor.zero_grad()
    encoder.zero_grad()
    discriminator.zero_grad()
    optimizer_D_T.zero_grad()

    mse_coare, mse_fine = 0., 0.
    hidden_set, real_seq, fake_seq = [], [], []
    x_generation = torch.zeros_like(x[0])

    #initial memo state in PredRNN
    memo = torch.zeros(opt.batch_size, opt.rnn_size*channels, 3, int(opt.image_width[0]/8), int(opt.image_width[1]/8)).cuda()
    context = torch.zeros(opt.batch_size, opt.rnn_size*channels*2, 3, int(opt.image_width[0]/8), int(opt.image_width[1]/8)).cuda()

    for i in range(1, opt.n_past + opt.n_future):
        if i <= opt.n_past:
            input = x[i - 1]
        else:
            # teaching schedule
            teaching_rate = 0.9 * np.power(0.9, epoch // 5)
            if np.random.uniform()<teaching_rate: input = x[i - 1]
            else: input = x_generation
        # encoder
        h = encoder(input, True)
        h_encode = h
        # global memory
        if i < opt.n_past:
            hidden_set.append(torch.cat((h[0], h[1]), dim=1))
        if i == opt.n_past:
            hidden_set.append(torch.cat((h[0], h[1]), dim=1))
            hidden_set = torch.stack(hidden_set, 1)
            b0, seq, h_size, d0, w0, h0 = hidden_set.shape
            context = hidden_set.reshape(b0, seq*h_size, d0, w0, h0)
        # initial hidden
        if i == 1: frame_predictor.hidden = frame_predictor.init_hidden(torch.cat((h[0], h[1]), dim=1))
        # iteratively prediction
        for j in range(opt.predictor_rnn_layers):
            h_pred, memo = frame_predictor(h, memo, j, i, context, h_encode)
            if j%2 == 0: # related to encoder structure
                x_pred = encoder((h_pred[1], h_pred[0]), False)
            else:
                x_pred = encoder(h_pred, False)
            x_generation, h = x_pred, h_pred

            if j == 0:
                real_seq.append(x[i][:, :, 2])
                fake_seq.append(x_pred[:, :, 2])
                mse_coare_ = area_loss(x_pred[:,:,0], x[i][:,:,0], x[i-1][:,:,0])
                mse_coare_ += area_loss(x_pred[:,:,1], x[i][:,:,1], x[i-1][:,:,1])
                mse_coare_ += area_loss(x_pred[:,:,2], x[i][:,:,2], x[i-1][:,:,2])
                mse_coare = mse_coare + mse_coare_*1e-4
            if j >= 0:
                mse_fine_ = l2_loss(x_pred[:,:,0], x[i][:,:,0])
                mse_fine_ += l2_loss(x_pred[:,:,1], x[i][:,:,1])
                mse_fine_ += l2_loss(x_pred[:,:,2], x[i][:,:,2])
                mse_fine = mse_fine + mse_fine_*(0.8**(opt.predictor_rnn_layers-j-1))
    mse = mse_coare*1e-4 + mse_fine

    #shuffle真实图像
    shuffle_fake_seq = real_seq
    copy_fake_seq = real_seq
    random.shuffle(shuffle_fake_seq)
    idx = random.randint(0, len(real_seq) - 1)
    for j in range(1, len(real_seq)-idx):
        copy_fake_seq[idx+j] = copy_fake_seq[idx]
    real_seq = torch.stack(real_seq, dim=1)
    copy_fake_seq = torch.stack(copy_fake_seq, dim=1)
    shuffle_fake_seq = torch.stack(shuffle_fake_seq, dim=1)
    fake_seq = torch.stack(fake_seq, dim=1)

    #获取Discriminator的loss和GAN Loss
    real_seq = real_seq.view(opt.batch_size, -1, opt.image_width[0], opt.image_width[1])
    fake_seq = fake_seq.view(opt.batch_size, -1, opt.image_width[0], opt.image_width[1])
    copy_fake_seq = copy_fake_seq.view(opt.batch_size, -1, opt.image_width[0], opt.image_width[1])
    shuffle_fake_seq = shuffle_fake_seq.view(opt.batch_size, -1, opt.image_width[0], opt.image_width[1])

    pred_real = discriminator.forward(real_seq)
    copy_fake_seq = discriminator.forward(copy_fake_seq)
    shuffle_fake_seq = discriminator.forward(shuffle_fake_seq)
    loss_D_real = Class_Loss(pred_real, True)
    loss_D_copy_fake = Class_Loss(copy_fake_seq, False)
    loss_D_shuffle_fake = Class_Loss(shuffle_fake_seq, False)
    loss_D = loss_D_shuffle_fake + loss_D_copy_fake + loss_D_real #* 0.5

    pred_fake = discriminator.forward(fake_seq)
    loss_ts = Class_Loss(pred_fake, True)

    #backpropagation
    loss = mse + loss_ts*0.001
    loss.backward()
    frame_predictor_optimizer.step()
    encoder_optimizer.step()
    loss_D.backward()
    optimizer_D_T.step()

    return mse.data.cpu().numpy() / (opt.n_past + opt.n_future), loss_ts.data.cpu().numpy(), \
           loss_D_shuffle_fake.data.cpu().numpy(), loss_D_copy_fake.data.cpu().numpy(), loss_D_real.data.cpu().numpy()


# --------- training loop ------------------------------------
def trainIters():
    print_para_nums()
    for epoch in range(opt.niter):
        t0 = time.time()
        encoder.train()
        frame_predictor.train()
        discriminator.train()
        epoch_mse, epoch_D_real, epoch_D_fake, epoch_D_s_fake, epoch_D_c_fake = 0,0,0,0,0

        if opt.just_sample:
            evaluation(opt.reload, epoch_mse)
            break
        for i, x in enumerate(train_loader, 0):
            x = x.permute(1,0,4,2,3).float().cuda()
            input = []
            #[c, w, h] -> [c,3,w,h]
            for j in range(opt.n_future+opt.n_past):
                k1 = x[j][:,:,None,:,:]
                k2 = x[j + 1][:,:,None,:,:]
                k3 = x[j + 2][:,:,None,:,:]
                input.append(torch.cat((k1,k2,k3),2))

            mse, D_fake, D_s_fake, D_c_fake, D_real = train(input, epoch)
            epoch_mse += mse
            epoch_D_fake += D_fake
            epoch_D_c_fake += D_c_fake
            epoch_D_s_fake += D_s_fake
            epoch_D_real += D_real
        scheduler1.step()
        scheduler2.step()
        scheduler3.step()

        length = len(train_loader)
        if epoch > 0 and epoch % 1 == 0:
            print('[%02d] train mse loss: %.5f, Discriminator fake(generate, shuffle, copy) loss: %.5f, %.5f,%.5f, '
                  'Discriminator real loss: %.5f, train time: %.1f' %
                  (epoch, epoch_mse/length, epoch_D_fake/length, epoch_D_s_fake/length, epoch_D_c_fake/length,
                   epoch_D_real/length, time.time()-t0))
        if epoch > 0 and epoch % 5 == 0:
            evaluation(epoch, epoch_mse)
            torch.save({ 'encoder': encoder, 'frame_predictor': frame_predictor, 'discriminator': discriminator, 'opt': opt},
                       '%s/model_%s.pth' % (opt.log_dir,epoch))


def evaluation(epoch, epoch_mse):
    t0 = time.time()
    with torch.no_grad():
        target_t, pred_t= [], []
        frame_predictor.eval()
        encoder.eval()
        discriminator.eval()
        img_mse, ssim, psnr, lp, mae= [], [], [], [], []
        for i in range(opt.n_eval):
            img_mse.append(0)
            ssim.append(0)
            psnr.append(0)
            lp.append(0)
            mae.append(0)

        for test_i, x in enumerate(test_loader):
            x = x.permute(1, 0, 4, 2, 3).float().cuda()
            input = []
            # [c, w, h] -> [c,3,w,h]
            for j in range(opt.n_eval+opt.n_past):
                k1 = x[j][:, :, None, :, :]
                k2 = x[j + 1][:, :, None, :, :]
                k3 = x[j + 2][:, :, None, :, :]
                input.append(torch.cat((k1, k2, k3), 2))

            gen_seq = []
            gt_seq = [input[i] for i in range(len(input))]
            memo = torch.zeros(opt.batch_size, opt.rnn_size*channels, 3, int(opt.image_width[0]/8), int(opt.image_width[1]/8)).cuda()
            context = torch.zeros(opt.batch_size, opt.rnn_size*channels*2, 3, int(opt.image_width[0] / 8), int(opt.image_width[1] / 8)).cuda()

            x_in = input[0]
            hidden_set=[]
            for i in range(1, opt.n_eval+opt.n_past):
                h = encoder(x_in)
                h_encode = h
                if i == 1: frame_predictor.hidden = frame_predictor.init_hidden(torch.cat((h[0], h[1]), dim=1))
                if i < opt.n_past:
                    hidden_set.append(torch.cat((h[0], h[1]), dim=1))
                    for j in range(opt.predictor_rnn_layers):
                        h_pred, memo = frame_predictor(h, memo, j, i, context)
                        h = h_pred
                        if j == opt.predictor_rnn_layers - 1:
                            x_in = input[i]
                elif i == opt.n_past:
                    hidden_set.append(torch.cat((h[0], h[1]), dim=1))
                    hidden_set = torch.stack(hidden_set, 1)
                    b0, seq, h_size, d0, w0, h0 = hidden_set.shape
                    context = hidden_set.reshape(b0, seq * h_size, d0, w0, h0)
                    for j in range(opt.predictor_rnn_layers):
                        h_pred, memo = frame_predictor(h, memo, j, i, context)
                        h = h_pred
                        if j == opt.predictor_rnn_layers-1:
                            x_in = encoder(h_pred, False).detach()
                            x_in[:, :, 0] = input[i][:, :, 0]
                            x_in[:, :, 1] = input[i][:, :, 1]
                            gen_seq.append(x_in)

                elif i == opt.n_past + 1:
                    for j in range(opt.predictor_rnn_layers):
                        h_pred, memo = frame_predictor(h, memo, j, i, context,h_encode)
                        h = h_pred
                        if j == opt.predictor_rnn_layers-1:
                            x_in = encoder(h_pred, False).detach()
                            x_in[:, :, 0] = input[i][:, :, 0]
                            gen_seq.append(x_in)
                else:
                    for j in range(opt.predictor_rnn_layers):
                        h_pred, memo = frame_predictor(h, memo, j, i, context,h_encode)
                        h = h_pred
                        if j == opt.predictor_rnn_layers - 1:
                            x_in = encoder(h_pred, False).detach()
                            gen_seq.append(x_in)

            target_tensor = torch.stack(gt_seq, dim=1)[:,:,:,2,:,:]
            predictions_tensor = torch.stack(gen_seq, 1)[:, :, :, 2, :, :]
            predictions = predictions_tensor.float().cpu().numpy()
            target = target_tensor.float().cpu().numpy()

            eval_predictions = predictions[:,-opt.n_eval:]
            eval_target = target[:,-opt.n_eval:]
            img_mse = f_mse(eval_target, eval_predictions, img_mse)
            ssim = f_ssim(eval_target, eval_predictions, ssim)
            psnr = f_psnr(eval_target, eval_predictions, psnr)
            mae = f_mae(eval_target, eval_predictions, mae)

            # magnify batch size
            observed_tensor = input[0][:,:,0:2].permute(0,2,1,3,4)
            total_gt = torch.cat((observed_tensor, target_tensor), 1)
            total_pr = torch.cat((total_gt[:,:opt.n_past+2], predictions_tensor), 1)
            if test_i<30:
                target_t.append(total_gt)
                pred_t.append(total_pr)

            if test_i == 2:
                s0,s1,s2,s3,s4 = target_t[0].shape
                target_plot = torch.stack(target_t, dim=0).reshape(3*s0,s1,s2,s3,s4)
                predict_plot = torch.stack(pred_t, dim=0).reshape(3*s0,s1,s2,s3,s4)
                fname = '%s/gen/sample_%d.png' % (opt.log_dir, epoch)
                plot(target_plot, predict_plot, fname)
                #residual map
                ress = torch.sum(abs(target_plot-predict_plot), dim=2)
                fname = '%s/gen/sample_%d_res.png' % (opt.log_dir, epoch)
                plot(ress.unsqueeze(2), ress.unsqueeze(2), fname)

        target_t = torch.stack(target_t, dim=1)
        predictions = torch.stack(pred_t, dim=1)
        lp = f_lpips(target_t[:,:,-opt.n_eval:], predictions[:,:,-opt.n_eval:], lp)

        length = len(test_loader)
        img_mse=np.array(img_mse)/length
        ssim=np.array(ssim)/length
        psnr=np.array(psnr)/length
        lp=np.array(lp)*100
        mae=np.array(mae)/length
        writer.add_scalar('Loss/mse', epoch_mse / len(train_loader), epoch)
        writer.add_scalar('Accuracy/mse', img_mse.mean(), epoch)
        writer.add_scalar('Accuracy/ssim', ssim.mean(), epoch)
        writer.add_scalar('Ture iteration', epoch * len(train_loader) * opt.batch_size, epoch)
        writer.close()

        print('[%03d] test time: %d, length: %d, mse, ssim, psnr, fvd, lpips, mae: %.3f, %.3f, %.3f, %.3f, %.5f, %.3f' %
              (epoch, time.time()-t0, length, img_mse.mean(), ssim.mean(), psnr.mean(), 0.0, lp.mean(), mae.mean()))
        print('mse', end='')
        print(img_mse)
        print('ssim', end='')
        print(ssim)
        print('psnr', end='')
        print(psnr)
        print('lpips', end='')
        print(lp)
        print('mae', end='')
        print(mae)

if opt.reload > 0:
    file_name = '%s/model_%s.pth' % (opt.log_dir, opt.reload)
    print("reload model from %s/model_%s.pth" % (opt.log_dir, opt.reload))
    checkpoint = torch.load(file_name)
    encoder = checkpoint['encoder']
    frame_predictor = checkpoint['frame_predictor']
    discriminator = checkpoint['discriminator']
    print("success reload")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_para_nums():
    print('frame_predictor, encoder parameters: ', count_parameters(frame_predictor), count_parameters(encoder))
    print('discriminator parameters: ', count_parameters(discriminator))

trainIters()
