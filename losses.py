import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
import numpy as np


def l1_loss(image_true, image_output):
    loss1 = nn.L1Loss()(image_true, image_output)
    return loss1


def l2_loss(image_true, image_output):
    loss = nn.MSELoss(reduction='none')(image_true, image_output)
    loss = loss.mean()
    return loss


def smooth_l1_loss(image_true, image_output):
    loss = nn.SmoothL1Loss()(image_true, image_output)
    return loss


def cosine_distance(image_true, image_output):
    eps = 1e-10
    image_true_factor = torch.norm(image_true, dim=-1, keepdim=True)
    image_output_factor = torch.norm(image_output, dim=-1, keepdim=True)
    image_true_norm = image_true / (image_true_factor + eps)
    image_output_norm = image_output / (image_output_factor + eps)
    loss = nn.CosineSimilarity(image_true_norm, image_output_norm)
    return loss


def edge_loss(image_true, image_output):
    loss = nn.MSELoss()(laplacian(image_output).cuda(), laplacian(image_true).cuda())
    return loss


def area_loss(target_image, generate_image, pre_image):

    target_res = torch.abs(target_image-pre_image)
    generate_res = torch.abs(generate_image-pre_image)
    # 0.05 0.03 0.1
    vk = torch.ones_like(pre_image)*0.05
    mask_k = torch.lt(target_res, vk)
    target_area = target_res.masked_fill(mask_k, 0.0)  # -1e18
    mask_k = torch.ge(target_res, vk)
    target_area = target_area.masked_fill(mask_k, 1.0)  # -1e18

    mask_k = torch.lt(generate_res, vk)
    generate_area = generate_res.masked_fill(mask_k, 0.0)  # -1e18
    mask_k = torch.ge(generate_res, vk)
    generate_area = generate_area.masked_fill(mask_k, 1.0)  # -1e18

    loss = l2_loss(target_area, generate_area)
    return torch.sum(loss)


def Cross_Eentropy_loss(image_true, image_output):
    reverse = torch.ones_like(image_output)-image_output
    com = torch.cat((reverse, image_output),dim=1)
    # image_true = torch.squeeze(image_true)
    bs, c, w, h = image_true.size()
    com = com.permute(0, 2, 3, 1).reshape(bs*w*h, c*2)
    image_true = image_true.permute(0, 2, 3, 1).reshape(bs*w*h)

    # CELoss = nn.NLLLoss()
    # com = torch.log(com)
    # image_true = torch.log(image_true)
    CELoss = nn.CrossEntropyLoss()
    loss = CELoss(com, image_true.type(torch.long))
    return loss


def laplacian(x, device):
    weight = torch.tensor([
        [[[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
         [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]],
        [[[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], [[8., 0., 0.], [0., 8., 0.], [0., 0., 8.]],
         [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]],
        [[[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
         [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]]
    ]).to(device)
    frame = torch.nn.functional.conv2d(x, weight, stride=1, padding=1)
    return frame


class Class_net(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(Class_net, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        gpu_id = input.get_device()
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).cuda(gpu_id).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).cuda(gpu_id).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 4):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 12):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 16):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


def loss_sum(target_image, generate_image):
    loss = {}
    loss['l2_loss'] = l2_loss(target_image, generate_image)
    loss_values = [val.mean() for val in loss.values()]
    loss = sum(loss_values)
    return loss