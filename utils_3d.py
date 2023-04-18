import torch
import torch.nn as nn

def split(x):
    n = int(x.size()[1]/2)
    x1 = x[:, :n, :, :, :].contiguous()
    x2 = x[:, n:, :, :, :].contiguous()
    return x1, x2


def merge(x1, x2):
    return torch.cat((x1, x2), 1)


class psi(nn.Module):
    def __init__(self, block_size):
        super(psi, self).__init__()
        self.block_size = block_size #2
        self.block_size_sq = block_size*block_size #2*2

    def inverse(self, input):

        output = input.permute(0, 2, 3, 4, 1)
        (batch_size, temp, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / 4)
        s_width = int(d_width * 2)
        s_height = int(d_height * 2)
        t_1 = output.contiguous().view(batch_size, temp, d_height, d_width, 4, s_depth)
        spl = t_1.split(2, 4)
        stack = [t_t.contiguous().view(batch_size, temp, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).transpose(1, 2).permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, temp, s_height, s_width, s_depth)
        output = output.permute(0, 4, 1, 2, 3)
        return output.contiguous()

    def forward(self, input):
        # 本来是：h是隔一行给到channel，w是中间切分给channel
        # 现在改成都是隔一行/列给到channel
        # [b,c,3,w,h]
        output = input.permute(0, 2, 3, 4, 1)
        # [b,3,w,h,c]
        (batch_size, temp, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size
        t_1 = output.split(self.block_size, 3)
        #t_1 [b,3,w,2,c], [b,3,w,2,c], [b,3,w,2,c] ... total:h/2
        stack = [t_t.contiguous().view(batch_size, temp, s_height, d_depth) for t_t in t_1]
        # stack [b,3,w,2*c] [b,3,w,2*c] [b,3,w,2*c]
        output = torch.stack(stack, 2)
        #output [b,3,h/2,w,2*c]
        d_width = int(s_width / self.block_size)
        t_1 = output.split(self.block_size, 3)
        d_depth = s_depth * self.block_size_sq
        stack = [t_t.contiguous().view(batch_size,temp, d_width, d_depth) for t_t in t_1]
        # stack [b,3, w/2, c*4] [b,3, w/2, c*4] [b,3, w/2, c*4] total:h/2

        output = torch.stack(stack, 3)
        # output [b,3,h/2, w/2, c*4]
        output = output.permute(0, 4, 1, 3, 2)
        # output [b,c*4,3,w/2,h/2]

        return output.contiguous()



class wavelet(nn.Module):
    def __init__(self, block_size):
        super(wavelet, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size



    def inverse(self, input):
        output = input.permute(0, 2, 3, 4, 1)
        (batch_size, temp, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / 4)
        s_width = int(d_width * 2)
        s_height = int(d_height * 2)
        t_1 = output.contiguous().view(batch_size, temp, d_height, d_width, 4, s_depth)
        spl = t_1.split(2, 4)
        stack = [t_t.contiguous().view(batch_size, temp, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).transpose(1, 2).permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, temp, s_height, s_width, s_depth)
        output = output.permute(0, 4, 1, 2, 3)
        return output.contiguous()

    def forward(self, input):
        output = input.permute(0, 2, 3, 4, 1)
        (batch_size, temp, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_height = int(s_height / self.block_size)
        print("utils_3d 77| output size: ", output.size())
        t_1 = output.split(self.block_size, 3)
        print("utils_3d 77| t_1 size: ", t_1.size())
        stack = [t_t.contiguous().view(batch_size,temp, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 2)
        output = output.permute(0, 4, 1, 3, 2)
        return output.contiguous()

