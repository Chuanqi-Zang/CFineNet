import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_3d import split, merge, psi


class irevnet_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, first=False, dropout_rate=0.,
                 affineBN=True, mult=2):
        """ buid invertible bottleneck block """
        super(irevnet_block, self).__init__()
        self.first = first
        self.stride = stride
        self.psi = psi(stride)
        layers = []
        if not first:
            layers.append(nn.BatchNorm3d(in_ch//2, affine=affineBN))
            layers.append(nn.ReLU(inplace=True))
        if int(out_ch//mult)==0:
            ch = 1
        else:
            ch =int(out_ch//mult)
        if self.stride ==2:
            layers.append(nn.Conv3d(in_ch // 2, ch, kernel_size=3,
                                    stride=(1,2,2), padding=1, bias=False))
        else:
            layers.append(nn.Conv3d(in_ch // 2, ch, kernel_size=3,
                                    stride=self.stride, padding=1, bias=False))
        layers.append(nn.BatchNorm3d(ch, affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv3d(ch, ch, kernel_size=3, padding=1, bias=False))
        layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.BatchNorm3d(ch, affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv3d(ch, out_ch, kernel_size=3, padding=1, bias=False))
        self.bottleneck_block = nn.Sequential(*layers)

    def forward(self, x):
        """ bijective or injective block forward """
        x1 = x[0]
        x2 = x[1]
        Fx2 = self.bottleneck_block(x2)
        if self.stride == 2:
            x1 = self.psi.forward(x1)
            x2 = self.psi.forward(x2)
        y1 = Fx2 + x1
        return (x2, y1)

    def inverse(self, x):
        """ bijective or injecitve block inverse """
        x2, y1 = x[0], x[1]
        if self.stride == 2:
            x2 = self.psi.inverse(x2)
        Fx2 = - self.bottleneck_block(x2)
        x1 = Fx2 + y1
        if self.stride == 2:
            x1 = self.psi.inverse(x1)
        x = (x1, x2)
        return x

class autoencoder(nn.Module):
    def __init__(self, nBlocks, nStrides, nChannels=None, init_ds=2,
                 dropout_rate=0., affineBN=True, in_shape=None, mult=2):
        super(autoencoder, self).__init__()
        self.ds = in_shape[2]//2**(nStrides.count(2)+init_ds//2)
        self.init_ds = init_ds
        self.in_ch = in_shape[0] * 2**self.init_ds
        self.nBlocks = nBlocks
        self.first = True
        if not nChannels:
            nChannels = [self.in_ch//2, self.in_ch//2 * 4,
                         self.in_ch//2 * 4**2, self.in_ch//2 * 4**3]

        self.init_psi = psi(self.init_ds)
        self.stack = self.irevnet_stack(irevnet_block, nChannels, nBlocks,
                                        nStrides, dropout_rate=dropout_rate,
                                        affineBN=affineBN, in_ch=self.in_ch,
                                        mult=mult)

    def irevnet_stack(self, _block, nChannels, nBlocks, nStrides, dropout_rate,
                      affineBN, in_ch, mult):
        """ Create stack of irevnet blocks """
        block_list = nn.ModuleList()
        strides = []
        channels = []
        for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
            strides = strides + ([stride] + [1]*(depth-1))
            channels = channels + ([channel]*depth)

        for channel, stride in zip(channels, strides):
            block_list.append(_block(in_ch, channel, stride,
                                     first=self.first,
                                     dropout_rate=dropout_rate,
                                     affineBN=affineBN, mult=mult))
            in_ch = 2 * channel
            self.first = False
        return block_list

    def forward(self, input, is_predict = True):
        if is_predict:
            n = self.in_ch // 2
            if self.init_ds != 0:
                x = self.init_psi.forward(input)
            out = (x[:, :n, :, :, :], x[:, n:, :, :, :])
            for block in self.stack:

                out = block.forward(out)
            x = out
        else:
            out = input
            for i in range(len(self.stack)):
                out = self.stack[-1 - i].inverse(out)
            out = merge(out[0], out[1])
            x = self.init_psi.inverse(out)
        return x


class STConvLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, memo_size, c=3):
        super(STConvLSTMCell,self).__init__()
        self.KERNEL_SIZE = 3
        self.PADDING = self.KERNEL_SIZE // 2
        self.input_size = input_size*c
        self.hidden_size = hidden_size*c
        self.memo_size = memo_size*c
        self.in_gate = nn.Conv3d(self.input_size + self.hidden_size + self.hidden_size, self.hidden_size, self.KERNEL_SIZE, padding=self.PADDING)
        self.remember_gate = nn.Conv3d(self.input_size + self.hidden_size + self.hidden_size, self.hidden_size, self.KERNEL_SIZE, padding=self.PADDING)
        self.cell_gate = nn.Conv3d(self.input_size + self.hidden_size + self.hidden_size, self.hidden_size , self.KERNEL_SIZE, padding=self.PADDING)

        self.in_gate1 = nn.Conv3d(self.input_size + self.memo_size + self.hidden_size, self.memo_size, self.KERNEL_SIZE, padding=self.PADDING)
        self.remember_gate1 = nn.Conv3d(self.input_size + self.memo_size + self.hidden_size, self.memo_size, self.KERNEL_SIZE, padding=self.PADDING)
        self.cell_gate1 = nn.Conv3d(self.input_size + self.memo_size + self.hidden_size, self.memo_size, self.KERNEL_SIZE, padding=self.PADDING)

        self.w1 = nn.Conv3d(self.hidden_size + self.memo_size, self.hidden_size, 1)
        self.out_gate = nn.Conv3d(self.input_size + self.hidden_size +self.hidden_size+self.memo_size, self.hidden_size, self.KERNEL_SIZE, padding=self.PADDING)

        self.w2 = nn.Conv3d(self.hidden_size + self.memo_size, self.hidden_size, 1)
        self.out_gate2 = nn.Conv3d(self.input_size + self.hidden_size +self.hidden_size+self.memo_size, self.hidden_size, self.KERNEL_SIZE, padding=self.PADDING)

    def forward(self, input, prev_state):
        input_,prev_memo = input
        prev_hidden, prev_cell, prev_mask = prev_state
        # data size [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden, prev_cell), 1)
        in_gate = F.sigmoid(self.in_gate(stacked_inputs))
        remember_gate = F.sigmoid(self.remember_gate(stacked_inputs))
        cell_gate = F.tanh(self.cell_gate(stacked_inputs))

        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)

        stacked_inputs1 = torch.cat((input_, prev_memo, cell), 1)

        in_gate1 = F.sigmoid(self.in_gate1(stacked_inputs1))
        remember_gate1 = F.sigmoid(self.remember_gate1(stacked_inputs1))
        cell_gate1 = F.tanh(self.cell_gate1(stacked_inputs1))

        memo = (remember_gate1 * prev_memo) + (in_gate1 * cell_gate1)

        out_gate = F.sigmoid(self.out_gate(torch.cat((input_, prev_hidden, cell, memo), 1)))
        hidden = out_gate * F.tanh(self.w1(torch.cat((cell, memo), 1)))
        out_gate2 = F.sigmoid(self.out_gate2(torch.cat((input_, prev_mask, cell, memo), 1)))
        Mask = out_gate2 * F.tanh(self.w2(torch.cat((cell, memo), 1)))
        Mask = F.sigmoid(Mask)
        return (hidden, cell, Mask), memo

#pixel correlation attention
class attention(nn.Module):
    def __init__(self, rnn_size, c=1, hs=1):
        super(attention, self).__init__()
        self.rnn_size = rnn_size
        self.convx1 = nn.Conv3d(rnn_size, 96*2, 1, padding=0)
        self.convx2 = nn.Conv3d(96*2, 96, 1, padding=0)
        self.convx3 = nn.Conv3d(96*3, 96*3, 1, padding=0)
        self.convx4 = nn.Conv3d(96*3, 96*3, 1, padding=0)
        self.convx5 = nn.Conv3d(96, 96, 1, padding=0)
        self.convx6 = nn.Conv3d(96*3, 96, 1, padding=0)
        self.softmax = nn.Softmax(dim=2)
    def forward(self, context, h_original, x):
        h_original = torch.cat([h_original[0], h_original[1]], dim=1)
        context = self.convx1(context)
        h_original = self.convx2(h_original)
        k = self.convx3(torch.cat([context, h_original], dim=1))
        v = self.convx4(torch.cat([context, h_original], dim=1))
        q = self.convx5(x)
        B,C,D,H,W = q.shape
        q = q.permute(0,2,3,4,1).reshape(B, D*H*W, C).repeat(1,1,3)
        k = k.reshape(B, C*3, D*H*W)
        correlation = self.softmax(torch.bmm(q,k))
        v = v.permute(0,2,3,4,1).reshape(B, D*H*W, C*3)
        result = torch.bmm(correlation, v)
        result = result.reshape(B, D, H, W, C*3).permute(0,4,1,2,3)
        result = self.convx6(result)+x
        return result


class Double_code(nn.Module):
    def __init__(self, rnn_size, c=1, hs=1):
        super(Double_code, self).__init__()
        self.rnn_size = rnn_size
        self.convx1 = nn.Conv3d(rnn_size, 96, 1, padding=0)
        self.convx2 = nn.Conv3d(96, 96, 3, padding=1)
        self.convx3 = nn.Conv3d(96*2, 96, 1, padding=0)
        self.convx4 = nn.Conv3d(96, 96, 3, padding=1)
        self.conv1 = nn.Conv3d(96*2+hs*c, hs*c, 1, padding=0)
        self.conv2 = nn.Conv3d(hs*c, hs*c, 3, padding=1)

    def forward(self, context, h_original, x):
        context = F.relu(self.convx1(context))
        context = F.relu(self.convx2(context))
        h_original = torch.cat([h_original[0], h_original[1]], dim=1)
        h_original = F.relu(self.convx3(h_original))
        h_original = F.relu(self.convx4(h_original))
        reference = torch.cat([context, h_original], dim=1)

        cor_context = torch.cat([reference, x], dim=1)
        output = F.relu(self.conv1(cor_context))
        output = F.relu(self.conv2(output))
        return output


class zig_rev_predictor(nn.Module):
    def __init__(self, input_size, hidden_size,output_size,n_layers,batch_size,temp =3, n_past=2, channels=3, dataset='human'):
        super(zig_rev_predictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.temp = temp
        self.n_past = n_past
        self.channels=channels
        self.con_vec = 2 if n_past<3 else 3

        self.in_convlstm = STConvLSTMCell(input_size, hidden_size, hidden_size, c=channels)
        self.cly_convlstm = STConvLSTMCell(hidden_size, hidden_size, hidden_size, c=channels)
        self.cly2_convlstm = STConvLSTMCell(hidden_size, hidden_size, hidden_size, c=channels)

        self.att = nn.ModuleList([nn.Sequential(nn.Conv3d(self.hidden_size*channels, self.hidden_size*channels, 1, 1, 0),
                                                nn.ReLU(),
                                                nn.Conv3d(self.hidden_size*channels, self.hidden_size*channels, 3, 1, 1),
                                                nn.Sigmoid()
                                                ) for i in range(2)])
        self.w = nn.Conv3d(channels*8*8, self.hidden_size*3*channels, 1)
        self.code = Double_code(self.hidden_size*channels*2*self.con_vec, c=channels, hs=hidden_size)

    def init_hidden(self, h):
        self.hidden = []
        init_whole = self.w(h)
        init_h = init_whole[:, :self.hidden_size * self.channels, :, :]
        init_c = init_whole[:, self.hidden_size * self.channels:self.hidden_size * self.channels*2, :, :]
        init_m = init_whole[:, self.hidden_size * self.channels*2:, :, :]
        for i in range(3):
            self.hidden.append((init_h, init_c, init_m))
        return self.hidden

    def forward(self, input_, memo, pred_layer=0, pred_time=0, context=None, h_original=None):

        x1, x2 = input_
        x1_ = x1
        if pred_layer==0:
            out = self.in_convlstm((x1_, memo), self.hidden[0])
            self.hidden[0] = out[0]
            memo = out[1]
            g = self.att[0](out[0][0])
            x2 = (1 - g) * x2 + g * out[0][0]
            x1, x2 = x2, x1

        else:
            if pred_time <= self.n_past:
                prev_hidden, prev_cell, prev_mask = self.hidden[1]
                out = self.cly_convlstm((x1_, memo), self.hidden[1])
                self.hidden[1] = out[0]
                memo = out[1]
                hidden_state = self.hidden[1][0]
                Mask = self.hidden[1][2]
                hidden_state = hidden_state * Mask + prev_hidden * (1 - Mask)
                g = self.att[1](hidden_state)
                x2 = (1 - g) * x2 + g * hidden_state
                x1, x2 = x2, x1
            else:
                if pred_time == self.n_past+1 and pred_layer==1:
                    self.hidden[2] = self.hidden[1]
                prev_hidden, prev_cell, prev_mask = self.hidden[2]
                prev_cell = self.code(context, h_original, prev_cell)
                self.hidden[2] = (prev_hidden, prev_cell, prev_mask)
                out = self.cly2_convlstm((x1_, memo), self.hidden[2])
                self.hidden[2] = out[0]
                memo = out[1]
                hidden_state = self.hidden[2][0]
                Mask = self.hidden[2][2]
                hidden_state = hidden_state*Mask + prev_hidden*(1-Mask)
                g = self.att[1](hidden_state)
                x2 = (1 - g) * x2 + g * hidden_state
                x1, x2 = x2, x1

        return (x1,x2), memo