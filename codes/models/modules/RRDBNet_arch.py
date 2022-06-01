import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
# from models.networks import ResidualBlock


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class RB(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2):
        super(RB, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=True))

        self.conv = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        c0 = self.conv(x)
        x = self.block(x)
        return x + c0

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.branch1 = nn.Sequential(RB(in_nc, nf, 0.2), RB(nf, nf, 0.2), RB(nf, nf, 0.2))
        self.branch2 = nn.Sequential(RB(in_nc, nf, 0.2), RB(nf, nf, 0.2), RB(nf, nf, 0.2))
        # self.sm = nn.Sequential(RB(in_nc, nf, 0.2), RB(nf, nf, 0.2), RB(nf, nf, 0.2), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Softmax(dim=1))
        self.sm = nn.Sequential(RB(in_nc, nf, 0.2), RB(nf, nf, 0.2), RB(nf, nf, 0.2), nn.Softmax(dim=1))
        
        self.tail_conv1 = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.tail_conv2 = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        mid_out = self.conv_last(self.lrelu(self.HRconv(fea)))

        mask = self.sm(mid_out)

        o1 = mask * self.branch1(mid_out)
        # out1 = self.tail_conv1(self.lrelu(o1))
        out1 = self.tail_conv1(o1)
        
        o2 = (1-mask) * self.branch2(mid_out)
        # out2 = self.tail_conv2(self.lrelu(o2))
        out2 = self.tail_conv2(o2)
        
        out = out1 + out2 

        return out, out1, out2

# a = torch.randn(16,3,128,128)
# Network = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
# x,y,z = Network(a)
# print(y.size())
# print(x.size())
# print(z.size())
