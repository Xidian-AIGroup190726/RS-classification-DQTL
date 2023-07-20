import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.utils import save_image
from model.generator import Generator


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 kernel
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def tranconv3x3(in_planes, out_planes, kernel_size=3, stride=1):
    # 3x3 kernel
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=1, bias=False)


class BasicBlk(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(BasicBlk, self).__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != self.expansion * out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, self.expansion * out_ch,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_ch)
            )
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:  # is not None
            x = self.downsample(x)  # resize the channel
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class BasictranBlk(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(BasictranBlk, self).__init__()
        self.conv1 = tranconv3x3(in_ch, out_ch, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = tranconv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != self.expansion * out_ch:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_ch, self.expansion * out_ch,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_ch)
            )
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:  # is not None
            x = self.downsample(x)  # resize the channel
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=7 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 压缩通道提取空间信息
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 经过卷积提取空间注意力权重
        x = torch.cat([max_out, avg_out], dim=1)
        out = self.conv1(x)
        # 输出非负
        out = self.sigmoid(out)  # torch.size([batch_size, 1, width, height])
        return out


# BasicBlock
class ResBlk(nn.Module):
    """
    resnet block
    """
    def __init__(self, ch_in, ch_out, stride=1):
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h , w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        return out


class qua_classification(nn.Module):
    def __init__(self, args):
        super(qua_classification, self).__init__()
        self.cfg = args
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        self.blk1_1 = ResBlk(64, 64, stride=1)
        self.blk2_1 = ResBlk(64, 128, stride=2)
        self.blk3_1 = ResBlk(128, 256, stride=2)
        self.blk4_1 = ResBlk(256, 256, stride=1)
        self.blk5_1 = ResBlk(256, 512, stride=2)

        self.blk1_2 = ResBlk(64, 64, stride=1)
        self.blk2_2 = ResBlk(64, 128, stride=2)
        self.blk3_2 = ResBlk(128, 256, stride=2)
        self.blk4_2 = ResBlk(256, 256, stride=1)
        self.blk5_2 = ResBlk(256, 512, stride=2)

        self.blk1_3 = ResBlk(64, 64, stride=1)
        self.blk2_3 = ResBlk(64, 128, stride=2)
        self.blk3_3 = ResBlk(128, 256, stride=2)
        self.blk4_3 = ResBlk(256, 256, stride=1)
        self.blk5_3 = ResBlk(256, 512, stride=2)

        self.blk1_4 = ResBlk(64, 64, stride=1)
        self.blk2_4 = ResBlk(64, 128, stride=2)
        self.blk3_4 = ResBlk(128, 256, stride=2)
        self.blk4_4 = ResBlk(256, 256, stride=1)
        self.blk5_4 = ResBlk(256, 512, stride=2)

        self.blk1234_1 = ResBlk(64, 64, stride=1)
        self.blk12_1 = ResBlk(128, 128, stride=1)
        self.blk34_1 = ResBlk(128, 128, stride=1)
        self.blk13_1 = ResBlk(256, 256, stride=1)
        self.blk24_1 = ResBlk(256, 256, stride=1)
        self.sa_m = SpatialAttention()
        self.sa_p = SpatialAttention()

        self.linear1 = nn.Linear(512, args['Categories_Number'])
        self.linear2 = nn.Linear(512, args['Categories_Number'])
        self.linear3 = nn.Linear(512, args['Categories_Number'])
        self.linear4 = nn.Linear(512, args['Categories_Number'])

    def forward(self, x):
        bs = int(x.shape[0]/4)
        ms, pan, ms_gan, pan_gan = x[:bs], x[bs:2*bs], x[2*bs:3*bs], x[-bs:]

        # print(ms.shape, pan.shape, ms_gan.shape, pan_gan.shape)
        out1 = F.gelu(self.conv1(ms))
        out2 = F.gelu(self.conv2(pan))
        out3 = F.gelu(self.conv3(ms_gan))
        out4 = F.gelu(self.conv4(pan_gan))

        # # out = torch.cat([out1, out2, out3, out4])
        # # out = F.gelu(self.blk1234_1(out))
        #
        out1 = F.gelu(self.blk2_1(F.gelu(self.blk1_1(out1))))
        out2 = F.gelu(self.blk2_2(F.gelu(self.blk1_2(out2))))
        out3 = F.gelu(self.blk2_3(F.gelu(self.blk1_3(out3))))
        out4 = F.gelu(self.blk2_4(F.gelu(self.blk1_4(out4))))

        out1 = torch.mul(out1, self.sa_m(out4))
        out2 = torch.mul(out2, self.sa_p(out3))

        out5 = torch.cat([out1, out2])  # 128
        out5 = F.gelu(self.blk12_1(out5))
        out6 = torch.cat([out3, out4])
        out6 = F.gelu(self.blk34_1(out6))

        out1 = F.gelu(self.blk4_1(F.gelu(self.blk3_1(out5[:bs]))))
        out2 = F.gelu(self.blk4_2(F.gelu(self.blk3_2(out5[bs:]))))
        out3 = F.gelu(self.blk4_3(F.gelu(self.blk3_3(out6[:bs]))))
        out4 = F.gelu(self.blk4_4(F.gelu(self.blk3_4(out6[:bs]))))

        out5 = torch.cat([out1, out3])  # 256
        out5 = F.gelu(self.blk13_1(out5))
        out6 = torch.cat([out2, out4])
        out6 = F.gelu(self.blk24_1(out6))

        out1 = F.gelu(self.blk5_1(out5[:bs]))
        out2 = F.gelu(self.blk5_2(out6[:bs]))
        out3 = F.gelu(self.blk5_3(out5[bs:]))
        out4 = F.gelu(self.blk5_4(out6[bs:]))
        out1 = F.avg_pool2d(out1, 2)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.linear1(out1)
        out2 = F.avg_pool2d(out2, 2)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.linear2(out2)
        out3 = F.avg_pool2d(out3, 2)
        out3 = out3.view(out3.size(0), -1)
        out3 = self.linear3(out3)
        out4 = F.avg_pool2d(out4, 2)
        out4 = out4.view(out4.size(0), -1)
        out4 = self.linear4(out4)
        out = torch.cat([out1, out2, out3, out4])  # 512

        # out1 = F.gelu(self.conv1(ms))
        # out1 = F.gelu(self.blk5_1(F.gelu(self.blk4_1(
        #     F.gelu(self.blk3_1(F.gelu(self.blk2_1(F.gelu(self.blk1_1(out1))))))))))
        # out2 = F.gelu(self.conv2(pan))
        # out2 = F.gelu(self.blk5_2(F.gelu(self.blk4_2(
        #     F.gelu(self.blk3_2(F.gelu(self.blk2_2(F.gelu(self.blk1_2(out2))))))))))
        # out3 = F.gelu(self.conv3(ms_gan))
        # out3 = F.gelu(self.blk5_3(F.gelu(self.blk4_3(
        #     F.gelu(self.blk3_3(F.gelu(self.blk2_3(F.gelu(self.blk1_3(out3))))))))))
        # out4 = F.gelu(self.conv4(pan_gan))
        # out4 = F.gelu(self.blk5_4(F.gelu(self.blk4_4(
        #     F.gelu(self.blk3_4(F.gelu(self.blk2_4(F.gelu(self.blk1_4(out4))))))))))
        # out = torch.cat([out1, out2, out3, out4])
        return out


class Four_branch(nn.Module):
    def __init__(self, args):
        super(Four_branch, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        self.blk1_1 = ResBlk(64, 64, stride=1)
        self.blk2_1 = ResBlk(64, 128, stride=2)
        self.blk3_1 = ResBlk(128, 256, stride=2)
        self.blk4_1 = ResBlk(256, 256, stride=1)
        self.blk5_1 = ResBlk(256, 512, stride=2)

        self.blk1_2 = ResBlk(64, 64, stride=1)
        self.blk2_2 = ResBlk(64, 128, stride=2)
        self.blk3_2 = ResBlk(128, 256, stride=2)
        self.blk4_2 = ResBlk(256, 256, stride=1)
        self.blk5_2 = ResBlk(256, 512, stride=2)

        self.blk1_3 = ResBlk(64, 64, stride=1)
        self.blk2_3 = ResBlk(64, 128, stride=2)
        self.blk3_3 = ResBlk(128, 256, stride=2)
        self.blk4_3 = ResBlk(256, 256, stride=1)
        self.blk5_3 = ResBlk(256, 512, stride=2)

        self.blk1_4 = ResBlk(64, 64, stride=1)
        self.blk2_4 = ResBlk(64, 128, stride=2)
        self.blk3_4 = ResBlk(128, 256, stride=2)
        self.blk4_4 = ResBlk(256, 256, stride=1)
        self.blk5_4 = ResBlk(256, 512, stride=2)

        self.blk1234_1 = ResBlk(64, 64, stride=1)
        self.blk12_1 = ResBlk(128, 128, stride=1)
        self.blk34_1 = ResBlk(128, 128, stride=1)
        self.blk13_1 = ResBlk(256, 256, stride=1)
        self.blk24_1 = ResBlk(256, 256, stride=1)
        self.sa_m = SpatialAttention()
        self.sa_p = SpatialAttention()

        self.linear1 = nn.Linear(512, args['Categories_Number'])
        self.linear2 = nn.Linear(512, args['Categories_Number'])
        self.linear3 = nn.Linear(512, args['Categories_Number'])
        self.linear4 = nn.Linear(512, args['Categories_Number'])

    def forward(self, x):
        bs = int(x.shape[0]/4)
        ms, pan, ms_gan, pan_gan = x[:bs], x[bs:2*bs], x[2*bs:3*bs], x[-bs:]

        # print(ms.shape, pan.shape, ms_gan.shape, pan_gan.shape)
        out1 = F.gelu(self.conv1(ms))
        out2 = F.gelu(self.conv2(pan))
        out3 = F.gelu(self.conv3(ms_gan))
        out4 = F.gelu(self.conv4(pan_gan))
        # print(out1.shape, out2.shape, out3.shape, out4.shape)
        out1 = F.gelu(self.blk2_1(F.gelu(self.blk1_1(out1))))
        out2 = F.gelu(self.blk2_2(F.gelu(self.blk1_2(out2))))
        out3 = F.gelu(self.blk2_3(F.gelu(self.blk1_3(out3))))
        out4 = F.gelu(self.blk2_4(F.gelu(self.blk1_4(out4))))
        # print(out1.shape, out2.shape, out3.shape, out4.shape)
        out1 = F.gelu(self.blk4_1(F.gelu(self.blk3_1(out1))))
        out2 = F.gelu(self.blk4_2(F.gelu(self.blk3_2(out2))))
        out3 = F.gelu(self.blk4_3(F.gelu(self.blk3_3(out3))))
        out4 = F.gelu(self.blk4_4(F.gelu(self.blk3_4(out4))))
        # print(out1.shape, out2.shape, out3.shape, out4.shape)
        out1 = F.gelu(self.blk5_1(out1))
        out2 = F.gelu(self.blk5_2(out2))
        out3 = F.gelu(self.blk5_3(out3))
        out4 = F.gelu(self.blk5_4(out4))
        # print(out1.shape, out2.shape, out3.shape, out4.shape)
        out1 = F.avg_pool2d(out1, 2)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.linear1(out1)
        out2 = F.avg_pool2d(out2, 2)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.linear2(out2)
        out3 = F.avg_pool2d(out3, 2)
        out3 = out3.view(out3.size(0), -1)
        out3 = self.linear3(out3)
        out4 = F.avg_pool2d(out4, 2)
        out4 = out4.view(out4.size(0), -1)
        out4 = self.linear4(out4)
        out = torch.cat([out1, out2, out3, out4])
        return out


def Net(args):
    return qua_classification(args)


def test():
    args = {
        'Categories_Number': 8
    }
    m = Generator(img_channels=4)
    p = Generator(img_channels=4)
    net = Net(args)
    y = net(torch.randn(80, 4, 16, 16))
    # print(y.size())


if __name__ == '__main__':
    test()