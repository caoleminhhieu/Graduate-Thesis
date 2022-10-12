

#!/usr/bin/python3
# coding=utf-8

from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.AdaptiveAvgPool2d, nn.Softmax, nn.Dropout2d)):
            pass
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(
            3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 3, stride=1, dilation=1)
        self.layer2 = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3 = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4 = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = nn.Sequential(nn.Conv2d(
            self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers = [Bottleneck(self.inplanes, planes, stride,
                             downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out2 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out2)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load(
            '/media/pcl/Bdata/MMC/D3Net/PFSNet/resnet/resnet50-19c8e357.pth'), strict=False)


class Rblock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Rblock, self).__init__()
        self.squeeze1 = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=3,
                                      stride=1, dilation=2, padding=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d(
            inplanes, outplanes, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.convg = nn.Conv2d(128, 128, 1)
        self.sftmax = nn.Softmax(dim=1)
        self.convAB = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bnAB = nn.BatchNorm2d(64)

    def forward(self, x, z):
        z = self.squeeze1(x) if z is None else self.squeeze1(x+z)
        x = self.squeeze2(x)
        x = torch.cat((x, z), 1)
        y = self.convg(self.gap(x))
        x = torch.mul(self.sftmax(y)*y.shape[1], x)
        x = F.relu(self.bnAB(self.convAB(x)), inplace=True)
        return x, z

    def initialize(self):
        weight_init(self)
        print("Rblock module init")


class dilated_Rblock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(dilated_Rblock, self).__init__()
        self.squeeze1_1 = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=3,
                                                  stride=1, dilation=2, padding=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze1_2 = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=3,
                                                  stride=1, dilation=3, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze1 = nn.Sequential(nn.Conv2d(2*outplanes, outplanes, kernel_size=1,
                                      stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.squeeze2 = nn.Sequential(nn.Conv2d(
            inplanes, outplanes, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.convg = nn.Conv2d(128, 128, 1)
        self.sftmax = nn.Softmax(dim=1)
        self.convAB = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bnAB = nn.BatchNorm2d(64)

    def forward(self, x, z):
        z1 = self.squeeze1_1(x) if z is None else self.squeeze1_1(x+z)
        z2 = self.squeeze1_2(x) if z is None else self.squeeze1_2(x+z)
        z = self.squeeze1(torch.cat((z1, z2), 1))

        x = self.squeeze2(x)
        x = torch.cat((x, z), 1)
        y = self.convg(self.gap(x))
        x = torch.mul(self.sftmax(y)*y.shape[1], x)
        x = F.relu(self.bnAB(self.convAB(x)), inplace=True)
        return x, z

    def initialize(self):
        weight_init(self)
        print("Rblock module init")


class Yblock(nn.Module):
    def __init__(self):
        super(Yblock, self).__init__()

        self.convA1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bnA1 = nn.BatchNorm2d(64)

        self.convB1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bnB1 = nn.BatchNorm2d(64)

        self.convAB = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bnAB = nn.BatchNorm2d(64)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.convg = nn.Conv2d(128, 128, 1)
        self.sftmax = nn.Softmax(dim=1)
      #  self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x, y):
        if x.size()[2:] != y.size()[2:]:
            y = F.interpolate(
                y, size=x.size()[2:], mode='bilinear', align_corners=True)
        fuze = torch.mul(x, y)
        y = F.relu(self.bnB1(self.convB1(fuze+y)), inplace=True)
        x = F.relu(self.bnA1(self.convA1(fuze+x)), inplace=True)
        x = torch.cat((x, y), 1)
        y = self.convg(self.gap(x))
        x = torch.mul(self.sftmax(y)*y.shape[1], x)
     #   x = self.dropout(x)
        x = F.relu(self.bnAB(self.convAB(x)), inplace=True)
        return x

    def initialize(self):
        weight_init(self)
        print("Yblock module init")


class PFSNet(nn.Module):
    def __init__(self, args):
        super(PFSNet, self).__init__()
        self.bkbone = ResNet()
        self.squeeze5 = Rblock(2048, 64)
        self.squeeze4 = Rblock(1024, 64)
        self.squeeze3 = Rblock(512, 64)
        self.squeeze2 = Rblock(256, 64)
        self.squeeze1 = Rblock(64, 64)

        self.conv1 = nn.Conv2d(64, 1024, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)

        self.conv2 = nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(512)

        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.Y11 = Yblock()
        self.Y12 = Yblock()
        self.Y13 = Yblock()
        self.Y14 = Yblock()

        self.Y21 = Yblock()
        self.Y22 = Yblock()
        self.Y23 = Yblock()

        self.Y31 = Yblock()
        self.Y32 = Yblock()

        self.Y41 = Yblock()

        self.linearp1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        if args.model == 'PFSNet':
            if args.pretrained == 'True':
                check_point = torch.load(args.pretrained_model)
                self.load_state_dict(check_point['model'])
                print('###### pre-trained PFSNet restored #####')

    def forward(self, x, shape=None):

        s1, s2, s3, s4, s5 = self.bkbone(x)

        s5, z = self.squeeze5(s5, None)
        z = F.interpolate(z, size=s4.size()[
                          2:], mode='bilinear', align_corners=True)
        z = F.relu(self.bn1(self.conv1(z)))
        s4, z = self.squeeze4(s4, z)
        z = F.interpolate(z, size=s3.size()[
                          2:], mode='bilinear', align_corners=True)
        z = F.relu(self.bn2(self.conv2(z)))
        s3, z = self.squeeze3(s3, z)
        z = F.interpolate(z, size=s2.size()[
                          2:], mode='bilinear', align_corners=True)
        z = F.relu(self.bn3(self.conv3(z)))
        s2, z = self.squeeze2(s2, z)
        z = F.interpolate(z, size=s1.size()[
                          2:], mode='bilinear', align_corners=True)
        z = F.relu(self.bn4(self.conv4(z)))
        s1, z = self.squeeze1(s1, z)

        s5 = self.Y14(s4, s5)
        s4 = self.Y13(s3, s4)
        s3 = self.Y12(s2, s3)
        s2 = self.Y11(s1, s2)

        s2 = self.Y21(s2, s3)
        s3 = self.Y22(s3, s4)
        s4 = self.Y23(s4, s5)

        s4 = self.Y32(s3, s4)
        s3 = self.Y31(s2, s3)

        s3 = self.Y41(s3, s4)

        shape = x.size()[2:] if shape is None else shape
        p1 = self.linearp1(s3)
        del s1, s2, s4, s5, z
        torch.cuda.empty_cache()
        return s3, p1


class OCR(nn.Module):
    def __init__(self, in_channels=64, key_channels=128, mid_channels=256, dropout=0.05, scale=1):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.key_channels = key_channels

        self.conv_in = nn.Sequential(nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))

        self.conv_q = nn.Sequential(nn.Conv2d(self.mid_channels, self.key_channels, kernel_size=1,
                                    stride=1, padding=0, bias=False), nn.BatchNorm2d(key_channels), nn.ReLU(inplace=True),
                                    nn.Conv2d(self.key_channels, self.key_channels, kernel_size=1,
                                    stride=1, padding=0, bias=False), nn.BatchNorm2d(key_channels), nn.ReLU(inplace=True))

        self.conv_k = nn.Sequential(nn.Conv2d(self.mid_channels, self.key_channels, kernel_size=1,
                                    stride=1, padding=0, bias=False), nn.BatchNorm2d(key_channels), nn.ReLU(inplace=True),
                                    nn.Conv2d(self.key_channels, self.key_channels, kernel_size=1,
                                    stride=1, padding=0, bias=False), nn.BatchNorm2d(key_channels), nn.ReLU(inplace=True))

        self.conv_v = nn.Sequential(nn.Conv2d(self.mid_channels, self.key_channels, kernel_size=1,
                                    stride=1, padding=0, bias=False), nn.BatchNorm2d(key_channels), nn.ReLU(inplace=True))

        self.conv_out = nn.Sequential(nn.Conv2d(self.key_channels, self.mid_channels, kernel_size=1,
                                                stride=1, padding=0, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))

        self.conv_fuse = nn.Sequential(
            nn.Conv2d(2*mid_channels, mid_channels,
                      kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, probs):
        feats = self.conv_in(feats)
        pixel_rep = feats

        # Extract salient representation
        batch_size, c, h, w = probs.size(0), probs.size(
            1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        context = torch.matmul(probs, feats).permute(
            0, 2, 1).unsqueeze(3)  # batch x k x c

        query = self.conv_q(pixel_rep).view(
            batch_size, self.key_channels, -1).permute(0, 2, 1)
        value = self.conv_v(context).view(
            batch_size, self.key_channels, -1).permute(0, 2, 1)
        key = self.conv_k(context).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = torch.sigmoid(sim_map)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, h, w)
        context = self.conv_out(context)

        pixel_rep = self.conv_fuse(torch.cat([context, pixel_rep], dim=1))
        return pixel_rep


class PFSNet_OCR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pfsnet = PFSNet(args)
        self.ocr = OCR(args.in_channels, args.key_channels,
                       args.mid_channels, args.dropout, args.scale)
        self.class_head = nn.Conv2d(
            args.mid_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)

        if args.model == 'PFSNet_OCR':
            if args.pretrained == 'True':
                check_point = torch.load(args.pretrained_model)
                self.load_state_dict(check_point['model'])
                print('###### pre-trained PFSNet_OCR Model restored #####')

    def forward(self, x):
        feats, aux_out = self.pfsnet(x)
        feats = self.ocr(feats, aux_out)
        out = self.class_head(feats)

        return torch.sigmoid(F.interpolate(aux_out, size=x.shape[2:], mode='bilinear', align_corners=True)), torch.sigmoid(F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True))


class dilated_PFSNet_OCR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pfsnet_ocr = PFSNet_OCR(args)
        self.pfsnet_ocr.pfsnet.squeeze5 = dilated_Rblock(2048, 64)

        if args.model == 'dilated_PFSNet_OCR':
            if args.pretrained == 'True':
                check_point = torch.load(args.pretrained_model)
                self.load_state_dict(check_point['model'])
                print('###### pre-trained dilated_PFSNet_OCR Model restored #####')

    def forward(self, x):
        return self.pfsnet_ocr(x)
