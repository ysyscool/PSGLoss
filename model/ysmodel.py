import torch
import torch.nn as nn
import torchvision.models as models

from model.ResNet import B2_ResNet,B1_ResNet,B1_DRN
import torch.nn.functional as F

from torch.autograd import Variable
import numpy as np
from math import exp


class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()

	def	forward(self, input, target):
		N = target.size(0)
		smooth = 1.

		input_flat = input.view(N, -1)
		target_flat = target.view(N, -1)

		intersection = input_flat * target_flat

		loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
		loss = 1 - loss.sum() / N

		return loss

class KldLoss(nn.Module):
    def __init__(self):
        super(KldLoss, self).__init__()
    
    def	forward(self, input, target):
        N = target.size(0)
        eps = 1e-8 
        input_flat = input.view(N, -1)  #(20, 147456)
        target_flat = target.view(N, -1)
        # _,M=input_flat.size()
        # print (input_flat.size()) #(20, 147456)
        # print (input_flat.sum(1).unsqueeze(1).size()) # (20,1)
        
        Q = torch.div(input_flat, input_flat.sum(1).unsqueeze(1) + eps)   #prediction
        P = torch.div(target_flat, target_flat.sum(1).unsqueeze(1) + eps)  #ground turth

        kld = P* torch.log(eps +  P/ (eps + Q))
        loss = kld.sum() / N
        return loss



class tvdLoss(nn.Module):
    def __init__(self):
        super(tvdLoss, self).__init__()
    
    def	forward(self, input, target):
        N = target.size(0)
        eps = 1e-8 
        input_flat = input.view(N, -1)  #(20, 147456)
        target_flat = target.view(N, -1)
        # _,M=input_flat.size()
        # print (input_flat.size()) #(20, 147456)
        # print (input_flat.sum(1).unsqueeze(1).size()) # (20,1)
        
        Q = torch.div(input_flat, input_flat.sum(1).unsqueeze(1) + eps)   #prediction
        P = torch.div(target_flat, target_flat.sum(1).unsqueeze(1) + eps)  #ground turth

        # kld = P* torch.log(eps +  P/ (eps + Q))
        tv = torch.abs(P-Q)
        loss = tv.sum() / N
        return loss


class DicebceLoss(nn.Module):
	def __init__(self):
		super(DicebceLoss, self).__init__()

	def	forward(self, input, target):
		N = target.size(0)
		smooth = 1.

		input_flat = input.view(N, -1)
		target_flat = target.view(N, -1)

		intersection = input_flat * target_flat

		loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
		loss = 1 - loss.sum() / N
		loss2 = torch.nn.BCELoss()(input, target)
		# loss = 1 - loss.sum() / N
        # loss2 = torch.nn.BCELoss()(input, target)

		return loss+loss2

class tvdbceLoss(nn.Module):
    def __init__(self):
        super(tvdbceLoss, self).__init__()
    
    def	forward(self, input, target):
        N = target.size(0)
        eps = 1e-8 
        input_flat = input.view(N, -1)  #(20, 147456)
        target_flat = target.view(N, -1)
        # _,M=input_flat.size()
        # print (input_flat.size()) #(20, 147456)
        # print (input_flat.sum(1).unsqueeze(1).size()) # (20,1)
        
        Q = torch.div(input_flat, input_flat.sum(1).unsqueeze(1) + eps)   #prediction
        P = torch.div(target_flat, target_flat.sum(1).unsqueeze(1) + eps)  #ground turth

        # kld = P* torch.log(eps +  P/ (eps + Q))
        tv = torch.abs(P-Q)
        loss = tv.sum() / N
        loss2 = torch.nn.BCELoss()(input, target)
        return loss +loss2


##########psg loss module
class postpnet(nn.Module):
    def __init__(self, kernel_size=3):
        super(postpnet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)

    def forward(self, x , y):
        # print (x.size())
        # x =F.sigmoid(x)
        x = self.maxpool(x)
        x = x.sigmoid()
        x = x * y
        
        return x      

#########model def######################

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class DIM_BAM(nn.Module):
    # RFB-like multi-scale module
    def __init__(self, in_channel, out_channel,Weighted = True, reduction=4, shortcut = True):
        super(DIM_BAM, self).__init__()
        self.relu = nn.ReLU(True)
        self.Weighted = Weighted
        self.shortcut = shortcut
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            # BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1, dilation=1),
            # BasicConv2d_nobn(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            # BasicConv2d_nobn(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=2, dilation=2)
        )
        self.branch2 = nn.Sequential(
            # BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1, dilation=1),
            BasicConv2d(out_channel, out_channel, 3, padding=4, dilation=4)
        )
        self.branch3 = nn.Sequential(
            # BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1, dilation=1),
            BasicConv2d(out_channel, out_channel, 3, padding=8, dilation=8)
        )
        self.conv_cat = BasicConv2d(out_channel, out_channel, 3, padding=1)
        
        self.conv_res = BasicConv2d(out_channel, out_channel, 1)
        self.fc1 = nn.Sequential(
            nn.Linear(out_channel, out_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel // reduction, 3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x0)
        x2 = self.branch2(x0)
        x3 = self.branch3(x0)
        b, c, _, _ = x0.size()
        y = F.avg_pool2d(x0, kernel_size=x0.size()[2:]).view(b, c)
        y = self.fc1(y).view(b, 3, 1, 1)
        yall = list( torch.split(y,1,dim=1))
        y1,y2,y3 = yall[0],yall[1],yall[2]
        if self.Weighted :
            # po = x1 * y1*3./(y1+y2+y3) + x2*y2*3./(y1+y2+y3) +x3*y3*3./(y1+y2+y3)
            po = x1 * y1 + x2*y2 +x3*y3
        else:
            po = x1+x2+x3
        # if self.shortcut:
        #     po = po + x0
        # po = x1 * y1 + x2*y2 +x3*y3
        x_cat = self.conv_cat(po)
        x_cat = self.conv_res(x_cat)
        x = self.relu(x_cat + x0)
        return x

class fpn_aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation model, such as DSS, amulet, and so on.
    # used after MSF
    #bn version
    def __init__(self, channel,Weighted =True):
        super(fpn_aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # BasicConv2d( 256, channel, kernel_size=1, stride=1, padding=0)
        self.smooth1 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)       
        self.smooth2 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.smooth3 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)

        self.latlayer1 = BasicConv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        self.latlayer2 =BasicConv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = BasicConv2d( channel , channel, kernel_size=1, stride=1, padding=0)
        self.latlayer4 =BasicConv2d( channel , channel, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x1, x2, x3 ,x4):
        # x1_1 = 
        x1 =  self.latlayer1(x1)
        p3 =  self._upsample_add(x1,self.latlayer2(x2))
        p3 =  self.smooth1(p3)
        p3 = self.upsample( p3)
        p4 =  self._upsample_add(p3,self.latlayer3(x3))
        # p4o = self.dimrfb(p4)
        p4 = self.smooth2(p4)
        p5 =  self._upsample_add(p4,self.latlayer4(x4))
        p5 = self.smooth3(p5)
        # p5o = self.dimrfb(p5)
        p4 = self.upsample( p4)
        p3 = self.upsample( p3)
        return p3,p4,p5



class ysmodel(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=64, Weighted = True):  #32
        super(ysmodel, self).__init__()
        self.resnet = B1_ResNet()
        self.rfb1_1 = BasicConv2d( 256, channel, kernel_size=1, stride=1, padding=0)

        self.rfb2_1 = DIM_BAM( 512, channel, Weighted)
        self.rfb3_1 = DIM_BAM( 1024, channel, Weighted)
        self.rfb4_1 = DIM_BAM( 2048, channel, Weighted)

        self.agg1 = fpn_aggregation(channel, Weighted)

        self.fusion0 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.fusion1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.finalout = nn.Conv2d( channel, 1, kernel_size=1, stride=1, padding=0)

        self.dimrfb2 = DIM_BAM(channel, channel, Weighted)
        self.dimrfb2a = DIM_BAM(channel, channel, Weighted)
        self.dimrfb2b = DIM_BAM(channel, channel, Weighted)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        if self.training:
            self.initialize_weights()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32

        x2_1 = x2
        x3_1 = self.resnet.layer3_1(x2_1)  # 1024 x 16 x 16
        x4_1 = self.resnet.layer4_1(x3_1)  # 2048 x 8 x 8
        x1_1 = self.rfb1_1(x1)
        x2_1 = self.rfb2_1(x2_1)
        x3_1 = self.rfb3_1(x3_1)
        x4_1 = self.rfb4_1(x4_1)
        am, attention_map, detection_map = self.agg1(x4_1, x3_1, x2_1,x1_1)

        detection_map = self.dimrfb2(detection_map)
        detection_map = self.dimrfb2a(detection_map)
        detection_map = self.dimrfb2b(detection_map)
        am = self.finalout(self.fusion1(self.fusion0(am)))
        attention_map = self.finalout(self.fusion1(self.fusion0(attention_map)))
        detection_map = self.finalout(self.fusion1(self.fusion0(detection_map)))

        return self.upsample(am),self.upsample(attention_map), self.upsample(detection_map)
        
    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v

        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params) 
