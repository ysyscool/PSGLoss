import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pdb, os, argparse
from datetime import datetime

from model.ysmodel import *
from data import get_loader
from utils import clip_gradient, adjust_lr

# seed1 = 1026

# np.random.seed(seed1)
# torch.manual_seed(seed1)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
# parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
parser.add_argument('--modelchoice', type=str, default='ysmodel')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
parser.add_argument('--trainset', type=str, default='DUTS-TR')
parser.add_argument('--loss', type=str, default='')
parser.add_argument('--onlypsgloss', action='store_true', help='only has psgloss')  #not working
parser.add_argument('--psgloss', action='store_true', help='has PSGLosss')
parser.add_argument('--alpha', type=float, default=1.0, help='has PSGLosss')
parser.add_argument('--kernel_size', type=int, default=3, help='has psgloss')
parser.add_argument('--weighteddim', action='store_false', help='weighted dim')
parser.add_argument('--randomflip', action='store_true', help='randomflip')
opt = parser.parse_args()

print(opt)
# print('Learning Rate: {} ResNet: {} Modelchoice: {} , postprocessing: {}'.format(opt.lr, opt.is_ResNet ,opt.modelchoice, opt.postp))
# build models
# if opt.is_ResNet:
if opt.modelchoice == 'ysmodel':
    model = ysmodel(Weighted = opt.weighteddim)
else:
    print('using default ysmodel')
    model = ysmodel(Weighted = opt.weighteddim)
# else:
    # model = CPD_VGG()

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# /home/syang/ysdata/salientobj_dataset/DUTS/DUTS-TR/DUTS-TR-Image

image_root = '/home/syang/ysdata/salientobj_dataset/DUTS/DUTS-TR/DUTS-TR-Image/'
gt_root = '/home/syang/ysdata/salientobj_dataset/DUTS/DUTS-TR/DUTS-TR-Mask/'


train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, randomflip = opt.randomflip)
total_step = len(train_loader)

if opt.loss =='l2':
    CE =torch.nn.MSELoss()
elif opt.loss =='kld':
    CE =KldLoss()
elif opt.loss =='l1':
    CE =torch.nn.L1Loss()
elif opt.loss =='tvd':
    CE =tvdLoss()
elif opt.loss =='bce':
    CE =torch.nn.BCELoss()
elif opt.loss =='dice':
    CE =DiceLoss()
elif opt.loss =='dicebce':
    CE =DicebceLoss()
else:
    CE =torch.nn.BCEWithLogitsLoss()
    print ('using normal bce loss')
# CE2 =torch.nn.BCELoss()

def print_network(model, name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(name)
    print(model)
    print("The number of parameters: {}".format(num_params))

def train(train_loader, model, optimizer, epoch):
    model.train()
    save_path ='models2/' + opt.modelchoice + str(opt.trainsize)+ '_' + str(opt.psgloss)+ str(opt.alpha) + '_' + str(opt.kernel_size) + '_aug' + str(opt.randomflip) + '_bs' + str(opt.batchsize) + '_' + str(opt.weighteddim)  + '_' + str(opt.loss) + '_lr' + str(opt.lr)+  '/'

    print (save_path)   
    # print_network(model, opt.modelchoice )
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        images = images.cuda()
        gts = gts.cuda()

        _, atts, dets = model(images)

        loss1 = CE(atts.sigmoid(), gts)
        loss2 = CE(dets.sigmoid(), gts)

        if opt.psgloss:
            with torch.no_grad():    
                gts1 =postpnet( kernel_size=opt.kernel_size)(atts,gts)
                #
                gts2 =postpnet( kernel_size=opt.kernel_size)(dets,gts)  

     
            loss1a = CE(atts.sigmoid(), gts1)
            loss2a = CE(dets.sigmoid(), gts2)

            if opt.onlypsgloss:
                loss = loss2a 
            else:
                loss = loss2 + loss2a *opt.alpha 
        else:
            loss = loss2
            # loss = loss2 + loss1
        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 100 == 0 or i == total_step:
            if opt.psgloss:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f}  Loss1a: {:.4f}  Loss2: {:0.4f}  Loss2a: {:.4f} Total loss: {:.4f} '.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss1a.data, loss2.data, loss2a.data, loss.data))
            else:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss2.data))

    # if opt.is_ResNet:
    #     save_path = 'models/CPD_Resnet_'+ opt.modelchoice + '_' + str(opt.postp) +  '/'
    # else:
    #     save_path = 'models/CPD_VGG_'+ opt.modelchoice + '/'
    # print (save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 20 == 0   :  #5  or epoch <6 
        print ('saving %d' % epoch)
        torch.save(model.state_dict(), save_path + opt.trainset + '_w.pth' + '.%d' % epoch)
    

print("Let's go!")
for epoch in range(1, opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
