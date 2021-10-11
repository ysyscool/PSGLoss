import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
from scipy import misc


from model.ysmodel import *
from data import test_dataset2
import time
import cv2

def get_test_info(sal_mode='ECSSD'):
    if sal_mode == 'ECSSD':
        image_root = '/home/syang/ysdata/salientobj_dataset/dataset_test/ECSSD/Imgs/'
        image_source = '/home/syang/ysdata/salientobj_dataset/dataset_test/ECSSD/test.lst'
    elif sal_mode == 'PASCAL':
        image_root = '/home/syang/ysdata/salientobj_dataset/dataset_test/PASCALS/Imgs/'
        image_source = '/home/syang/ysdata/salientobj_dataset/dataset_test/PASCALS/test.lst'
    elif sal_mode == 'DUT-OMRON':
        image_root = '/home/syang/ysdata/salientobj_dataset/dataset_test/DUTOMRON/Imgs/'
        image_source = '/home/syang/ysdata/salientobj_dataset/dataset_test/DUTOMRON/test.lst'
    elif sal_mode == 'HKUIS':
        image_root = '/home/syang/ysdata/salientobj_dataset/dataset_test/HKU-IS/Imgs/'
        image_source = '/home/syang/ysdata/salientobj_dataset/dataset_test/HKU-IS/test.lst'
    elif sal_mode == 'SOD':
        image_root = '/home/syang/ysdata/salientobj_dataset/dataset_test/SOD/Imgs/'
        image_source = '/home/syang/ysdata/salientobj_dataset/dataset_test/SOD/test.lst'
    elif sal_mode == 'DUTS-TEST':
        image_root = '/home/syang/ysdata/salientobj_dataset/dataset_test/DUTS-TE/Imgs/'
        image_source = '/home/syang/ysdata/salientobj_dataset/dataset_test/DUTS-TE/test.lst'
    elif sal_mode == 'm_r': # for speed test  n.a.
        image_root = '/home/syang/ysdata/salientobj_dataset/dataset_test/MSRA/Imgs_resized/'
        image_source = '/home/syang/ysdata/salientobj_dataset/dataset_test/MSRA/test_resized.lst'
    elif sal_mode == 'kk': # for speed test  n.a.
        image_root = '/home/syang/ysdata/QA_dataset/koniq10k_1024x768/1024x768/'
        image_source = ''
    return image_root, image_source


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
# parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
parser.add_argument('--checkpointfile', type=str, default='', help='loaded model file')
parser.add_argument('--modelchoice', type=str, default='ysmodel')
parser.add_argument('--loss', type=str, default='')
parser.add_argument('--psgloss', action='store_true', help='has postploss')
parser.add_argument('--alpha', type=float, default=1.0, help='has postploss')
parser.add_argument('--kernel_size', type=int, default=3, help='has postploss')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--weighteddim', action='store_false', help='weighted dim')
# parser.add_argument('--otherbackbone', action='store_true', help='weighted dim')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--noclip', action='store_true', help='noclip')
parser.add_argument('--modelv', type=str, default='')
opt = parser.parse_args()

if opt.modelchoice == 'ysmodel':
    model = ysmodel(Weighted = opt.weighteddim)
else:
    print('using default ysmodel')
    model = ysmodel(Weighted = opt.weighteddim)

print (opt.checkpointfile)

# model=nn.DataParallel(model)   #multi-gpu testing

model.load_state_dict(torch.load(opt.checkpointfile))
    # model = CPD_VGG()

model.cuda()
model.eval()

# test_datasets =['ECSSD']
# test_datasets = ['PASCAL', 'ECSSD', 'DUT-OMRON', 'DUTS-TEST', 'HKUIS','SOD']

# test_datasets = ['ECSSD',  'HKUIS' ,'PASCAL', 'SOD', 'DUT-OMRON',  'DUTS-TEST']
test_datasets = ['ECSSD',  'HKUIS' ,'PASCAL', 'SOD', 'DUT-OMRON',  'DUTS-TEST']
# test_datasets = ['ECSSD',  'HKUIS' ]

# test_datasets = ['DUTS-TEST' ]
for dataset in test_datasets:

    save_path = './results/'  + opt.modelchoice + str(opt.trainsize)+ '_' + str(opt.psgloss) + '/'+  dataset + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root,_ = get_test_info(dataset)
    # image_root = dataset_path + dataset + '/images/'
    gt_root = image_root
    print('testing')
    print(image_root)
    print (save_path)

    # print(len(image_root))

    test_loader = test_dataset2(image_root, gt_root, opt.testsize)
    accu_time = 0.0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        curr0 = time.time()
        _,_, res = model(image)
        accu_time += time.time() - curr0
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255*res
        cv2.imwrite(save_path+name, res)
        
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # misc.imsave(save_path+name, res)
    
    average_time = accu_time/test_loader.size
    print('average_time:', average_time)
