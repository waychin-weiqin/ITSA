import os
import time
import math
import argparse
import skimage.io
import os.path as osp
from PIL import Image
import matplotlib.pyplot as plt 

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF

from utils import *
from models.metric import *
from dataloader import readpfm as rp
import dataloader.middleburyinferlist as listM
import dataloader.sceneflowinferlist as s_lst

# Argparse Load Saved Model
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default='0', type=str, help='Select GPU. Default: 0')
parser.add_argument('--loadmodel', default=None, help='Path to state_dict. Default: None')
parser.add_argument('--model', default='PSMNet', help='Select network: [PSMNet/ GwcNet /CFNet]. Default: PSMNet')
parser.add_argument('--maxdisp', type=int, default=192, help='Maximum disparity range. Default: 192')
parser.add_argument('--savepath', default=None, help='Path to directory for saving disparity maps. Default: None')

# Dataset
## KITTI
parser.add_argument('--kitti15', action='store_true', default=False, help='Test using KITTI2015 dataset. Default: False')
parser.add_argument('--kitti12', action='store_true', default=False, help='Test using KITTI2012 dataset. Default: False')

## Middlebury
parser.add_argument('--midFull', action='store_true', default=False, help='Test using Middlebury-Full dataset. Default: False')
parser.add_argument('--midHalf', action='store_true', default=False, help='Test using Middlebury-Half dataset. Default: False')
parser.add_argument('--midQuar', action='store_true', default=False, help='Test using Middlebury-Quarter dataset. Default: False')

## ETH3D
parser.add_argument('--eth', action='store_true', default=False, help='Test using ETH3D dataset. Default: False')

parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA training. Default: False')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (Default: 1)')
parser.add_argument('--verbose', action='store_true', default=False, help='Print progress for each sample. Default: False')
args = parser.parse_args()

kitti = args.kitti12 or args.kitti15 
mid = args.midFull or args.midHalf or args.midQuar

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if not args.no_cuda:
    torch.cuda.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    print('>>>[INFO] Using GPU: {}'.format(args.cuda))
else:
    print('>>>[INFO] Not using GPU')


if args.model == 'PSMNet': 
    from models.PSMNet.stackhourglass import PSMNet
    model = PSMNet(args.maxdisp)
    
elif args.model == 'GwcNet': 
    from models.GwcNet.stackhourglass import GwcNet_G
    model = GwcNet_G(args.maxdisp)
    
elif args.model == 'CFNet': 
    from models.CFNet.stackhourglass import CFNet
    model = CFNet(args.maxdisp)

else:
    raise Exception("Invalid network is selected. Expected one from [PSMNet, GwcNet, CFNet].")

model = nn.DataParallel(model)
model.cuda()
    
# Load testing data
if args.kitti15:
    print(">>>[INFO] Loading data: KITTI 2015")
    filepath = '/media/SSD2/wei/Dataset/KITTI/data_scene_flow/train/'
    test_left, test_right, test_disp = dataloader(filepath, submission=False)
    if args.savepath:
        save_path = osp.join(args.savepath, 'KITTI', '2015')

elif args.kitti12:
    print(">>>[INFO] Loading data: KITTI 2012")
    filepath = '/media/SSD2/wei/Dataset/KITTI/data_stereo_flow/training/'
    test_left, test_right, test_disp = dataloader(filepath, submission=False)
    if args.savepath:
        save_path = osp.join(args.savepath, 'KITTI', '2012')

elif args.midFull:
    print(">>>[INFO] Loading data: Middlebury Full")
    filepath = '/media/SSD2/wei/Dataset/Middlebury-Full/'
    [test_left, test_right, test_disp, mask_val] = listM.dataloader(filepath)
    if args.savepath:
        save_path = osp.join(args.savepath, 'Middlebury', 'Full')

elif args.midHalf:
    print(">>>[INFO] Loading data: Middlebury Half")
    filepath = '/media/SSD2/wei/Dataset/Middlebury-Half/'
    [test_left, test_right, test_disp, mask_val] = listM.dataloader(filepath)
    if args.savepath:
        save_path = osp.join(args.savepath, 'Middlebury', 'Half')

elif args.midQuar:
    print(">>>[INFO] Loading data: Middlebury Quarter")
    filepath = '/media/SSD2/wei/Dataset/Middlebury-Quarter/'
    [test_left, test_right, test_disp, mask_val] = listM.dataloader(filepath)
    if args.savepath:
        save_path = osp.join(args.savepath, 'Middlebury', 'Quarter')

elif args.eth:
    print(">>>[INFO] Loading data: ETH3D")
    filepath = '/media/SSD2/wei/Dataset/ETH3D/'
    [test_left, test_right, test_disp, mask_val] = listM.dataloader(filepath)
    if args.savepath:
        save_path = osp.join(args.savepath, 'eth')

else:  # Scene Flow
    print(">>>[INFO] Loading data: FlyingThings3D")
    filepath = '/media/SSD2/wei/SceneFlow/'
    [test_left, test_right, test_disp] = s_lst.dataloader(filepath)
    if args.savepath:
        save_path = osp.join(args.savepath, 'SceneFlow')

if args.savepath is not None:
    print('>>>[INFO] Saving to {}'.format(save_path))
else:
    print('>>>[INFO] Not saving outputs')

state_dict_list = []

if os.path.isdir(args.loadmodel):  # if a directory with checkpoints is selected
    state_dict_list = os.listdir(args.loadmodel)
else:  # path to a specific checkpoint (e.g. ckpt.tar)
    state_dict_list.append(args.loadmodel)
  
print('>>>[INFO] Found {} ckpts in {}'.format(len(state_dict_list), args.loadmodel))


# Loop through ckpt_list
for i in range(len(state_dict_list)):
    mean_d1 = 0
    mean_epe = 0
    avg_time = 0

    if len(state_dict_list) > 1:
        ckpt_path = os.path.join(args.loadmodel, 'ckpt_' + str(i+1) + '.tar')
    else:
        ckpt_path = state_dict_list[i]
      
    print(">>>[INFO] Loading: {}".format(ckpt_path))
      
    model = load_ckpt(model, state_dict_path=ckpt_path)
    
    for idx in range(len(test_left)):
        left_o = Image.open(test_left[idx]).convert('RGB')
        right_o = Image.open(test_right[idx]).convert('RGB')
    
        if kitti: 
            disp_o = Image.open(test_disp[idx])
            disp_true = np.ascontiguousarray(disp_o, dtype=np.float32)
            disp_true = torch.tensor(disp_true, dtype=torch.float32)
            disp_true = disp_true / 256
                
        elif (mid or args.eth):
            mask_o = Image.open(mask_val[idx])
            disp_o, _ = disparity_loader(test_disp[idx])
            disp_true = np.ascontiguousarray(disp_o, dtype=np.float32)
            disp_true = torch.tensor(disp_true, dtype=torch.float32)
            mask_o = torch.from_numpy(np.ascontiguousarray(mask_o, dtype=np.float32))
            mask = (disp_true > 0) & (disp_true < args.maxdisp) & (mask_o == 255)
            
        else:  #SceneFlow
            disp_o, _ = disparity_loader(test_disp[idx])
            disp_true = np.ascontiguousarray(disp_o, dtype=np.float32)
            disp_true = torch.tensor(disp_true, dtype=torch.float32)          
    
        left = process(left_o).numpy()
        right = process(right_o).numpy()
    
        left = np.reshape(left, [1, 3, left.shape[1], left.shape[2]])
        right = np.reshape(right, [1, 3, right.shape[1], right.shape[2]])
    
        top_padded_size = math.ceil(left.shape[2] / 64) * 64
        left_padded_size = math.ceil(left.shape[3] / 64) * 64
        top_pad = top_padded_size - left.shape[2]
        left_pad = left_padded_size - left.shape[3]
    
        left = np.lib.pad(left, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)
        right = np.lib.pad(right, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)
    
        left, right = torch.from_numpy(left), torch.from_numpy(right)
            
        start_time = time.time()
        
        model.eval()
    
        with torch.no_grad():
            dispPred = model(left.cuda(), right.cuda())  
            dispPred = dispPred.data.cpu()
            dispPred = torch.squeeze(dispPred)          
            
        time_per_img = time.time() - start_time
        avg_time += time_per_img
        
        # Remove padding (if any)
        if top_pad > 0:
            if left_pad == 0:
                dispPred = dispPred[top_pad:, :]
            else:
                dispPred = dispPred[top_pad:, :-left_pad]
        else:
            if left_pad > 0:
                dispPred = dispPred[:, :-left_pad]
    
        # Performance evaluation
        if kitti:
            # D1 (3-px)
            epe_err = epe_metric(dispPred, disp_true, (disp_true > 0)&(disp_true < args.maxdisp))
            d1_err = d1_metric(dispPred, disp_true, (disp_true > 0)&(disp_true < args.maxdisp))
                        
        elif mid:
            # 2-px 
            epe_err = epe_metric(dispPred, disp_true, mask)
            d1_err = thres_metric(dispPred, disp_true, thres=2, mask=mask)
            imgName = test_left[idx].split('/')[-2]
            if imgName in ['PianoL', 'Playroom', 'Playtable', 'Shelves', 'Vintage']:
                d1_err = d1_err * 0.5
                    
        elif args.eth:
            # 1-px
            epe_err = epe_metric(dispPred, disp_true, mask)
            d1_err = thres_metric(dispPred, disp_true, thres=1, mask=mask)
        
        else:
            epe_err = epe_metric(dispPred, disp_true, (disp_true < args.maxdisp) & (disp_true > 0))
            d1_err = d1_metric(dispPred, disp_true, (disp_true < args.maxdisp) & (disp_true > 0))
            
        mean_d1 += d1_err
        mean_epe += epe_err
    
        if args.verbose:
            if mid:
                print('[INFO] Processing: %d/%d   %s   D1: %.3f   EPE: %.3f   Time: %.3f' % (
                idx + 1, len(test_left), test_left[idx].split('/')[-2], d1_err * 100, epe_err, time_per_img))
            else:
                print('[INFO] Processing: %d/%d   %s   D1: %.3f   EPE: %.3f   Time: %.3f' % (
                idx + 1, len(test_left), test_left[idx].split('/')[-1], d1_err * 100, epe_err, time_per_img))
    
        if args.savepath is not None:
            if kitti:
                fileName = test_left[idx].split('/')[-1]
                
            elif mid:
                fileName = test_left[idx].split('/')[-2] + '_' + test_left[idx].split('/')[-1]
        
            else:
                if not osp.exists(osp.join(save_path, test_left[idx].split('/')[-4], test_left[idx].split('/')[-3])):
                    os.mkdir(osp.join(save_path, test_left[idx].split('/')[-4], test_left[idx].split('/')[-3]))
                    
                fileName = osp.join(test_left[idx].split('/')[-4], test_left[idx].split('/')[-3],
                                         test_left[idx].split('/')[-1])
                
            skimage.io.imsave(osp.join(save_path, fileName), (dispPred.numpy() * 256).astype('uint16'))
          
   
    if mid:
        print('[INFO] Mean D1: %.3f   Mean EPE: %.3f' % (mean_d1 * 100 / 12.5, mean_epe / len(test_left)))
    else:
        print('[INFO] Mean D1: %.3f   Mean EPE: %.3f' % (mean_d1 * 100 / len(test_left), mean_epe / len(test_left)))
        
    print('[INFO] Average Time: %.3f' % (avg_time / len(test_left)))
    
