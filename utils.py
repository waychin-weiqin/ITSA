import os 
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from dataloader import readpfm as rp

# Check
def disparity_loader(path):
    return rp.readPFM(path)
    
def oxford_list(filepath, img_list_path):
    dirL = 'image_2/'
    dirR = 'image_3/'
    dirDisp = 'disp_occ_0/'
    
    if img_list_path is None:
        img_list = [img for img in os.listdir(os.path.join(filepath, dirL)) if img.find('png') > -1]
    else:
        img_list = np.loadtxt(img_list_path, dtype=str)
    
    image = [img for img in img_list]
    
    left = [filepath + dirL + img for img in image]
    right = [filepath + dirR + img for img in image]
    disp = [filepath + dirDisp + img for img in image]
    
    return left, right, disp
        
    
def dataloader(filepath, submission=False, kitti=True):
    left_fold = 'image_2/'
    right_fold = 'image_3/'

    if not submission:
        disp_fold = 'disp_occ_0/'
    
    
    if kitti:
        image = [img for img in os.listdir(filepath + left_fold) if img.find('_10') > -1]
    else:
        image = [img for img in os.listdir(filepath + left_fold) if img.find('.jpg') > -1]

    left_test = [filepath + left_fold + img for img in image]
    right_test = [filepath + right_fold + img for img in image]
    
    if not submission:
        if kitti:
            disp_test = [filepath + disp_fold + img for img in image]
        else:
            disp_test = []
            for img in image:
                img = img.split('.')[0] + '.png'
                disp_test.append(filepath + disp_fold + img)    
            
        return left_test, right_test, disp_test
            
    else:
        return left_test, right_test

          
