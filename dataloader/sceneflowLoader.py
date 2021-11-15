from dataloader import readpfm as rp
from PIL import Image
import torch.utils.data as data
import numpy as np
import random
from torchvision import transforms
import torch
import torchvision.transforms.functional as TF

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return rp.readPFM(path)

toTensor = transforms.ToTensor()
toPIL = transforms.ToPILImage()

class loadData(data.Dataset):
    def __init__(self, left, right, disp, training, augment=True, loader=default_loader, dploader=disparity_loader):
        self.left = left
        self.right = right
        self.disp = disp
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.augment = augment

    def __getitem__(self, idx):
        left = self.left[idx]
        right = self.right[idx]
        disp = self.disp[idx]

        imgL = self.loader(left)
        imgR = self.loader(right)
        disp_img, disp_scale = self.dploader(disp)
        disp_img = np.ascontiguousarray(disp_img, dtype=np.float32)

        if self.training:
            w, h = imgL.size
            th, tw = 256, 512

            x1 = random.randint(0, w-tw)
            y1 = random.randint(0, h-th)

            imgR = imgR.crop((x1, y1, x1 + tw, y1 + th))
            imgL = imgL.crop((x1, y1, x1 + tw, y1 + th))
            disp_img = disp_img[y1: y1+th, x1: x1+tw]
            
            imgL = self.add_sensor_noise(imgL)
            imgR = self.add_sensor_noise(imgR)
            
            imgL = self.process(imgL)
            imgR = self.process(imgR)
            
            return imgL, imgR, disp_img
              
        else:
            w, h = imgL.size
            imgL = imgL.crop((w-960, h-576, w, h))
            imgR = imgR.crop((w - 960, h - 576, w, h))

            imgL = self.process(imgL)
            imgR = self.process(imgR)

            return imgL, imgR, disp_img

    def __len__(self):
        return len(self.left)
        
    def add_sensor_noise(self, img):
        img = toTensor(img)
        mask = torch.randn_like(img) * 0.01 
        img = img + mask 
        img = torch.clip(img, 0, 1)
        img = toPIL(img)
        return img
        
        
    def process(self, img):
        __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}
   
        preprocess = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(**__imagenet_stats)])
        
        img = preprocess(img)
        
        return img
                

        
                     
                                          

    

    

   
