import os 
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from dataloader import readpfm as rp

def kl_div(pred, target, is_log=True, eps=1e-6):
    '''
    KL(P||Q) = sum(p(x) * log(p(x)) - log(q(x)))
        P(X) = Target
        Q(X) = Approx.
    '''
    if not is_log:
        pred = F.softmax(pred, dim=1)       # [B, D, H, W]
        target = F.softmax(target, dim=1)   # [B, D, H, W]

    pred = pred + eps
    target = target + eps
    
    # Clip off gradient
    target = target.detach()

    div = torch.mul(target, torch.log(target) - torch.log(pred))
    div = torch.sum(div, dim=1).sum(-1).sum(-1)     # [B, H, W]
    div = torch.mean(div)

    return div

def loss_func(disp_pred, disp_true, cost_pred, maxdisp=192, verbose=True):

    kl_loss = F.kl_div(F.softmax(cost_pred[0],dim=1), F.softmax(cost_pred[1], dim=1), reduction='batchmean', log_target=True) + \
              F.kl_div(F.softmax(cost_pred[1], dim=1), F.softmax(cost_pred[0], dim=1), reduction='batchmean',
                       log_target=True)

    mask = (disp_true > 0) & (disp_true < maxdisp)
    mask = mask.detach()

    dispLoss_1 = F.smooth_l1_loss(disp_pred[0][mask], disp_true[mask], reduction='mean')
    dispLoss_2 = F.smooth_l1_loss(disp_pred[1][mask], disp_true[mask], reduction='mean')

    loss = dispLoss_1 + dispLoss_2 + 0.1*kl_loss
    loss.backward()
    optimizer.step()

    if verbose:
        print("[INFO] Disparity Loss: {1} %.3f  {2} %.3f  KL Loss: %.3f"
              %(dispLoss_1.item(), dispLoss_2.item(), kl_loss.item()))

    return loss


def test(imgL, imgR, disp_true, model):
    model.eval()

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

    mask = (disp_true < args.maxdisp) & (disp_true > 0)
    mask = mask.detach()

    with torch.no_grad():
        output = model(imgL, imgR)

    disp_pred = output.data

    del output

    top_pad = disp_pred.shape[1] - disp_true.shape[1]
    right_pad = disp_pred.shape[2] - disp_true.shape[2]

    if right_pad == 0:
        disp_pred = disp_pred[:, top_pad:, :]
    else:
        disp_pred = disp_pred[:, top_pad:, :-right_pad]

    test_epe = epe_metric(disp_pred, disp_true, disp_true < args.maxdisp)
    test_disp_loss = F.smooth_l1_loss(disp_pred[mask], disp_true[mask], reduction='mean')

    test_loss = test_disp_loss

    torch.cuda.empty_cache()

    return test_epe.item(), test_loss.item()
    

import os 
import torch
from torchvision import transforms
from dataloader import readpfm as rp


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


def load_ckpt(model, state_dict_path=None):
    if state_dict_path == None:
        raise Exception("No ckpt path stated.")
    print("[INFO] Loading saved state dict")
    print('[INFO] from ' + state_dict_path)
    ckpt = torch.load(state_dict_path, map_location=torch.device('cpu'))
    state_dict = ckpt["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    
    return model

def process(img):
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}
                    
    preprocess = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(**__imagenet_stats)])
    img = preprocess(img)
    
    return img

def grad_process(grad, channel=False):
        #print(torch.quantile(grad.flatten(-2,-1), dim=-1, q=0.75))
        #print(torch.quantile(grad.flatten(-2,-1), dim=-1, q=0.5))
        #print(torch.quantile(grad.flatten(-2,-1), dim=-1, q=0.25))
        
        # UPDATED: 6-SEPT-2021
        # Remove "outliers"
        up_bound = torch.quantile(grad.flatten(-2,-1), q=0.75, dim=-1).view(grad.size(0),3,1,1)
        low_bound = torch.quantile(grad.flatten(-2,-1), q=0.25, dim=-1).view(grad.size(0),3,1,1)
        
        grad = torch.where(grad<=up_bound, grad, up_bound)
        grad = torch.where(grad>=low_bound, grad, low_bound)
        
        # Normalize to [0,1]
        grad = grad.abs()
        channel_max = grad.flatten(-2,-1).max(dim=-1)[0]
        channel_max = channel_max.view(grad.size(0),3,1,1)
        grad = grad.div(channel_max)  # [0, 1]
               
        spatial = grad.mean(1,keepdim=True) * (torch.randn_like(grad))#*0.4-0.2)#-0.5)
        #spatial
        
        if channel:
          channel = F.adaptive_avg_pool2d(grad.abs(), 1) 
          channel = channel * (torch.rand_like(channel)-0.5)
          
          return spatial, channel
          
        else:
          return spatial
          
def grad_processv2(grad, img):
    
    grad = grad.abs().cpu()
    grad = torch.mean(grad, dim=0)
    
    return newImg
    
          
def denorm(img):
    mu = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    sigma = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    img = (img * sigma) + mu

    return img

def normalize(x):
  x_ = torch.flatten(x, -2, -1)
  x_max = x_.max(-1)[0].view(3, 1, 1)
  x_min = x_.min(-1)[0].view(3, 1, 1)

  x = (x - x_min) / (x_max - x_min)

  return x
  
  

def patch_augment(img):
        img = transforms.ToTensor()(img)
        num_blocks = int(np.random.uniform(3, 6))

        for block_id in range(num_blocks):
            #noise = []
            kx = int(np.random.uniform(10, 40))  # Kernel size
            ky = int(np.random.uniform(10, 40))  # Kernel size

            cx = int(np.random.uniform(kx, img.shape[2] - kx))
            cy = int(np.random.uniform(ky, img.shape[1] - ky))

            patch = img[:, cy - ky:cy + ky, cx - kx:cx + kx]
            patch = torch.reshape(patch, shape=[3, (2 * kx * 2 * ky)])

            mu = torch.mean(patch, dim=1)
            sigma = torch.std(patch, dim=1)

            for i in range(3):
                noise = np.random.normal(loc=mu[i], scale=sigma[i], size=2)
                alpha = noise[0].astype("float32")
                beta = noise[1].astype("float32")
                canvas = img[i, cy - ky:cy + ky, cx - kx:cx + kx]
                #grains = np.random.normal(loc=0, scale=0.03, size=canvas.shape)
                canvas = (canvas * alpha) + beta + (torch.randn_like(canvas)*0.01)
                img[i, cy - ky:cy + ky, cx - kx:cx + kx] = canvas
                #noise.append(np.random.normal(loc=mu[i], scale=sigma[i] / 2.0, size=[2 * ky, 2 * kx]))

            #noise = np.stack(noise, axis=0)
            #img[:, cy - ky:cy + ky, cx - kx:cx + kx] += noise.astype('float32')
        img = torch.clip(img, min=0, max=1)
        img = transforms.ToPILImage()(img)

        return img