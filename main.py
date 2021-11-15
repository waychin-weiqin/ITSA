import os
import time
import argparse

import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from dataloader import sceneflowList as listSF
from dataloader import sceneflowLoader as loaderSF
from models.metric import *

parser = argparse.ArgumentParser()

parser.add_argument('--maxdisp', type=int, default=192,
                    help='maximum disparity')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train')
parser.add_argument('--batch', type=int, default=12,
                    help='training batch size. Default: 12')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='Verbose: Print detailed training information.')
parser.add_argument('--color', action='store_true', default=False,
                    help='Asymmetrical Colour Augmentation (For CFNet only).')

# Hyperparameters 
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning Rate. Default: 1e-3')
parser.add_argument('--itsa', action='store_true', default=False,
                    help='ITSA Domain Generalization.')
parser.add_argument('--lambd', type=float, default=1.0,
                    help='Hyperparameter Lambda for Fisher Loss term. Default: 1.0')
parser.add_argument('--eps', type=float, default=0.0,
                    help='Hyperparameter Epsilon for Perturbation Strength. Default: 0.0')

parser.add_argument('--datapath', default="/media/SSD2/wei/SceneFlow/",
                    help='datapath')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--savemodel', default='./checkpoints/',
                    help='save model')
parser.add_argument('--model', default='PSMNet',
                    help='Select model [PSMNet, GwcNet, CFNet]')

parser.add_argument('--cuda_group', type=int, default=1,
                    help='Select GPUs cluster [1: {0,1}, 2: {2,3}]')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()

if args.cuda_group == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
  
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.itsa:
    print('>>>[INFO] ITSA Domain Generalization: Enabled')
    print(">>>[INFO] Lambda: %.2f"%(args.lambd))
    print('>>>[INFO] EPS: %.4f'%(args.eps))
else:
    print('>>>[INFO] ITSA Domain Generalization: Disabled')

print(">>>[INFO] Model: %s"%(args.model))
print(">>>[INFO] Color Augmentation: %s"%(args.color))
    
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Data loaders 
all_left_img, all_right_img, all_left_disp, \
test_left_img, test_right_img, test_left_disp = listSF.dataloader(args.datapath)

TrainDataSF = loaderSF.loadData(all_left_img, all_right_img, all_left_disp, augment=args.color, training=True)
TestDataSF = loaderSF.loadData(test_left_img, test_right_img, test_left_disp, augment=False, training=False)

TrainImgLoader = torch.utils.data.DataLoader(TrainDataSF, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=False)
TestImgLoader = torch.utils.data.DataLoader(TestDataSF, batch_size=args.batch, shuffle=False, num_workers=4, drop_last=False)

# Load model
if args.model == "PSMNet":
    from models.PSMNet.stackhourglass import PSMNet 
    model = PSMNet(args.maxdisp, eps=args.eps, itsa=args.itsa)

elif args.model == "GwcNet":
    from models.GwcNet.stackhourglass import GwcNet_G 
    model = GwcNet_G(args.maxdisp, eps=args.eps, itsa=args.itsa)
    
elif args.model == 'CFNet':
    from models.CFNet.stackhourglass import CFNet
    model = CFNet(args.maxdisp, eps=args.eps, itsa=args.itsa)

else:
    raise Exception("Invalid selection. Please select one from the options [PSMNet/GwcNet/CFNet]")

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()
    
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

if args.loadmodel is not None:
    print(">>>[INFO] Loading pre-trained weights from %s"%(args.loadmodel))
    ckpt = torch.load(args.loadmodel)
    model.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])

# Helper functions 
def fisher_loss(feat1, feat2):
    lossF = torch.mean((feat1 - feat2).pow(2))
    
    return lossF


def compute_loss(featEsts, dispEsts, dispTrue, verbose=False):
    lossDisp = []
    
    # Generate mask 
    mask = (dispTrue > 0) & (dispTrue < args.maxdisp)
    mask = mask.detach()
    
    # Smooth L1 loss for disparity estimation 
    if args.model == "PSMNet":
        weights = [0.5, 0.7, 1.0]
    elif args.model == "GwcNet":
        weights = [0.5, 0.5, 0.7, 1.0]
    elif args.model == "CFNet":
        weights = [0.5 * 0.5, 0.5 * 0.7, 0.5 * 1.0, 1 * 0.5, 1 * 0.7, 1 * 1.0, 2 * 0.5, 2 * 0.7, 2 * 1.0]
      
    for batch_ii, (weight, dispEst) in enumerate(zip(weights, dispEsts)):        
        lossDisp.append(weight * F.smooth_l1_loss(dispEst[mask], dispTrue[mask], reduction="mean"))
    
    if args.itsa:
        fishLossL = fisher_loss(featEsts["left"], featEsts["left_scp"]) 
        fishLossR = fisher_loss(featEsts["right"], featEsts["right_scp"]) 
            
        fishLoss = (fishLossL + fishLossR).div(2.0)
    else:
        fishLoss = 0.0
    
    loss = sum(lossDisp) + args.lambd*(fishLoss)
    
    # Backpropagation and update 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 
    
    if verbose:
        if args.itsa:
            if args.model == "PSMNet":
                print(">>>[INFO] Disp {1}: %.3f  {2}: %.3f  {3}: %.3f\t Feat {L}: %.3f  {R}: %.3f"%(lossDisp[0].item(), lossDisp[1].item(), lossDisp[2].item(), fishLossL.item(), fishLossR.item()))
            elif args.model == "GwcNet":
                print(">>>[INFO] Disp {1}: %.3f  {2}: %.3f  {3}: %.3f  {4}: %.3f\t Feat {L}: %.3f  {R}: %.3f"%(lossDisp[0].item(), lossDisp[1].item(), lossDisp[2].item(), lossDisp[3].item(), fishLossL.item(), fishLossR.item()))
            elif args.model == "CFNet":
                print(">>>[INFO] Disp {1}: %.3f  {2}: %.3f  {3}: %.3f  {4}: %.3f  {5}: %.3f  {6}: %.3f  {7}: %.3f  {8}: %.3f  {9}: %.3f\t Feat {L}: %.3f  {R}: %.3f"%(lossDisp[0].item(), lossDisp[1].item(), lossDisp[2].item(), lossDisp[3].item(),lossDisp[4].item(), lossDisp[5].item(), lossDisp[6].item(), lossDisp[7].item(), lossDisp[8].item(), fishLossL.item(), fishLossR.item()))

        else:
            if args.model == "PSMNet":
                print(">>>[INFO] Disp {1}: %.3f  {2}: %.3f  {3}: %.3f"%(lossDisp[0].item(), lossDisp[1].item(), lossDisp[2].item()))
            elif args.model == "GwcNet":
                print(">>>[INFO] Disp {1}: %.3f  {2}: %.3f  {3}: %.3f  {4}: %.3f"%(lossDisp[0].item(), lossDisp[1].item(), lossDisp[2].item(), lossDisp[3].item()))
            elif args.model == "CFNet":
                print(">>>[INFO] Disp {1}: %.3f  {2}: %.3f  {3}: %.3f  {4}: %.3f  {5}: %.3f  {6}: %.3f  {7}: %.3f  {8}: %.3f  {9}: %.3f"%(lossDisp[0].item(), lossDisp[1].item(), lossDisp[2].item(), lossDisp[3].item(),lossDisp[4].item(), lossDisp[5].item(), lossDisp[6].item(), lossDisp[7].item(), lossDisp[8].item()))
        
    return loss.item()


def adjust_lr(optimizer, epoch, thres=10):
    if epoch <= thres:
        lr = args.lr
    else:
        lr = args.lr/2

    print(">>>[INFO] Learning rate: {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

def train(data, epoch):
    imgL, imgR, dispTrue = data
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()
        dispTrue = dispTrue.cuda()
        
    model.train()
    if args.itsa:
        featEsts, dispEsts = model(imgL, imgR)
        
    else:  
        dispEsts = model(imgL, imgR)
        
    loss = compute_loss(featEsts, dispEsts, dispTrue, verbose=args.verbose)
    
    return loss
    

def test(data):
    imgL, imgR, dispTrue = data
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()
        dispTrue = dispTrue.cuda()
    
    model.eval()
    with torch.no_grad():
        dispEst = model(imgL, imgR)
        
    # Remove padded regions 
    top_pad = dispEst.size(1) - dispTrue.size(1)
    right_pad = dispEst.size(2) - dispTrue.size(2)
    
    if right_pad == 0:
        dispEst = dispEst[:, top_pad:, :]
    else:
        dispEst = dispEst[:, top_pad:, :-right_pad]
        
    mask = (dispTrue < args.maxdisp) & (dispTrue > 0)
    mask = mask.detach()
        
    # Disparity Loss
    lossDisp = F.smooth_l1_loss(dispEst[mask], dispTrue[mask], reduction="mean")
    
    return lossDisp.item()


def save_model(total_train_loss, total_test_loss=None, epoch=0):
    # Path name 
    savefilename = args.savemodel + 'ckpt_' + str(epoch) + '.tar'
    
    torch.save({
          'epoch': epoch,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'train_loss': total_train_loss,
          'test_loss': total_test_loss,
      }, savefilename)
      

def main():
    max_acc = 100
    max_epo = 0
    tic_all = time.time()

    for epoch in range(1, args.epochs + 1):
        #adjust_lr(optimizer, epoch)
        total_train_loss = 0
        total_test_loss = 0
        total_feat_loss = 0
        
        print(">>>[INFO] Training Epoch %d" % (epoch))
        tic_epoch = time.time()
        
        ## Train ##
        for batch_idx, train_data in enumerate(TrainImgLoader):
            tic = time.time()
            loss_train = train(train_data, epoch)
            toc = time.time()
            print('>>>[INFO] Epoch: %d/%d  Iter: %d/%d  Total Loss: %.3f  Time: %.2f\n'
                  % (epoch, args.epochs, batch_idx, len(TrainImgLoader), loss_train, toc - tic))
            total_train_loss += loss_train
        
        # Average total losses 
        total_train_loss /= len(TrainImgLoader)
        
        print('>>>[INFO] Epoch: %d/%d   Total Training Loss: %.3f  Time/Epoch: %.2f hrs' %
              (epoch, args.epochs, total_train_loss, (time.time() - tic_epoch) / 3600))

        save_model(total_train_loss, epoch=epoch)
        
        if (epoch % args.interval) == 0:
            ## Test ##
            for batch_idx, test_data in enumerate(TestImgLoader):
                tic = time.time()
                test_loss = test(test_data)
                toc = time.time()
                print('>>>[INFO] Iter: %d/%d   Loss: %.3f time: %.2f' %(batch_idx + 1, len(TestImgLoader), test_loss, toc - tic))
    
                total_test_loss += test_loss
                
            # Average losses 
            total_test_loss /= len(TestImgLoader)
            
            print('>>>[INFO] Epoch: %d   Total Test Loss %.3f' %(epoch, total_test_loss))
    
            if total_test_loss < max_acc:
                max_acc = total_test_loss 
                max_epo = epoch
                
            print('>>>[INFO] BEST epoch: %d   Total Test Error = %.3f  Time/Epoch: %.2f hrs' % (
            max_epo, max_acc, (time.time() - tic_epoch) / 3600))
            print(" ")
        
            save_model(total_train_loss, total_test_loss, epoch=epoch)

    print('>>>[INFO] Total Time = %.2f HR' % ((time.time() - tic_all) / 3600))


if __name__ == '__main__':
    main()



