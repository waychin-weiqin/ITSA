from __future__ import print_function
import math
import torch.utils.data
from models.PSMNet.submodules import *

                    
class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class PSMNet(nn.Module):
    def __init__(self, maxdisp, eps=0.1, itsa=False):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.eps = eps
        self.itsa = itsa
        
        # Deterministic 
        self.feature_extraction = feature_extraction()

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
    def build_cv(self, featL, featR):
        cost = Variable(
            torch.FloatTensor(featL.size()[0], featL.size()[1] * 2, (self.maxdisp // 4), featL.size()[2],
                              featL.size()[3]).zero_()).cuda()

        for i in range(self.maxdisp // 4):
            if i > 0:
                cost[:, :featL.size()[1], i, :, i:] = featL[:, :, :, i:]
                cost[:, featL.size()[1]:, i, :, i:] = featR[:, :, :, :-i]
            else:
                cost[:, :featL.size()[1], i, :, :] = featL
                cost[:, featL.size()[1]:, i, :, :] = featR

        cost = cost.contiguous()
        
        return cost
        
    def disparityregression(self, cost):
        disp = torch.arange(0, self.maxdisp).cuda()
        disp = disp.view(1, -1, 1, 1).to(torch.float32)
        
        out = torch.sum(cost * disp, 1)
        
        return out
        
    
    def cost_regularization(self, cost):
    
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        cost3 = F.interpolate(cost3, scale_factor=4, mode='trilinear', align_corners=True)
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        pred3 = self.disparityregression(pred3)
        
        if self.training:

            cost1 = F.interpolate(cost1, scale_factor=4, mode='trilinear', align_corners=True)
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = self.disparityregression(pred1)
            
            cost2 = F.interpolate(cost2, scale_factor=4, mode='trilinear', align_corners=True)
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = self.disparityregression(pred2)

            return [pred1, pred2, pred3]

        else:

            return pred3
            
        
    def clip(self, img, img_min=None, img_max=None):
        if img_min is None:
            img_min = torch.tensor([-2.1179, -2.0357, -1.8044]).view(1,3,1,1).cuda()

        if img_max is None:
            img_max = torch.tensor([2.2489, 2.4286, 2.6400]).view(3,1,1).cuda()

        img = torch.clip(img, min=img_min, max=img_max)
        
        return img
        
    def grad_norm(self, grad):
        grad = grad.pow(2)
        grad = F.normalize(grad, p=2, dim=1) 
        grad = grad * self.eps
          
        return grad 

    def forward(self, imgL, imgR):
        if self.itsa & self.training:
            #=================================================#
            # SCP Augmentation 
            imgL_ = imgL.clone().detach()
            imgL_.requires_grad = True 
            
            imgR_ = imgR.clone().detach() 
            imgR_.requires_grad = True 
            
            self.eval()
             
            featL_ = self.feature_extraction(imgL_)
            gradL = torch.autograd.grad(outputs=featL_, inputs=imgL_, grad_outputs=torch.ones_like(featL_), create_graph=False)
            gradL = gradL[0].clone().detach()  # B,C,H,W
            
            featR_ = self.feature_extraction(imgR_)
            gradR = torch.autograd.grad(outputs=featR_, inputs=imgR_, grad_outputs=torch.ones_like(featR_), create_graph=False)
            gradR = gradR[0].clone().detach()  # B,C,H,W
                    
            gradL = self.grad_norm(gradL)
            gradR = self.grad_norm(gradR)
            
            imgL_scp = imgL.clone().detach() + gradL
            imgR_scp = imgR.clone().detach() + gradR
            
            imgL_scp = self.clip(imgL_scp).detach()
            imgR_scp = self.clip(imgR_scp).detach()
            
            del imgL_, imgR_
            
            self.train()
            # Forward Pass
            featL_scp = self.feature_extraction(imgL_scp)
            featR_scp = self.feature_extraction(imgR_scp)
            
            featL = self.feature_extraction(imgL)
            featR = self.feature_extraction(imgR) 
               
            cost = self.build_cv(featL_scp, featR_scp)  
            dispEsts = self.cost_regularization(cost)
            
            featEsts = {"left": featL,
                        "right": featR,
                        "left_scp": featL_scp,
                        "right_scp": featR_scp}
            
            return featEsts, dispEsts
            
            
        else:
            # Forward Pass 
            featL = self.feature_extraction(imgL)
            featR = self.feature_extraction(imgR)    
            cost = self.build_cv(featL, featR)  
            dispEsts = self.cost_regularization(cost)      
          
            return dispEsts
        
    
              
    
        
