import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.submodule import *
import torch
from torchvision import transforms
import math
toPIL = transforms.ToPILImage() 
from models.feature_fusion import SCAM
from solvers.image_reconstruction import apply_tikhonov_inverse

class feature_extraction(nn.Module):
    def __init__(self,*args, **kargs):
        super(feature_extraction, self).__init__()
        self.inplanes = 16
        
        
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       convbn(32, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True))
        self.firstconv2 = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       convbn(32, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True))
        #self.firstconv_volume = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       #nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 32, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 64, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)
        
        #self.firstconv_right = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       #nn.ReLU(inplace=True))
        #self.layer1_right = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        #self.layer2_right = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        #self.layer3_right = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        #self.layer4_right = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)
        

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, captimgs_left, captimgs_right, hparams,  *args, **kargs):

        if hparams.position_code:
            
            if captimgs_left[0,3,0,0]==1:
                feature_left = self.firstconv(captimgs_left[:,:3,...])
                feature_right = self.firstconv(captimgs_right[:,:3,...])
            else:
                feature_left = self.firstconv2(captimgs_left[:,:3,...])
                feature_right = self.firstconv2(captimgs_right[:,:3,...])
        else:
            feature_left = self.firstconv(captimgs_left)
            feature_right = self.firstconv(captimgs_right)
        l2_left = self.layer2(feature_left)
        l3_left = self.layer3(l2_left)
        l4_left = self.layer4(l3_left)
        gwc_feature_left = torch.cat((l2_left, l3_left, l4_left), dim=1)

        
        l2_right = self.layer2(feature_right)
        l3_right = self.layer3(l2_right)
        l4_right = self.layer4(l3_right)
        gwc_feature_right = torch.cat((l2_right, l3_right, l4_right), dim=1)
        #feature_right = self.firstconv_right(captimgs_right)
        #l2_right = self.layer2_right(feature_right)
        #l3_right = self.layer3_right(l2_right)
        #l4_right = self.layer4_right(l3_right)
        #gwc_feature_right = torch.cat((l2_right, l3_right, l4_right), dim=1)
        return gwc_feature_left, gwc_feature_right
    
class hourglass2D(nn.Module):
    def __init__(self, in_channels):
        super(hourglass2D, self).__init__()

        self.expanse_ratio = 2

        self.conv1 = MobileV2_Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)

        self.conv2 = MobileV2_Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

        self.conv3 = MobileV2_Residual(in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)

        self.conv4 = MobileV2_Residual(in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels))

        self.redir1 = MobileV2_Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
        self.redir2 = MobileV2_Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6
class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        #self.attention_block = attention_block(channels_3d=in_channels * 4, num_heads=16, block=(4, 4, 4))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        #conv4 = self.attention_block(conv4)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6

class CVNet(nn.Module):
    
    def __init__(self,hparams, *args, **kargs):
        super(CVNet, self).__init__()
        self.maxdisp = hparams.maxdisp
        self.num_groups = 40
        self.concat_channels = 16
        self.feature_extraction = feature_extraction()
        self.concatconv = nn.Sequential(convbn(224, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, self.concat_channels, kernel_size=1, padding=0, stride=1,bias=False))



        self.dres0 = nn.Sequential(convbn_3d(self.concat_channels*2 , 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 
      
        self.dres2 = hourglass(32)
        self.dres3 = hourglass(32)
        self.classif0 = nn.Sequential(convbn_3d(32, 16, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 16, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))
        
        self.classif2 = nn.Sequential(convbn_3d(32, 16, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))
    
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


    def forward(self,captimgs_left, captimgs_right, captimgs_left_m, captimgs_right_m, hparams, *args, **kargs):
        
        self.hparams=hparams 
        
        features = self.feature_extraction(captimgs_left, captimgs_right, self.hparams)
        concat_feature_left = self.concatconv(features[0])
        concat_feature_right = self.concatconv(features[1])  
        #concat_feature_left, concat_feature_right=self.scam(concat_feature_left,concat_feature_right)
        concat_volume = build_concat_volume(concat_feature_left, concat_feature_right, self.maxdisp // 4)  

        features_mirror=self.feature_extraction(captimgs_left_m, captimgs_right_m, self.hparams)
        concat_feature_left_m = self.concatconv(features_mirror[0])
        concat_feature_right_m = self.concatconv(features_mirror[1])  
        concat_volume_m = build_concat_volume(concat_feature_left_m, concat_feature_right_m, self.maxdisp // 4)  

        Volume=[concat_volume, concat_volume_m]
        pred=[]
        for concat_volume in Volume:

            ac_volume = concat_volume
            cost0 = self.dres0(ac_volume)
            cost0 = self.dres1(cost0) + cost0
            out0 = self.classif0(cost0)
            out0 = F.upsample(out0, [self.maxdisp, captimgs_left.size()[2], captimgs_left.size()[3]], mode='trilinear')
            out0 = torch.squeeze(out0, 1)
            pred0 = F.softmax(out0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)

            out1 = self.dres2(cost0)
            cost1 = self.classif1(out1)
            cost1 = F.upsample(cost1, [self.maxdisp, captimgs_left.size()[2], captimgs_left.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp) 
            
            out2 = self.dres3(out1)
            cost2 = self.classif2(out2)    
            cost2 = self.classif2(out2)   
            cost2 = F.upsample(cost2, [self.maxdisp, captimgs_left.size()[2], captimgs_left.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            pred.append(0.4*pred0+0.6*pred1)
            pred.append(pred2)


        
        
        return pred #,pred2



def acv(d):
    return CVNet(d)