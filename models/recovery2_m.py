import torch
import torch.nn as nn
import torch.nn.functional as F
from models.outputs_container import OutputsContainer
#from models.unet_y import UNet
from models.unet import UNet
from models.feature_fusion import SCAM


class Recovery(nn.Module):

    def __init__(self, hparams, *args, **kargs):
        super().__init__()
        self.preinverse = hparams.preinverse
        self.scale = hparams.scale
        depth_ch = 1
        color_ch = 3
        color=3
        n_layers = 4
        n_depths = hparams.n_depths
        base_ch = 32 #hparams.model_base_ch
        preinv_input_ch = color * n_depths + color_ch
        #self.scam=SCAM(c=base_ch)
        #self.scam1=SCAM(c=16)
        #self.scam0=SCAM(c=1)
        base_input_layers = nn.Sequential(
            nn.Conv2d(preinv_input_ch+color, preinv_input_ch+color, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(preinv_input_ch+color),
            nn.ReLU(),
            nn.Conv2d(preinv_input_ch+color, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
            #nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(base_ch),
            #nn.ReLU(),
        )
        base_input_layers2 = nn.Sequential(
            nn.Conv2d(preinv_input_ch+color, preinv_input_ch+color, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(preinv_input_ch+color),
            nn.ReLU(),
            nn.Conv2d(preinv_input_ch+color, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
            #nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(base_ch),
            #nn.ReLU(),
        )
       
        # Without the preinverse input, it has ((color_ch * preinv_input_ch) + preinv_input_ch * 2) more parameters than
        # with the preinverse input. (255 params)
        
        input_layers = base_input_layers
        input_layers2= base_input_layers2
        '''else:
            input_layers = nn.Sequential(
                    nn.Conv2d(color_ch+color, preinv_input_ch+color, kernel_size=1, bias=False),
                    nn.BatchNorm2d(preinv_input_ch+color),
                    nn.ReLU(),
                    base_input_layers)
            input_layers2 = nn.Sequential(
                    nn.Conv2d(color_ch+color, preinv_input_ch+color, kernel_size=1, bias=False),
                    nn.BatchNorm2d(preinv_input_ch+color),
                    nn.ReLU(),
                    base_input_layers2)'''

        input_rough_layers = nn.Sequential(
            nn.Conv2d(depth_ch, base_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU())
        output_layers = nn.Sequential(
            nn.Conv2d(base_ch, color, kernel_size=1, bias=True)
        )
        output_layers_2 = nn.Sequential(
            nn.Conv2d(base_ch, depth_ch, kernel_size=1, bias=True)
        )
        base_ch2 = 16
        outdepth_layers = nn.Sequential(
                    nn.Conv2d(depth_ch+color, base_ch2, kernel_size=1, bias=False),
                    nn.BatchNorm2d(base_ch2),
                    nn.ReLU(),
                    nn.Conv2d(base_ch2, base_ch2, kernel_size=1, bias=False),
                    nn.BatchNorm2d(base_ch2),
                    nn.ReLU())

        refine_depth_layers = nn.Sequential(
                    nn.Conv2d(depth_ch, base_ch2, kernel_size=1, bias=False),
                    nn.BatchNorm2d(base_ch2),
                    nn.ReLU(),
                    nn.Conv2d(base_ch2, base_ch2, kernel_size=1, bias=False),
                    nn.BatchNorm2d(base_ch2),
                    nn.ReLU())
        refine_rgb_layers = nn.Sequential(
                    nn.Conv2d(color, base_ch2, kernel_size=1, bias=False),
                    nn.BatchNorm2d(base_ch2),
                    nn.ReLU(),
                    nn.Conv2d(base_ch2, base_ch2, kernel_size=1, bias=False),
                    nn.BatchNorm2d(base_ch2),
                    nn.ReLU())

        outdepth_layers2 = nn.Sequential(
                    nn.Conv2d(base_ch2, base_ch2, kernel_size=1, bias=False),
                    nn.BatchNorm2d(base_ch2),
                    nn.ReLU(),
                    nn.Conv2d(base_ch2, depth_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(depth_ch),
                    nn.ReLU())

        out_refine = nn.Sequential(
                    nn.Conv2d(2*base_ch2, base_ch2, kernel_size=1, bias=False),
                    nn.BatchNorm2d(base_ch2),
                    nn.ReLU(),
                    nn.Conv2d(base_ch2, depth_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(depth_ch),
                    nn.ReLU())

        self.input_layers=input_layers
        self.input_layers2=input_layers2
        self.input_rough_layers=input_rough_layers
        self.outdepth_layers=outdepth_layers
        self.outdepth_layers2=outdepth_layers2
        self.refine_depth_layers=refine_depth_layers
        self.refine_rgb_layers=refine_rgb_layers

        self.decoder = nn.Sequential(
            UNet(
                channels=[base_ch, base_ch, 2 * base_ch, 2 * base_ch, 4 * base_ch, 4 * base_ch, 4 * base_ch, 4 * base_ch], #, 4 * base_ch, 4 * base_ch],  #, 8 * base_ch, 8 * base_ch],
                #channels=[base_ch*2, base_ch*2, 4 * base_ch, 4 * base_ch, 4 * base_ch, 4 * base_ch, 8 * base_ch, 8 * base_ch, 8 * base_ch, 8 * base_ch],
                n_layers=n_layers,
            )
        )
        self.est_feature=nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
            nn.Conv2d(base_ch, base_ch2, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_ch2),
            nn.ReLU()
        )

        self.decoder2 = nn.Sequential(
            UNet(
                channels=[base_ch2, base_ch2, 2 * base_ch2, 2 * base_ch2],  #, 8 * base_ch, 8 * base_ch],
                #channels=[base_ch*2, base_ch*2, 4 * base_ch, 4 * base_ch, 4 * base_ch, 4 * base_ch, 8 * base_ch, 8 * base_ch, 8 * base_ch, 8 * base_ch],
                n_layers=2,
            ),
            self.outdepth_layers2
        )
        self.refine = nn.Sequential(
            UNet(
                channels=[2*base_ch2, 2*base_ch2, 2*base_ch2, 2*base_ch2],  #, 8 * base_ch, 8 * base_ch],
                #channels=[base_ch*2, base_ch*2, 4 * base_ch, 4 * base_ch, 4 * base_ch, 4 * base_ch, 8 * base_ch, 8 * base_ch, 8 * base_ch, 8 * base_ch],
                n_layers=2,
            ),
            out_refine
        )
        self.output_layers=output_layers
        self.output_layers_2=output_layers_2
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, captimgs_left, pinv_volumes_left, captimgs_right, rough_depth, *args, **kargs):
        
        b_sz, _, h_sz, w_sz = captimgs_left.shape
        captimgs_left=F.interpolate(captimgs_left, size=(int(h_sz*self.scale),int(w_sz*self.scale)), mode='bilinear', align_corners=False)
        captimgs_right=F.interpolate(captimgs_right, size=(int(h_sz*self.scale),int(w_sz*self.scale)), mode='bilinear', align_corners=False)
        rough_depth = rough_depth.reshape(b_sz, 1, int(h_sz*self.scale),int(w_sz*self.scale))
        #rough_depth=F.interpolate(rough_depth.reshape(b_sz, -1, h_sz, w_sz), size=(int(h_sz*self.scale),int(w_sz*self.scale)), mode='bilinear', align_corners=False)
        pinv_volumes_left=F.interpolate(pinv_volumes_left, size=[pinv_volumes_left.shape[2],int(h_sz*self.scale),int(w_sz*self.scale)], mode='trilinear', align_corners=False)
        inputs = torch.cat([captimgs_left[:,:3,...].unsqueeze(2), pinv_volumes_left], dim=2)
        inputs= torch.cat([inputs.reshape(b_sz, -1, int(h_sz*self.scale), int(w_sz*self.scale)),captimgs_right], dim=1)
        inputs=self.input_layers(inputs.reshape(b_sz, -1, int(h_sz*self.scale), int(w_sz*self.scale)))
        
        est= self.decoder(inputs)
        est_images= torch.sigmoid(self.output_layers(est))
        est_depthmaps_dfd= torch.sigmoid(self.output_layers_2(est))    
        est_disp=rough_depth
        
        depth_feature=self.refine_depth_layers(rough_depth)
        est= self.est_feature(est)
        est_depthmaps=self.refine(torch.cat([depth_feature, est], dim=1))

        outputs = OutputsContainer(
            est_images=est_images,
            est_dfd=est_depthmaps,
            est_depthmaps=est_disp)

        return outputs