import torch
import torch.nn as nn
import torch.nn.functional as F
from models.outputs_container import OutputsContainer
from models.unet import UNet
from models.feature_fusion import SCAM  # 使用下文改进后的 SCAM

class Recovery(nn.Module):
    def __init__(self, hparams, *args, **kargs):
        super().__init__()
        self.preinverse = hparams.preinverse
        self.scale = hparams.scale
        depth_ch = 1
        color_ch = 3
        color = 3
        n_layers = 4
        n_depths = hparams.n_depths
        base_ch = 32  # hparams.model_base_ch
        preinv_input_ch = color * n_depths + color_ch

        base_input_layers = nn.Sequential(
            nn.Conv2d(preinv_input_ch+color, preinv_input_ch+color, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(preinv_input_ch+color),
            nn.ReLU(),
            nn.Conv2d(preinv_input_ch+color, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
        )
        base_input_layers2 = nn.Sequential(
            nn.Conv2d(preinv_input_ch+color, preinv_input_ch+color, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(preinv_input_ch+color),
            nn.ReLU(),
            nn.Conv2d(preinv_input_ch+color, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
        )
        input_layers = base_input_layers
        input_layers2 = base_input_layers2

        input_rough_layers = nn.Sequential(
            nn.Conv2d(depth_ch, base_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU()
        )
        output_layers = nn.Sequential(
            nn.Conv2d(base_ch, color, kernel_size=1, bias=True)
        )
        output_layers_2 = nn.Sequential(
            nn.Conv2d(base_ch, depth_ch, kernel_size=1, bias=True)
        )
        base_ch2 = 16  # 用于深度融合分支的特征通道数

        # ----------------- 新增深度信息融合分支 -----------------
        # 将 1 通道深度映射到 base_ch2 维度
        self.depth_feature_conv = nn.Sequential(
            nn.Conv2d(depth_ch, base_ch2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch2),
            nn.ReLU()
        )
        self.dfd_feature_conv = nn.Sequential(
            nn.Conv2d(depth_ch, base_ch2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch2),
            nn.ReLU()
        )
        # 使用改进后的 SCAM 模块进行特征融合
        self.scam_depth = SCAM(base_ch2)
        # 细化融合后的特征，生成最终深度图（1通道）
        self.depth_refine = nn.Sequential(
            nn.Conv2d(base_ch2, base_ch2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch2),
            nn.ReLU(),
            nn.Conv2d(base_ch2, depth_ch, kernel_size=1, bias=True)
        )
        # --------------------------------------------------------

        # 原始的解码器与其他分支保持不变
        self.input_layers = input_layers
        self.input_layers2 = input_layers2
        self.input_rough_layers = input_rough_layers
        self.output_layers = output_layers
        self.output_layers_2 = output_layers_2

        self.decoder = nn.Sequential(
            UNet(
                channels=[base_ch, base_ch, 2 * base_ch, 2 * base_ch, 4 * base_ch, 4 * base_ch, 4 * base_ch, 4 * base_ch],
                n_layers=n_layers,
            )
        )
        self.est_feature = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
            nn.Conv2d(base_ch, base_ch2, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_ch2),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            UNet(
                channels=[base_ch2, base_ch2, 2 * base_ch2, 2 * base_ch2],
                n_layers=2,
            ),
            nn.Sequential(
                nn.Conv2d(base_ch2, base_ch2, kernel_size=1, bias=False),
                nn.BatchNorm2d(base_ch2),
                nn.ReLU(),
                nn.Conv2d(base_ch2, depth_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(depth_ch),
                nn.ReLU()
            )
        )
        self.refine = nn.Sequential(
            UNet(
                channels=[2*base_ch2, 2*base_ch2, 2*base_ch2, 2*base_ch2],
                n_layers=2,
            ),
            nn.Sequential(
                nn.Conv2d(2*base_ch2, base_ch2, kernel_size=1, bias=False),
                nn.BatchNorm2d(base_ch2),
                nn.ReLU(),
                nn.Conv2d(base_ch2, depth_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(depth_ch),
                nn.ReLU()
            )
        )

        # 权重初始化
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
        captimgs_left = F.interpolate(captimgs_left, size=(int(h_sz*self.scale), int(w_sz*self.scale)),
                                      mode='bilinear', align_corners=False)
        captimgs_right = F.interpolate(captimgs_right, size=(int(h_sz*self.scale), int(w_sz*self.scale)),
                                       mode='bilinear', align_corners=False)
        rough_depth = rough_depth.reshape(b_sz, 1, int(h_sz*self.scale), int(w_sz*self.scale))
        pinv_volumes_left = F.interpolate(pinv_volumes_left, size=[pinv_volumes_left.shape[2],
                                                                    int(h_sz*self.scale),
                                                                    int(w_sz*self.scale)],
                                         mode='trilinear', align_corners=False)
        inputs = torch.cat([captimgs_left[:, :3, ...].unsqueeze(2), pinv_volumes_left], dim=2)
        inputs = torch.cat([inputs.reshape(b_sz, -1, int(h_sz*self.scale), int(w_sz*self.scale)), captimgs_right], dim=1)
        inputs = self.input_layers(inputs)
        
        # 得到解码器输出特征，并分别预测 RGB 图像和深度（DFD分支）
        est = self.decoder(inputs)
        est_images = torch.sigmoid(self.output_layers(est))
        est_depthmaps_dfd = torch.sigmoid(self.output_layers_2(est))
        
        # ------------------- 深度信息融合 -------------------
        # 分别将粗糙深度和 DFD 预测深度映射到特征空间
        rough_feat = self.depth_feature_conv(rough_depth)
        dfd_feat = self.dfd_feature_conv(est_depthmaps_dfd)
        # 利用 SCAM 模块进行交叉注意力融合，并强化高频信息
        fused_feat = self.scam_depth(rough_feat, dfd_feat)
        # 细化融合特征得到最终深度图（采用 sigmoid 限定输出范围）
        est_depthmaps = torch.sigmoid(self.depth_refine(fused_feat))
        # -----------------------------------------------------

        outputs = OutputsContainer(
            est_images=est_images,
            est_dfd=est_depthmaps_dfd,
            est_depthmaps=est_depthmaps
        )
        return outputs
