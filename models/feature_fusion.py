import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=[0, 2, 3]), grad_output.sum(dim=[0, 2, 3]), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SCAM(nn.Module):
    '''
    Stereo Cross Attention Module (SCAM) —— 改进版
    除了原有的交叉注意力分支外，增加了高频信息提取模块以强化细节。
    '''
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.l_proj3 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj3 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        # 新增：高频信息提取卷积，将输入特征（高频部分）映射到相同维度
        self.hf_conv = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1)

    def forward(self, x_l, x_r):
        # 交叉注意力计算
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # (B, H, W, c)
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3) # (B, H, c, W)
        V_r = self.r_proj3(x_r).permute(0, 2, 3, 1)                # (B, H, W, c)

        attention = torch.matmul(Q_l, Q_r_T) * self.scale          # (B, H, W, W)
        attn_weights = torch.softmax(attention, dim=-1)
        F_r2l = torch.matmul(attn_weights, V_r)                    # (B, H, W, c)
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta

        # 高频信息提取
        # 通过平均池化得到低频部分，再做差得到高频成分
        x_r_blur = F.avg_pool2d(x_r, kernel_size=3, stride=1, padding=1)
        hf = x_r - x_r_blur
        hf = self.hf_conv(hf)  # 映射到相同通道

        # 融合原始 x_l、交叉注意力特征和高频信息（后者由参数 gamma 调控贡献）
        out = x_l + F_r2l + self.gamma * hf
        return out
