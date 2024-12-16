import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim


def mae(image1, image2):
        MAE=F.l1_loss(image1, image2)
        return MAE.item()
def mse(image1, image2):
        MSE=F.mse_loss(image1, image2)
        return MSE.item()
def rmse(image1, image2):
        RMSE=torch.sqrt(F.mse_loss(image1, image2))
        return RMSE.item()
def calculate_psnr(image1, image2):
        MSE = F.mse_loss(image1, image2)
        PSNR = 10 * torch.log10(1 / MSE)
        return PSNR.item()

def calculate_ssim(img1, img2):
    
        img1=img1*255
        img2=img2*255
        ssim_value=ssim(img1,img2)
        return ssim_value.item()
            
def calculate_3px(depth1,depth2,threshold=3.0):
        abs_diff=torch.abs(depth1-depth2)
        error_mask=abs_diff>threshold
        error=error_mask.float().mean()*100
        return error