import abc
import math
from typing import List, Union
from ls_asm.input_field import InputField
from ls_asm.LSASM import LeastSamplingASM
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
#from ls_asm import PropagationModel
from util import complex, cubicspline
from util.fft import fftshift
from util.helper import copy_quadruple, depthmap_to_layereddepth, ips_to_metric, over_op, \
    refractive_index
from psf.psf_import import *
try:
    from torch import irfft
    from torch import rfft
except ImportError:
    from torch.fft import irfft
    from torch.fft import rfft
    def rfft(x, d):
        t=torch.fft.fft2(x, dim = (-d,-1))
        return torch.stack((t.real, t.imag), -1)
    def irfft(x, d, signal_sizes):
        t = torch.fft.ifft2(torch.complex(x[...,0], x[...,1]), dim = (-d,-1))
        return t.real
import cv2
class BaseCamera(nn.Module, metaclass=abc.ABCMeta):


    def __init__(self, focal_depth, min_depth, max_depth, n_depths, image_size, mask_size, mask_pitch,
                 focal_length, mask_diameter, camera_pixel_pitch, wavelengths, full_size=1920, mask_upsample_factor=1,
                 diffraction_efficiency=0.7, require_grads=False, **kwargs):
        super().__init__()
        assert min_depth > 1e-6, f'Minimum depth is too small. min_depth: {min_depth}'
        scene_distances = ips_to_metric(torch.linspace(0, 1, steps=n_depths), min_depth, max_depth)
        scene_distances = scene_distances.flip(-1)
        
        self.modulate_phase=require_grads
        self._register_wavlength(wavelengths)
        self.n_depths = n_depths
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.focal_depth = focal_depth
        self.mask_diameter = mask_diameter
        self.camera_pixel_pitch = camera_pixel_pitch
        self.focal_length = focal_length
        self.f_number = self.focal_length / self.mask_diameter
        self.image_size = self._normalize_image_size(image_size)
        self.mask_pitch = mask_pitch#self.mask_diameter / mask_size
        self.mask_size = mask_size
        self.full_size = full_size
        self.device = 'cuda'
        self.register_buffer('scene_distances', scene_distances)
        self.experiment=False
        

    def _register_wavlength(self, wavelengths):
        if isinstance(wavelengths, list):
            wavelengths = torch.tensor(wavelengths)  # in [meter]
        elif isinstance(wavelengths, float):
            wavelengths = torch.tensor([wavelengths])
        else:
            raise ValueError('wavelengths has to be a float or a list of floats.')

        if len(wavelengths) % 3 != 0:
            raise ValueError('the number of wavelengths has to be a multiple of 3.')

        self.n_wl = len(wavelengths)
        if not hasattr(self, 'wavelengths'):
            self.register_buffer('wavelengths', wavelengths)
        else:
            self.wavelengths = wavelengths.to(self.wavelengths.device)


    def sensor_distance(self):
        return 1. / (1. / self.focal_length - 1. / self.focal_depth)

    def normalize_psf(self, psfimg):
        # Scale the psf
        # As the incoming light doesn't change, we compute the PSF energy without the phase modulation
        # and use it to normalize PSF with phase modulation.
        return psfimg / psfimg.sum(dim=(-2, -1), keepdims=True)

    def _capture_impl(self, volume, layered_depth, psf, occlusion, eps=1e-3):
        scale = volume.max()
        volume = volume / scale
        Fpsf = rfft(psf, 2)
        if occlusion:
            Fvolume = rfft(volume, 2)
            Flayered_depth = rfft(layered_depth, 2)
            blurred_alpha_rgb = irfft(
                complex.multiply(Flayered_depth, Fpsf), 2, signal_sizes=volume.shape[-2:])

            blurred_volume = irfft(
                complex.multiply(Fvolume, Fpsf), 2, signal_sizes=volume.shape[-2:])
         
            # Normalize the blurred intensity
            cumsum_alpha = torch.flip(torch.cumsum(torch.flip(layered_depth, dims=(-3,)), dim=-3), dims=(-3,))
            Fcumsum_alpha = rfft(cumsum_alpha, 2)
            blurred_cumsum_alpha = irfft(
                complex.multiply(Fcumsum_alpha, Fpsf), 2, signal_sizes=volume.shape[-2:])
            
            blurred_volume = blurred_volume / (blurred_cumsum_alpha + eps)
            blurred_alpha_rgb = blurred_alpha_rgb / (blurred_cumsum_alpha + eps)
            over_alpha = over_op(blurred_alpha_rgb)


            captimg = torch.sum(over_alpha * blurred_volume, dim=2)
            
        else:
            Fvolume = rfft(volume, 2)
            Fcaptimg = complex.multiply(Fvolume, Fpsf).sum(dim=2)
            captimg = irfft(Fcaptimg, 2, signal_sizes=volume.shape[-2:0])
        
        #[1,3,5,160,160], [1,3,5,160,160,2]
        #pp=psf[0,:,0,...]/psf[0,:,0,...].max()
        #torchvision.utils.save_image(pp, 'psf.png')
        #torchvision.utils.save_image(Fpsf[0,:,0,...,0], 'Fpsf.png')
        #[1,3,5,160,160], [1,3,5,160,160,2]
        #pp=psf[0,:,-2,...]/psf[0,:,-2,...].max()
        #torchvision.utils.save_image(fftshift(pp,dims=(-1, -2)), 'psf_l.png')
        #Fpp=fftshift(rfft(pp, 2),dims=(-2, -3))
        #torchvision.utils.save_image(Fpp[...,0], 'Fpsf_l.png')
        
        
        captimg = scale * captimg
        volume = scale * volume
        return captimg, volume

    def _capture_from_rgbd_with_psf_impl(self, img, depthmap, psf, occlusion):
       # psf = F.interpolate(psf, size=(self.n_depths, psf.shape[-2], psf.shape[-1]), mode='trilinear', align_corners=False)
        layered_depth = depthmap_to_layereddepth(depthmap, self.n_depths, binary=True)
        volume = layered_depth * img[:, :, None, ...]
        return self._capture_impl(volume, layered_depth, psf, occlusion)

    def capture_from_rgbd(self, img, depthmap, occlusion):
        psf = self.psf_at_camera(img.shape[-2:], self.modulate_phase)  # add batch dimension
        psf=fftshift(self.normalize_psf(psf),dims=(-1,-2))
        return self.capture_from_rgbd_with_psf(img, depthmap, psf, occlusion)

    def capture_from_rgbd_with_psf(self, img, depthmap, psf, occlusion):
        return self._capture_from_rgbd_with_psf_impl(img, depthmap,psf, occlusion)[0]
        

    @abc.abstractmethod
    def psf_at_camera(self, size, modulate_phase, is_training=torch.tensor(False)):
        pass

    @abc.abstractmethod
    def height(self):
        pass

    def forward(self, img, depthmap, occlusion, modulate_phase, is_training=torch.tensor(False)):
        """
        Args:
            img: B x C x H x W

        Returns:
            captured image: B x C x H x W
        """
        psf = self.psf_at_camera(img.shape[-2:], modulate_phase, is_training=is_training).unsqueeze(0)  # add batch dimension
        
        psf = fftshift(self.normalize_psf(psf),dims=(-1,-2))
        #psf = F.interpolate(psf, size=(self.n_depths, psf.shape[-2], psf.shape[-1]), mode='trilinear', align_corners=False)
        captimg, volume = self._capture_from_rgbd_with_psf_impl(img, depthmap, psf, occlusion)
        return captimg, volume, psf

    def _normalize_image_size(self, image_size):
        if isinstance(image_size, int):
            image_size = [image_size, image_size]
        elif isinstance(image_size, list):
            if image_size[0] % 2 == 1 or image_size[1] % 2 == 1:
                raise ValueError('Image size has to be even.')
        else:
            raise ValueError('image_size has to be int or list of int.')
        return image_size

    def set_image_size(self, image_size):
        image_size = self._normalize_image_size(image_size)
        self.image_size = image_size

    def set_wavelengths(self, wavelengths):
        self._register_wavlength(wavelengths)

    def set_n_depths(self, n_depths):
        self.n_depths = n_depths

    def Propagation(self, scene_distances, modulate_phase):
        r = self.focal_length/self.f_number/2
        self.k = 2 * torch.pi / self.wavelengths
        self.modulate_phase = modulate_phase
        self.s_LSASM = 1
        self.thetaX = 0
        self.thetaY = 0
        wavelength=self.wavelengths.reshape(-1).double()
        zf = 1 / (1 / self.focal_length - 1 / self.focal_depth)
        Uin = InputField("12", wavelength, r,  scene_distances, self.focal_length, zf, self.mask_size, torch.tensor(False), None)
        E0=Uin.E0
        
        return Uin, E0

    
    def extra_repr(self):
        msg = f'Right Camera module...\n' \
              f'Refcative index for center wavelength: {refractive_index(self.wavelengths[self.n_wl // 2])} \n' \
              f'Mask pitch: {self.mask_pitch }[m] \n' \
              f'f number: {self.f_number} \n' \
              f'mask diameter: {self.mask_diameter}[m] \n' \
              f'Depths: {self.scene_distances} \n' \
              f'Input image size: {self.image_size} \n'
        return msg


class MixedCamera(BaseCamera):
    def __init__(self, focal_depth: float, min_depth: float, max_depth: float, n_depths: int,
                 image_size: Union[int, List[int]], mask_size: int, mask_pitch: float, focal_length: float, mask_diameter: float,
                 camera_pixel_pitch: float, wavelengths=torch.tensor([632e-9, 550e-9, 450e-9]), full_size=100, mask_upsample_factor=1,
                 diffraction_efficiency=0.7,  requires_grad: bool = True):
        self.diffraction_efficiency = diffraction_efficiency
        super().__init__(focal_depth, min_depth, max_depth, n_depths, image_size, mask_size, mask_pitch, focal_length,
                         mask_diameter, camera_pixel_pitch, wavelengths, full_size, mask_upsample_factor,
                         requires_grad)
        self.full_size = full_size
        # Rank 1 Initialization
        n_param= mask_size//2 // mask_upsample_factor
        y=torch.arange(0,n_param,1).cuda()
        
        n= torch.tensor(n_param).cuda()
        init_heightmap1d_x_0 = 4*torch.ones(n_param).cuda().float()
        init_heightmap1d_x_1 = 4*(1-(torch.square(y))/torch.square(n)).cuda().float()
        init_heightmap1d_y_0 = 2*(1-(torch.square(y))/torch.square(n)).cuda().float()
        init_heightmap1d_y_1 = 4*torch.ones(n_param).cuda().float()

        

        init_heightmap1d_x_0=torch.load('doe_vector/x_0_l.pth')
        init_heightmap1d_y_0=torch.load('doe_vector/y_0_l.pth')
        init_heightmap1d_x_1=torch.load('doe_vector/x_1_l.pth')
        init_heightmap1d_y_1=torch.load('doe_vector/y_1_l.pth')
        
        

        self.heightmap1d_x_0 = torch.nn.Parameter(init_heightmap1d_x_0, requires_grad=requires_grad)
        self.heightmap1d_y_0= torch.nn.Parameter(init_heightmap1d_y_0, requires_grad=requires_grad)
        self.heightmap1d_x_1 = torch.nn.Parameter(init_heightmap1d_x_1, requires_grad=requires_grad)
        self.heightmap1d_y_1 = torch.nn.Parameter(init_heightmap1d_y_1, requires_grad=requires_grad)

        self.h_max=0.55/0.5625*1e-6#1.1*1e-6 #0.55/0.5626*1e-6
        
        self.mask_upsample_factor = mask_upsample_factor
        self.modulate_phase = requires_grad
        self.Uin, self.E0=self.Propagation(self.scene_distances, self.modulate_phase)
        self.wvls = self.wavelengths  # wavelength of light in vacuum
        self.k = 2 * torch.pi / self.wavelengths 
        self.s = 1.0 #1.5
        self.zf = 1 / (1 / self.focal_length - 1 / self.focal_depth)#Uin.zf
        self.D = self.focal_length/self.f_number
        self.pupil = self.Uin.pupil

        self.fcX, self.fcY = self.Uin.fcX, self.Uin.fcY
        self.fbX, self.fbY = self.Uin.fbX, self.Uin.fbY
        self.xi, self.eta = self.Uin.xi, self.Uin.eta
        self.xi_, self.eta_ = self.Uin.xi_, self.Uin.eta_

    # Rank 1 Coding
    def heightmap1d_x(self):
        
        return F.interpolate(self.heightmap1d_x_0.reshape(1, 1, -1),
                             scale_factor=self.mask_upsample_factor, mode='nearest').reshape(-1)
    def heightmap1d_y(self):
        return F.interpolate(self.heightmap1d_y_0.reshape(1, 1, -1),
                             scale_factor=self.mask_upsample_factor, mode='nearest').reshape(-1)
    def heightmap1d_x1(self):
        return F.interpolate(self.heightmap1d_x_1.reshape(1, 1, -1),
                             scale_factor=self.mask_upsample_factor, mode='nearest').reshape(-1)
    def heightmap1d_y1(self):
        return F.interpolate(self.heightmap1d_y_1.reshape(1, 1, -1),
                             scale_factor=self.mask_upsample_factor, mode='nearest').reshape(-1)

    
    # Rank 1 Coding
    def heightmap2d(self):
        return torch.matmul(self.heightmap1d_y().reshape(-1,1),self.heightmap1d_x().reshape(1,-1))+torch.matmul(self.heightmap1d_y1().reshape(-1,1),self.heightmap1d_x1().reshape(1,-1))

    def forward_train(self, img, depthmap, occlusion):
        return self.forward(img, depthmap, occlusion)
    
    def rot_matric(self,theta):
        theta = torch.tensor(theta).cuda()
        #return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                            #[torch.sin(theta), torch.cos(theta), 0]])*torch.sqrt(torch.tensor(1.9))
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                            [torch.sin(theta), torch.cos(theta), 0]])
    
    def rot_height(self, x, theta):
        rot_mat = self.rot_matric(theta).repeat(x.shape[0],1,1)
        grid = F.affine_grid(rot_mat, x.size()).cuda()
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border')
        return x

    def aperture(self):
        x = torch.arange(self.mask_size).float().cuda()
        y = torch.arange(self.mask_size).float().cuda()
        X, Y = torch.meshgrid(x, y)
        r = torch.sqrt((X+0.5 - self.mask_size//2) ** 2 + (Y+0.5 - self.mask_size//2) ** 2)
        aperture=torch.where(r<self.mask_size//2, 1, 0)
        return aperture
    def height(self):
        heightmap2d=self.heightmap2d()
        heightmap2d=copy_quadruple(heightmap2d.unsqueeze(0).unsqueeze(0))
        heightmap2d=self.rot_height(heightmap2d, (torch.pi/4))
        height_map=heightmap2d.squeeze(0).squeeze(0)
        height_map=torch.remainder(height_map, 1)
        height_map=height_map*self.aperture() 
        return height_map.to(self.device)

    def phase(self):
        
        heightmap=torch.remainder(self.height(),1)
        k = 2 * torch.pi / self.wavelengths.reshape(-1,1,1,1)
        phase = heightmap*k *self.h_max * (refractive_index(self.wavelengths.reshape(-1,1,1,1)) - 1)
        phase=torch.remainder(phase, 2*torch.pi)
        return phase

    def through_plate(self, Ein, heightmap):
        
        heightmap=heightmap.unsqueeze(0).unsqueeze(0)
        #1.125*1e-6
        k = 2 * torch.pi / self.wavelengths.reshape(-1,1,1,1)
        phase = heightmap*k *self.h_max * (refractive_index(self.wavelengths.reshape(-1,1,1,1)) - 1)
        return Ein*torch.exp(1j * phase)

    def psf_obs(self, Ein):
        '''r = self.focal_length/self.f_number/2
        Mx, My = [1000,1000]#self.mask_size, self.mask_size
        l=r*0.5'''
        Mx, My = [256,256]
        l=5.86e-6*Mx
        z=1 / (1 / self.focal_length - 1 / self.focal_depth)
        x = torch.linspace(-l / 2 , l / 2 , Mx)
        y = torch.linspace(-l / 2 , l / 2 , My)    
        device = self.device
        prop2 = LeastSamplingASM(self, x, y, z, device)
        U2 = prop2(Ein).cuda()
        self.psf_phase=torch.remainder(torch.angle(U2), 2*torch.pi)
        return abs(U2)**2
    def psf_ph(self):
        return self.psf_phase
    def psf_obs_full(self, Ein):
        #r = self.focal_length/self.f_number/2
        #l = r * 0.25
        Mx, My = self.full_size, self.full_size#[50,50] #self.full_size, self.full_size #self.mask_size, self.mask_size
        l=5.86e-6*Mx
        #xc=0; yc=0
        z=1 / (1 / self.focal_length - 1 / self.focal_depth)
        x = torch.linspace(-l / 2 , l / 2 , Mx)
        y = torch.linspace(-l / 2 , l / 2 , My)    
        device = self.device
        prop2 = LeastSamplingASM(self, x, y, z, device)
        U2 = prop2(Ein).cuda()
        return abs(U2)**2
    
    def psf_full(self, modulate_phase):
        if modulate_phase: 
            Ein = self.through_plate(self.E0, self.height())
        if not modulate_phase:
            Ein = self.E0
        psf_full = F.relu(self.psf_obs_full(Ein))
        return psf_full


    def psf_at_camera(self, size, modulate_phase, is_training=torch.tensor(False)):
        device = self.device
        if not self.experiment:
            heightmap=self.height()#.to(device)
            if is_training:
                scene_distances = ips_to_metric(
                    torch.linspace(0, 1, steps=self.n_depths, device=device) +
                    1 / self.n_depths * (torch.rand(self.n_depths, device=device) - 0.5),
                    self.min_depth, self.max_depth)
                
                scene_distances[-1] += torch.rand(1, device=device)[0] * (100.0 - self.max_depth)
            else:
                scene_distances = ips_to_metric(torch.linspace(0, 1, steps=self.n_depths, device=device),
                                                self.min_depth, self.max_depth)
            #if modulate_phase: 
                #Ein = self.through_plate(self.E0, heightmap)
            #if not modulate_phase:
                #Ein = self.E0
            Ein = self.through_plate(self.E0, heightmap)

            diffracted_psf = F.relu(self.psf_obs(Ein))
            undiffracted_psf=F.relu(self.psf_obs(self.E0))
            self.diff_normalization_scaler = torch.tensor(diffracted_psf.sum(dim=(-1, -2), keepdim=True))
            self.undiff_normalization_scaler = undiffracted_psf.sum(dim=(-1, -2), keepdim=True)

            
            diffracted_psf = diffracted_psf / self.diff_normalization_scaler
            undiffracted_psf = undiffracted_psf / self.undiff_normalization_scaler

            psf = self.diffraction_efficiency * diffracted_psf + (1 - self.diffraction_efficiency) * undiffracted_psf
        if self.experiment:
            psf= psf_captured(device)[0].squeeze(0).double()
        # In training, randomly pixel-shifts the PSF around green channel.
        if is_training:
            psf=transforms.RandomRotation(1)(psf)
            max_shift = 2
            r_shift = tuple(torch.randint(-max_shift,max_shift, (2,)))
            b_shift = tuple(torch.randint(-max_shift,max_shift, (2,)))
            psf_r = torch.roll(psf[0], shifts=r_shift, dims=(-1, -2))
            psf_g = psf[1]
            psf_b = torch.roll(psf[2], shifts=b_shift, dims=(-1, -2))
            psf = torch.stack([psf_r, psf_g, psf_b], dim=0)
        #Blur the PSF
        kernel_size=5
        c,d,h,w=psf.shape
        kernel=torch.tensor(cv2.getGaussianKernel(kernel_size, 0.5)).cuda()
        kernel_2d=(kernel*kernel.T)
        for i in range(c):
                for j in range(d):
                    psf[[i],[j],...]=F.conv2d(psf[[i],[j],...], kernel_2d.expand(1,1,kernel_size,kernel_size), padding=((kernel_size-1)//2,(kernel_size-1)//2))
        
        psf=transforms.CenterCrop(size)(psf)
        #psf = F.pad(psf, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        #psf= fftshift(psf, dims=(-1, -2))
        #psf = F.interpolate(psf.unsqueeze(0), size=(self.n_depths, psf.shape[-2], psf.shape[-1]), mode='trilinear', align_corners=False)
        return psf.squeeze(0)


    def psf_out_of_fov_energy(self, psf_size: int):
        # As this quadruple will be copied to the other three, rho = 0 is avoided.
        device = 'cuda' #if torch.cuda.is_available() else 'cpu'
        
        psf_diffracted = self.psf_full(self.modulate_phase)

        try: 
            psf_diffracted = psf_diffracted / (self.diff_normalization_scaler)           
        except:
            psf_diffracted = psf_diffracted 
       
        # Cross Mask
        mask_c= torch.ones_like(psf_diffracted)
        center= mask_c.shape[-1]//2

        
        #mask = torch.zeros(100, 100)
        x = torch.arange(2*center).float().cuda()
        y = torch.arange(2*center).float().cuda()

        X, Y = torch.meshgrid(x, y)
        X=X.cuda()
        Y=Y.cuda()
        dist= torch.sqrt((X+0.5 - center) ** 2 + (Y+0.5 - center) ** 2).cuda()
        dist2=torch.where(dist>10, 1, 0).cuda()
        mask_c[..., :, :]=dist2 
        # Regularization for Rank 1 and Rank 2
        psf_out_of_fov = (psf_diffracted * mask_c).float()
        
        return psf_out_of_fov.sum()/10

    def forward_train(self, img, depthmap, occlusion):
        return self.forward(img, depthmap, occlusion, is_training=torch.tensor(True), modulate_phase=self.modulate_phase)

    def set_diffraction_efficiency(self, de: float):
        self.diffraction_efficiency = de





