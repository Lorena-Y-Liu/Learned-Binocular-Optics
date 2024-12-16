"""
This script includes the modulations and components of the input field.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from authors).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

    
Technical Paper:
Haoyu Wei, Xin Liu, Xiang Hao, Edmund Y. Lam, and Yifan Peng, "Modeling off-axis diffraction with the least-sampling angular spectrum method," Optica 10, 959-962 (2023)
"""
import numpy as np
import torch
    
class SphericalWave():
    def __init__(self, k, x0, y0, z0, angles, zf) -> None:
        self.k = k.reshape(-1,1,1,1)
        thetaX, thetaY = angles
        self.fcX=torch.tensor(0)
        self.fcY=torch.tensor(0)
        #self.fcX = - torch.sin(thetaX / 180 * torch.pi) * k / (2 * torch.pi)
        #self.fcY = - torch.sin(thetaY / 180 * torch.pi) * k / (2 * torch.pi)
        self.x0, self.y0, self.z0 =x0.reshape(1,-1,1,1), y0.reshape(1,-1,1,1), z0.reshape(1,-1,1,1)
        self.zf = zf
        self.device='cuda'


    def forward(self, E0, xi_, eta_):
        ''' 
        Apply a spherical phase shift to E0 at coordinates xi_ and eta_
        '''
        E0 = torch.from_numpy(E0).to(self.device)
        E0 = E0.unsqueeze(0).unsqueeze(0)
        xi_=torch.from_numpy(xi_).unsqueeze(0).unsqueeze(0).to(self.device)
        eta_=torch.from_numpy(eta_).unsqueeze(0).unsqueeze(0).to(self.device)
        radius = torch.sqrt((self.z0.to(self.device))**2 + (xi_ - self.x0.to(self.device))**2 + (eta_ - self.y0.to(self.device))**2)
        phase = self.k.to(self.device) * radius
        amplitude = 1 / radius

        E = amplitude * torch.exp(1j * phase)
        E *= torch.exp(1j * 2 * torch.pi * (-self.fcX * xi_ - self.fcY * eta_))  # LPC

        return E0 * E

    
    def phase_gradient(self, xi, eta):
        '''
        Compute phase gradients at point (xi, eta)
        '''
        
        denom = torch.sqrt((xi - self.x0)**2 + (eta - self.y0)**2 + self.z0**2)
        grad_uX = self.k * (xi - self.x0) / denom
        grad_uY = self.k * (eta - self.y0) / denom

        grad_linearX = 2 * np.pi * self.fcX
        grad_linearY = 2 * np.pi * self.fcY

        gradientX = grad_uX - grad_linearX
        gradientY = grad_uY - grad_linearY

        return gradientX, gradientY


class ThinLens():
    def __init__(self, k, f) -> None:
        
        self.device='cuda'
        self.k = k.to(self.device).reshape(-1,1,1,1)
        self.f = f
        self.fcX = self.fcY = 0


    def forward(self, E0, xi_, eta_):
        xi_=torch.from_numpy(xi_).unsqueeze(0).unsqueeze(0).to(self.device)
        eta_=torch.from_numpy(eta_).unsqueeze(0).unsqueeze(0).to(self.device)
        phase = self.k.to(self.device) / 2 * (-1 / self.f) * (xi_**2 + eta_**2)
        return E0 * torch.exp(1j * phase)


    def phase_gradient(self, xi, eta):
        #xi_=xi_.unsqueeze(0).unsqueeze(0)
        #eta_=eta_.unsqueeze(0).unsqueeze(0)
        grad_uX = -self.k / self.f * xi
        grad_uY = -self.k / self.f * eta

        return grad_uX, grad_uY
    