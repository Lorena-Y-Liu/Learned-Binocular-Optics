import torch
import torch.nn.functional as F

class Warp():

    def __init__(self):
        super().__init__()  

    def meshgrid(self, img, homogeneous=False):

        b, _, h, w = img.size()

        x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(img)  # [1, H, W]
        y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(img)

        grid = torch.cat((x_range, y_range), dim=0)  # [2, H, W], grid[:, i, j] = [j, i]
        grid = grid.unsqueeze(0).expand(b, 2, h, w)  # [B, 2, H, W]

        if homogeneous:
            ones = torch.ones_like(x_range).unsqueeze(0).expand(b, 1, h, w)  # [B, 1, H, W]
            grid = torch.cat((grid, ones), dim=1)  # [B, 3, H, W]
            assert grid.size(1) == 3
        return grid

    def normalize_coords(self, grid):
        """Normalize coordinates of image scale to [-1, 1]
        Args:
            grid: [B, 2, H, W]
        """
        assert grid.size(1) == 2
        h, w = grid.size()[2:]
        grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1  # x: [-1, 1]
        grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1  # y: [-1, 1]
        grid = grid.permute((0, 2, 3, 1))  # [B, H, W, 2]
        return grid


    def warp(self, r_img, L_disp, padding_mode='border'):
        """Warping by disparity
        Args:
            img: [B, 3, H, W]
            disp: [B, 1, H, W], positive
            padding_mode: 'zeros' or 'border'
        Returns:
            warped_img: [B, 3, H, W]
            valid_mask: [B, 3, H, W]
        """
        #assert disp.min() >= 0
        disp = L_disp #torch.clamp(L_disp, min=0)

        grid = self.meshgrid(r_img)  # [B, 2, H, W] in image scale
        # Note that -disp here
        offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)  # [B, 2, H, W]
        sample_grid = grid + offset
        sample_grid = self.normalize_coords(sample_grid)  # [B, H, W, 2] in [-1, 1]
        sample_grid = sample_grid.to(r_img.dtype)
        warped_img = F.grid_sample(r_img, sample_grid, mode='bilinear', padding_mode=padding_mode)


        return warped_img


    def warp_disp(self, img, L_disp, R_disp):
            """Warping by disparity
            Args:
                img: [B, 3, H, W], left or right img
                disp: [B, 1, H, W]
                padding_mode: 'zeros' or 'border'
                direction: '<-' warp right to left, '->' warp left to right
            Returns:
                warped_img: [B, 3, H, W]
                valid_mask: [B, 3, H, W]
            """
            direction = "<-"

            if direction == '->':
                L_disp, R_disp, img = R_disp.flip(-1), L_disp.flip(-1), img.flip(-1)

            R_disp = abs(R_disp)
            L_disp = abs(L_disp)
            # vaild mask
            L_warp= self.warp(R_disp, L_disp,padding_mode = 'zeros')
            diff_g = L_disp - L_warp
            M = torch.zeros_like(diff_g)
            M[abs(diff_g)<1] = 1
            M_ = torch.zeros_like(diff_g)
            M_[abs(L_disp)<1] = 1
            # M_b = torch.zeros_like(diff_g)

            mask = M + M_

            warped_img = self.warp(img, L_disp)

            if direction == '->':
                warped_img, mask = warped_img.flip(-1), mask.flip(-1)


            return warped_img*mask,mask