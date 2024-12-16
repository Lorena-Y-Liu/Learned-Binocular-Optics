import cv2
import numpy as np
import torch
import os
import glob
import torch.nn.functional as F


def retrieve_files(directory, prefix):
    files = glob.glob(os.path.join(directory, prefix + '*.tif'))  
    files.sort(key=lambda f: int(os.path.basename(f).replace(prefix, '').replace('.tif', '')))
    return files



def stereo_rectify(left_img, right_img,directory):
    chessboard_size = (8, 6)
    square_size = 2.4 #cm  

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size
    file_list = os.listdir(directory)
    img_left_16bit_list = retrieve_files(directory,'left_cal_')
    img_right_16bit_list = retrieve_files(directory,'right_cal_')
    

    img_left_16bit = cv2.imread(img_left_16bit_list[0], cv2.IMREAD_UNCHANGED)
    img_right_16bit = cv2.imread(img_right_16bit_list[0], cv2.IMREAD_UNCHANGED)

    img_left_8bit = cv2.normalize(img_left_16bit, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    img_right_8bit = cv2.normalize(img_right_16bit, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    img_left_rgb = cv2.cvtColor(img_left_8bit, cv2.COLOR_BAYER_BG2BGR)
    img_right_rgb = cv2.cvtColor(img_right_8bit, cv2.COLOR_BAYER_BG2BGR)

    gray_left = cv2.cvtColor(img_left_rgb, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right_rgb, cv2.COLOR_BGR2GRAY)

    ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

    if ret_left and ret_right:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
        mtx = np.eye(3, dtype=np.float32)
        dist = np.zeros(5, dtype=np.float32)
        _, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            [objp], [corners_left], [corners_right], 
            mtx, dist, mtx, dist, 
            gray_left.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC
        )

        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            mtx, dist, mtx, dist, 
            gray_left.shape[::-1], R, T
        )
        
        mapx_left, mapy_left = cv2.initUndistortRectifyMap(
            mtx, dist, R1, P1, gray_left.shape[::-1], cv2.CV_32FC1
        )
        mapx_right, mapy_right = cv2.initUndistortRectifyMap(
            mtx, dist, R2, P2, gray_right.shape[::-1], cv2.CV_32FC1
        )

        rectified_left = cv2.remap(img_left_rgb, mapx_left, mapy_left, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(img_right_rgb, mapx_right, mapy_right, cv2.INTER_LINEAR)
        rectified_input_left = remap_torch(left_img, mapx_left, mapy_left)
        rectified_input_right = remap_torch(right_img, mapx_left, mapy_left)


        return(rectified_input_left,rectified_input_right)
    else:
        print("Failed") 

def remap_torch(image, mapx, mapy):
    """
    Perform remapping on torch tensor image using mapx and mapy.

    :param image: torch tensor of shape (C, H, W)
    :param mapx: np.ndarray of shape (H, W)
    :param mapy: np.ndarray of shape (H, W)
    :return: remapped image as torch tensor of shape (C, H, W)
    """
    # Normalize mapx and mapy to [-1, 1]
    h, w = mapx.shape
    mapx = 2 * mapx / (w - 1) - 1
    mapy = 2 * mapy / (h - 1) - 1

    # Stack to create grid
    grid = torch.stack([torch.Tensor(mapx), torch.Tensor(mapy)], dim=-1).unsqueeze(0)

    # Perform grid sample
    remapped_image = F.grid_sample(image.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=False)

    return remapped_image.squeeze(0)