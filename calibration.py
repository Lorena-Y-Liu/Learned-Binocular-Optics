import cv2
import numpy as np
import os
import glob
import torch
import torch.nn.functional as F
from debayer import Debayer2x2

def retrieve_files(directory, prefix):
    files = glob.glob(os.path.join(directory, prefix + '*.tif'))
    files.sort(key=lambda f: int(os.path.basename(f).replace(prefix, '').replace('.tif', '')))
    return files

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    return img
def torch_to_numpy(img_torch):
    return (img_torch.permute(1, 2, 0).numpy() * 65535).astype(np.uint16)
def stereo_rectify(left_img, right_img,directory):
    chessboard_size = (8, 6)
    square_size = 2.4  

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
    # Normalize mapx and mapy to [-1, 1]
    h, w = mapx.shape
    mapx = 2 * mapx / (w - 1) - 1
    mapy = 2 * mapy / (h - 1) - 1

    # Stack to create grid
    grid = torch.stack([torch.Tensor(mapx), torch.Tensor(mapy)], dim=-1).unsqueeze(0)

    # Perform grid sample
    remapped_image = F.grid_sample(image.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=False)

    return remapped_image.squeeze(0)

def white_balance(left_image, right_image,directory ):
    chessboard_size = (8, 6)
    square_size = 2.4 #cm  

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size
    file_list = os.listdir(directory)
    img_left_16bit_list = retrieve_files(directory,'left_cal_')
    img_right_16bit_list = retrieve_files(directory,'right_cal_')
    print (img_left_16bit_list)

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

    if (corners_left[0][0][0] + corners_left[0][0][1]) % 2 == 0:
        row, col = 0, 0  
    else:
        row, col = 0, 1  

    indices = [
        row * chessboard_size[0] + col,
        row * chessboard_size[0] + col + 1,
        (row + 1) * chessboard_size[0] + col,
        (row + 1) * chessboard_size[0] + col + 1
    ]

    square_corners = corners_left[indices].reshape(-1, 2).astype(int)
    square_size = np.linalg.norm(square_corners[0] - square_corners[1])
    square_corners = square_corners.astype(int)
    # img_marked = img_left_rgb.copy()
    # cv2.polylines(img_marked, [square_corners], isClosed=True, color=(0, 255, 0), thickness=2)


    # # cv2.imshow('Marked Image', img_marked)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()



    center_x, center_y = np.mean(square_corners, axis=0).astype(int)
    roi_size = square_size / 3  
    roi_x, roi_y = center_x - roi_size, center_y - roi_size
    roi_w, roi_h = roi_size * 2, roi_size * 2
    roi = img_left_rgb[int(roi_y):int(roi_y+roi_h), int(roi_x):int(roi_x+roi_w)]
    
        
    # img_with_roi = img_left_rgb.copy()
    # cv2.rectangle(img_with_roi, (int(roi_x), int(roi_y)), (int(roi_x + roi_w), int(roi_y + roi_h)), (0, 255, 0), 2)
    # cv2.imshow('Original Image', img_left_rgb)
    # cv2.imshow('Image with ROI', img_with_roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    mean_color = cv2.mean(roi)[:3]
    wb_coeffs = 255.0 / np.array(mean_color)
    wb_coeffs = torch.tensor(wb_coeffs, dtype=torch.float32).view(3, 1, 1)
    # corrected_img_left = (left_image.astype('float32') * wb_coeffs).clip(0, 255).astype('uint8')
    # corrected_img_right = (right_image.astype('float32') * wb_coeffs).clip(0, 255).astype('uint8')
    corrected_img_left = left_image * wb_coeffs
    corrected_img_right = right_image * wb_coeffs
    return corrected_img_left, corrected_img_right



if __name__ == "__main__":
    directory = "C:\\Users\\liangxun\\Desktop\\Capture\\Ring_test"  
    left_img = cv2.imread("C:\\Users\\liangxun\\Desktop\\Capture\\Ring_test\\left_1.tif", cv2.IMREAD_UNCHANGED) 
    right_img = cv2.imread("C:\\Users\\liangxun\\Desktop\\Capture\\Ring_test\\right_1.tif", cv2.IMREAD_UNCHANGED)  
    left_img = torch.from_numpy(left_img.astype(np.uint8)).unsqueeze(0).unsqueeze(0) 
    right_img = torch.from_numpy(right_img.astype(np.uint8)).unsqueeze(0).unsqueeze(0)
    debayer = Debayer2x2()
    left_linear = debayer(left_img).squeeze() #torch.Size([3, 1200, 1920])
    right_linear = debayer(right_img).squeeze()
    l, r = white_balance(left_img, right_img, directory)



    cv2.imshow('Rectified Left Image', l.numpy())
    cv2.imshow('Rectified Right Image', r.numpy())

    cv2.waitKey(0)


