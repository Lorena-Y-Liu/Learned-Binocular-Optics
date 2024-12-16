import os
import math
import PySpin
import sys
import cv2 
import numpy as np
import argparse
import torch
from argparse import ArgumentParser
from tkinter import *
from tkinter import simpledialog, messagebox
#from run_captured_stereo_only import *
from run_captured_stereo_only import load_model, run_deepstereo, to_uint8, rescale_image
import skimage.io
from Exposure_QuickSpin import configure_exposure
import torch.nn.functional as F




def srgb_to_linear(x, eps=1e-8):
    x = x.clamp(eps, 1.)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(x, eps=1e-8):
    a = 0.055
    x = x.clamp(eps, 1.)
    return torch.where(x <= 0.0031308, 12.92 * x, (1. + a) * x ** (1. / 2.4) - a)
def linear_to_srgb_np(x, eps=1e-8):
   
    x = torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [C, H, W]
    x = linear_to_srgb(x) 
    return x.squeeze(0).permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]

def srgb_to_linear_np(x, eps=1e-8):
    x = torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [C, H, W]
    x = srgb_to_linear(x)
    return x.squeeze(0).permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]


def im2float(im, dtype=np.float32):
    """convert uint16 or uint8 image to float32, with range scaled to 0-1

    :param im: image
    :param dtype: default np.float32
    :return:
    """
    if issubclass(im.dtype.type, np.floating):
        return im.astype(dtype)
    elif issubclass(im.dtype.type, np.integer):
        return im / dtype(np.iinfo(im.dtype).max)
    else:
        raise ValueError(f'Unsupported data type {im.dtype}')

def overlapping_resolution(depth: float, 
                           pixel_size: float = 5.86e-6, 
                           focal_length: float = 0.035, 
                           resolution_width: int = 1920, 
                           camera_distance: float = 0.070) -> int:
    
    sensor_width = 1920 * pixel_size
    fov = 2 * math.atan(sensor_width / (2 * focal_length))
    scene_width = 2 * math.tan(fov / 2) * depth
    overlap_width = scene_width - camera_distance
    overlap_resolution = int((overlap_width / scene_width) * resolution_width)

    return overlap_resolution


def getSerialNumber(nodemap):
    node_feature = PySpin.CValuePtr(nodemap.GetNode('DeviceSerialNumber'))
    #print('%s: %s' % (node_feature.GetName(), node_feature.ToString()))
    return node_feature.ToString()


def display_images(display_list,overlapping_resolution = 0, type = 'image'):


#     if type == 'image':
#         color_left = cv2.cvtColor(display_list[0], cv2.COLOR_BayerRG2RGB)  
#         color_right = cv2.cvtColor(display_list[1], cv2.COLOR_BayerRG2RGB)  
#         display_left = (color_left / 256).astype('uint8')
#         display_right = (color_right / 256).astype('uint8')  
#         concatenated_image = np.concatenate((display_left, display_right), axis=1)  
#         boundary_col_left = display_list[0].shape[1] - overlapping_resolution
#         boundary_col_right = display_list[0].shape[1] + overlapping_resolution
    
#         concatenated_image_dis = cv2.line(concatenated_image, (boundary_col_left, 0), (boundary_col_left, concatenated_image.shape[0]), (0, 0, 255), 8)
#         concatenated_image_dis = cv2.line(concatenated_image_dis, (boundary_col_right, 0), (boundary_col_right, concatenated_image.shape[0]), (0, 0, 255), 8)

#         scale_factor = 0.4
#         resized_image = cv2.resize(concatenated_image_dis, None, fx=scale_factor, fy=scale_factor)
        
#         cv2.imshow('Resized Image', resized_image)
    if type == 'image':
        color_left = cv2.cvtColor(display_list[0], cv2.COLOR_BayerRG2RGB)  
        color_right = cv2.cvtColor(display_list[1], cv2.COLOR_BayerRG2RGB)  
        display_left = (color_left / 256.0).astype('uint8')
        display_right = (color_right / 256.0).astype('uint8') 
        '''display_left = linear_to_srgb_np(display_left / 255.0) 
        display_right = linear_to_srgb_np(display_right / 255.0) '''
        concatenated_image = np.concatenate((display_left, display_right), axis=1)  
        boundary_col_left = display_list[0].shape[1] - overlapping_resolution
        boundary_col_right = display_list[0].shape[1] + overlapping_resolution

        concatenated_image_dis = cv2.line(concatenated_image, (boundary_col_left, 0), (boundary_col_left, concatenated_image.shape[0]), (0, 0, 255), 8)
        concatenated_image_dis = cv2.line(concatenated_image_dis, (boundary_col_right, 0), (boundary_col_right, concatenated_image.shape[0]), (0, 0, 255), 8)

        scale_factor = 0.35
        resized_image = cv2.resize(concatenated_image_dis, None, fx=scale_factor, fy=scale_factor)
        
        cv2.imshow('Resized Image', resized_image)


    elif type == 'depth':
        detph_map = rescale_image(display_list[0])
        data = (255 * (1 - detph_map).squeeze().clamp(0, 1)).to(torch.uint8).numpy()
        image_inferno = cv2.applyColorMap(data, cv2.COLORMAP_INFERNO)  
        image_gray = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)

        detph_map_2 = rescale_image(display_list[1])
        data_2 = (255 * (1 - detph_map_2).squeeze().clamp(0, 1)).to(torch.uint8).numpy()
        image_inferno_2 = cv2.applyColorMap(data_2, cv2.COLORMAP_INFERNO)  
        image_gray_2 = cv2.cvtColor(data_2, cv2.COLOR_GRAY2BGR)

        concatenated_image = np.hstack((image_inferno, image_inferno_2))
        scale_factor = 0.35
        resized_image = cv2.resize(concatenated_image, None, fx=scale_factor, fy=scale_factor)
        cv2.imshow('Depth maps', resized_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    elif type == 'deblur':
        display_left = display_list[0]/255
        display_right = display_list[1]/255
        concatenated_image = torch.cat((display_left,display_right), dim = 3)
        #concatenated_image = F.interpolate(concatenated_image, scale_factor=0.75, mode='bilinear', align_corners=False)
        concatenated_image = concatenated_image.squeeze(0).mul(255).clamp(0, 255).permute(1, 2, 0)
        concatenated_image = concatenated_image.cpu().numpy().astype('uint8')
        concatenated_image = cv2.cvtColor(concatenated_image, cv2.COLOR_RGB2BGR)
        scale_factor = 0.35
        resized_image = cv2.resize(concatenated_image, None, fx=scale_factor, fy=scale_factor)
        cv2.imshow('Deblur Image', resized_image)

def acquire_images(cam_list, NUM_IMAGES = 1):


    print('*** IMAGE ACQUISITION ***\n')
    try:
        result = True


        for i, cam in enumerate(cam_list):
            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            SerialNumber = getSerialNumber(nodemap_tldevice)
            if (SerialNumber != "20129287"):
                # Set acquisition mode to continuous
                node_acquisition_mode = PySpin.CEnumerationPtr(cam.GetNodeMap().GetNode('AcquisitionMode'))
                if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                    print('Unable to set acquisition mode to continuous (node retrieval; camera %d). Aborting... \n' % i)
                    return False

                node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
                if not PySpin.IsReadable(node_acquisition_mode_continuous):
                    print('Unable to set acquisition mode to continuous (entry \'continuous\' retrieval %d). \
                    Aborting... \n' % i)
                    return False

                acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

                node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

                print('Camera %d acquisition mode set to continuous...' % i)

                # Begin acquiring images
                cam.BeginAcquisition()

                print('Camera %d started acquiring images...' % i)

                print()

 
        processor = PySpin.ImageProcessor()
        processor.SetColorProcessing(PySpin.SPINNAKER_IMAGE_FILE_FORMAT_RAW)
        display_list= []
        for n in range(NUM_IMAGES):
            for i, cam in enumerate(cam_list):
                nodemap_tldevice = cam.GetTLDeviceNodeMap()
                SerialNumber = getSerialNumber(nodemap_tldevice)
                if (SerialNumber != "20129287"):
                    try:     
                        # Retrieve next received image and ensure image completion
                        image_result = cam.GetNextImage(1000)

                        if image_result.IsIncomplete():
                            print('Image incomplete with image status %d ... \n' % image_result.GetImageStatus())
                        else:
                            # Print image information
                            width = image_result.GetWidth()
                            height = image_result.GetHeight()
                            print('Camera %d grabbed image %d, width = %d, height = %d' % (i, n, width, height))

                            # Convert image to mono 8
                            
                            #image_converted = processor.Convert(image_result, PySpin.PixelFormat_BayerRG8)
                            #image_converted = processor.Convert(image_result, PySpin.PixelFormat_BayerRG16)

                            img_np_array = image_result.GetNDArray()  # Convert Spinnaker ImagePtr to numpy array
                            img_np_array = img_np_array - 64

                            if SerialNumber == "23191048":
                                display_list.insert(0,img_np_array)
                                
                            else:
                                display_list.append(img_np_array)
        
                        # Release image
                        image_result.Release()
                        print()

                    except PySpin.SpinnakerException as ex:
                        print('Error: %s' % ex)
                        result = False

        for cam in cam_list:
            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            SerialNumber = getSerialNumber(nodemap_tldevice)
            if (SerialNumber != "20129287"):
            # End acquisition
                cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    #display_list[0], display_list[1] = display_list[1], display_list[0]

    return result,display_list


def print_device_info(nodemap, cam_num):


    print('Printing device information for camera %d... \n' % cam_num)

    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

        if PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print('%s: %s' % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

        else:
            print('Device control information not readable.')
        print()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result

def run_multiple_cameras(cam_list,exposure):

    try:
        result = True


        print('*** DEVICE INFORMATION ***\n')

        for i, cam in enumerate(cam_list):
            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            SerialNumber = getSerialNumber(nodemap_tldevice)
            if (SerialNumber != "20129287"):
    
                # Print device information
                result &= print_device_info(nodemap_tldevice, i)

        for i, cam in enumerate(cam_list):
            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            SerialNumber = getSerialNumber(nodemap_tldevice)
            if (SerialNumber != "20129287"):
                cam.Init()
                result1 = configure_exposure(cam,exposure)

        # Acquire images on all cameras
        acquire_images(cam_list)
        result2, display_list = acquire_images(cam_list)
        result = result1 * result2
        for cam in cam_list:
            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            SerialNumber = getSerialNumber(nodemap_tldevice)
            if (SerialNumber != "20129287"):
                # Deinitialize camera
                cam.DeInit()
        del cam

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False
   #display_list = [display_list[1],display_list[0]]
    return result,display_list

def save_with_incremental_name(directory, prefix, image_array, type):  
    # List all files in the directory
    files = os.listdir(directory)
    # Use regex to extract the numbers from filenames that match the prefix
    if prefix == "left_cal" or prefix == "right_cal" :
        numbers = [int(re.search(f"{prefix}_(\d+)", file).group(1)) for file in files if re.match(f"{prefix}_(\d+)", file)]
        if numbers:
            new_number = max(numbers) +1
        else:
            new_number = 1
    else:
        numbers_l = [int(re.search(f"left_(\d+)", file).group(1)) for file in files if re.match(f"left_(\d+)", file)]
        numbers_r = [int(re.search(f"right_(\d+)", file).group(1)) for file in files if re.match(f"right_(\d+)", file)]


        try:
            numbers_l = [int(re.search(f"left_(\d+)", file).group(1)) for file in files if re.match(f"left_(\d+)", file)]
            max_l = max(numbers_l) if numbers_l else None
        except ValueError:
            max_l = None

        try:
            numbers_r = [int(re.search(f"right_(\d+)", file).group(1)) for file in files if re.match(f"right_(\d+)", file)]
            max_r = max(numbers_r) if numbers_r else None
        except ValueError:
            max_r = None

        if max_l is not None and max_r is not None:
            new_number = min(max_l, max_r)+(prefix == "left" or  prefix == "right")
        else:
            new_number = 1


    # Construct the new filename and save the image
    new_filename = os.path.join(directory, f"{prefix}_{new_number}.tif")
    if type == 'numpy':
        cv2.imwrite(new_filename, image_array)
        print('Image saved as %s' % new_filename)

    elif type == 'debayer':
        #skimage.io.imsave(new_filename, to_uint16(rescale_image(image_array)))
        data = to_uint8(image_array).numpy()
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.imwrite(new_filename, data)
        print('Image saved as %s' % new_filename)

    elif type == 'color_depth':
        data = (255 * (1 - image_array).squeeze().clamp(0, 1)).to(torch.uint8).numpy()
        data = cv2.applyColorMap(data, cv2.COLORMAP_INFERNO)  
        cv2.imwrite(new_filename, data)
        print('color_depth saved as %s' % new_filename)

    elif type == 'gray_depth':
        data = (255 * (1 - image_array).squeeze().clamp(0, 1)).to(torch.uint8).numpy()
        cv2.imwrite(new_filename, data)
        print('gray_depth saved as %s' % new_filename)


class CustomDialog(Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Choice")
        
        Label(self, text="Do you want to save the image?").pack(pady=10)
        
        Button(self, text="Save Image", command=self.save_image).pack(side="left", padx=10, pady=10)
        Button(self, text="Save Image and Run Model", command=self.save_image_run_model).pack(side="left", padx=10, pady=10)
        Button(self, text="Run Only", command=self.save_for_calibration).pack(side="left", padx=10, pady=10)
        
    def save_image(self):
        self.choice = 'save'
        self.destroy()
    
    def save_image_run_model(self):
        self.choice = 'save and run'
        self.destroy()
        
    def save_for_calibration(self):
        self.choice = 'run only'
        self.destroy()
        

def main_logic(depth, exposure, cam_list, directory, root, args , model, matching):

    display_list = run_multiple_cameras(cam_list,exposure)[1]
    if depth is not None:
        overlap_pixel_width = overlapping_resolution(depth)
        display_images(display_list, overlap_pixel_width)

        dialog = CustomDialog(root)
        root.wait_window(dialog)
        if dialog.choice == 'save':
            save_with_incremental_name(directory, "left", display_list[0],'numpy')
            save_with_incremental_name(directory, "right", display_list[1],'numpy')
            return
        elif dialog.choice == 'save and run':

            #left, right, estdepth = run_IGEV(display_list,directory,args,model)
            left,right,left_deblur,right_deblur,disp_ests,disp_ests_2 = run_deepstereo(display_list,args,model,matching)
            save_with_incremental_name(directory, "left", left.squeeze().cpu(),'debayer')
            save_with_incremental_name(directory, "right", right.squeeze().cpu(),'debayer')

            save_with_incremental_name(directory, "left_deblur", left_deblur.squeeze().cpu(),'debayer')
            save_with_incremental_name(directory, "right_deblur", right_deblur.squeeze().cpu(),'debayer')
            save_with_incremental_name(directory, "depthL", disp_ests.cpu(),'debayer')
            save_with_incremental_name(directory, "depth_colorL", disp_ests.cpu(),'color_depth')
            save_with_incremental_name(directory, "depth_grayL", disp_ests.cpu(),'gray_depth')
            save_with_incremental_name(directory, "depthR", disp_ests_2.cpu(),'debayer')
            save_with_incremental_name(directory, "depth_colorR", disp_ests_2.cpu(),'color_depth')
            save_with_incremental_name(directory, "depth_grayR", disp_ests_2.cpu(),'gray_depth')
            disp_list = [disp_ests.detach().cpu(),disp_ests_2.detach().cpu()]
            display_images(disp_list,overlapping_resolution = 0, type = 'depth')
            display_deblur = [left_deblur*255,right_deblur*255]

            display_images(display_deblur,overlapping_resolution = 0, type = 'deblur')
            # show_synced_images(display_image,display_deblur,disp_ests)

            return
        
        elif dialog.choice == 'run only':
            #left, right, estdepth = run_IGEV(display_list,directory,args,model)
            left,right,left_deblur,right_deblur,disp_ests,disp_ests_2 = run_deepstereo(display_list,args,model,matching)
            disp_list = [disp_ests.detach().cpu(),disp_ests_2.detach().cpu()]
            display_images(disp_list,overlapping_resolution = 0, type = 'depth')
            display_deblur = [left_deblur*255,right_deblur*255]

            display_images(display_deblur,overlapping_resolution = 0, type = 'deblur')
            return
        
        
def setup_input_window(root, cam_list, args):
    window = root
    window.title("Input Parameters")

    Label(window, text="Enter the project number:").pack(pady=(0, 5))
    project_entry = Entry(window)
    project_entry.insert(0, "scene_test") 
    project_entry.pack(pady=(0, 10))
    
    Label(window, text="Enter the depth (in meters):").pack(pady=(0, 5))
    depth_entry = Entry(window)
    depth_entry.insert(0, "5") 
    depth_entry.pack(pady=(0, 10))
    
    Label(window, text="Enter the exposure (in us):").pack(pady=(0, 5))
    exposure_entry = Entry(window)
    exposure_entry.insert(0, "80000") 
    exposure_entry.pack(pady=(0, 20))
    
    model, matching = load_model(args)
    def close_all_toplevels():
        for widget in root.winfo_children():
            if isinstance(widget, Toplevel):
                widget.destroy()
    def on_submit():
        project_number = project_entry.get()
        depth = float(depth_entry.get())
        exposure = float(exposure_entry.get())

        directory = os.path.join(args.savepath, project_number)
        if not os.path.exists(directory):
            os.makedirs(directory)
        close_all_toplevels()
        main_logic(depth, exposure, cam_list, directory, root, args, model, matching)
        
    #submit_button = Button(window, text="Submit", command=on_submit(root,cam_list,args))
    submit_button = Button(window, text="Submit", command=on_submit)
    submit_button.pack(pady=(0, 10))


def main(args):
    """
    Example entry point; please see Enumeration example for more in-depth
    comments on preparing and cleaning up the system.

    :return: True if successful, False otherwise.
    :rtype: bool
    """

    # Since this application saves images in the current folder
    # we must ensure that we have permission to write to this folder.
    # If we do not have permission, fail right away.
    try:
        test_file = open('test.txt', 'w+')
    except IOError:
        print('Unable to write to current directory. Please check permissions.')
        input('Press Enter to exit...')
        return False

    test_file.close()
    os.remove(test_file.name)

    result = True

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print('Number of cameras detected: %d' % num_cameras)

    # Finish if there are no cameras
    if num_cameras == 0:

        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        return False

    # Run example on all cameras
    print('Running example for all cameras...')

    result = run_multiple_cameras(cam_list)
    root = Tk()
    #root.title("Depth Input")
    setup_input_window(root, cam_list,args)
    root.mainloop()


    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()

    input('Done! Press Enter to exit...')
    return result

