import os
import math
import PySpin
import sys
import cv2 
import numpy as np
import argparse
from argparse import ArgumentParser
from tkinter import *
from tkinter import simpledialog, messagebox,ttk

from Exposure_QuickSpin import configure_exposure



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


def getSerialNumber(nodemap):
    node_feature = PySpin.CValuePtr(nodemap.GetNode('DeviceSerialNumber'))
    #print('%s: %s' % (node_feature.GetName(), node_feature.ToString()))
    return node_feature.ToString()

def display_image(img, type):


    if type == 'Image with cross line':

        
        color = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)   
        display = (color / 256).astype('uint8')
        image_line = cv2.line(display, (480, 600), (1440, 600), (0, 0, 255), 1)
        image_line = cv2.line(image_line, (960, 300), (960,900), (0, 0, 255), 1)
        scale_factor = 0.75
        resized_image = cv2.resize(image_line, None, fx=scale_factor, fy=scale_factor)
        
        cv2.imshow('PSF', resized_image)

    elif type == 'Capture Image': 
        
        color = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)   
        display = (color / 256).astype('uint8')

        scale_factor = 0.75
        resized_image = cv2.resize(display, None, fx=scale_factor, fy=scale_factor)
        
        cv2.imshow('PSF', resized_image)




def acquire_images(cam_list, SN, NUM_IMAGES):


    print('*** IMAGE ACQUISITION ***\n')
    try:
        result = True


        for i, cam in enumerate(cam_list):
            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            SerialNumber = getSerialNumber(nodemap_tldevice)
            if (SerialNumber == SN):
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
        for i, cam in enumerate(cam_list):
            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            SerialNumber = getSerialNumber(nodemap_tldevice)
            if (SerialNumber == SN):
                for n in range(NUM_IMAGES):
            
                    try:

                        # Retrieve next received image and ensure image completion
                        image_result = cam.GetNextImage(300)

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
                            if SerialNumber == "23191053":
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
            if (SerialNumber == SN):
            # End acquisition
                cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    average_image = np.mean(display_list, axis=0)
    average_image = np.uint16(average_image)
    display_list.insert(0, average_image)
    print(display_list[0].shape)

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

def run_multiple_cameras(cam_list,exposure,LorR):
    if LorR == 'L':
        SN = '23191053'
    else:
        SN = '23191048'
    try:
        result = True


        print('*** DEVICE INFORMATION ***\n')

        for i, cam in enumerate(cam_list):
            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            SerialNumber = getSerialNumber(nodemap_tldevice)
            if (SerialNumber == SN):
    
                # Print device information
                result &= print_device_info(nodemap_tldevice, i)

        for i, cam in enumerate(cam_list):
            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            SerialNumber = getSerialNumber(nodemap_tldevice)
            if (SerialNumber == SN):
                cam.Init()
                result1 = configure_exposure(cam,exposure)

        # Acquire images on all cameras
        acquire_images(cam_list,SN,1)
        result2, display_list = acquire_images(cam_list,SN,5)
        result = result1 * result2
        for cam in cam_list:
            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            SerialNumber = getSerialNumber(nodemap_tldevice)
            if (SerialNumber == SN):
                # Deinitialize camera
                cam.DeInit()
        del cam

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False
   #display_list = [display_list[1],display_list[0]]
    return result,display_list

def save_img(directory, prefix, image):  
    # List all files in the directory
    files = os.listdir(directory)
    # Use regex to extract the numbers from filenames that match the prefix
    numbers = [int(re.search(f"{prefix}_(\d+)", file).group(1)) for file in files if re.match(f"{prefix}_(\d+)", file)]

    # Find the max number and add 1 to it
    if numbers:
        new_number = max(numbers) + 1
    else:
        new_number = 1

    # Construct the new filename and save the image
    new_filename = os.path.join(directory, f"{prefix}_{new_number}.tif")
    
    cv2.imwrite(new_filename, image)
   


class CustomDialog(Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Choice")
        Label(self, text="Do you want to save the image?").pack(pady=10)
        
        Button(self, text="Save Image", command=self.save_image).pack(side="left", padx=10, pady=10)
        Button(self, text="Re-enter Depth", command=self.re_enter_depth).pack(side="left", padx=10, pady=10)
        
        
    def save_image(self):
        self.choice = 'save'
        self.destroy()
    def re_enter_depth(self):
        self.choice = 're_enter'
        self.destroy()
        
def main_logic(depth, exposure, cam_list, var, directory, root, cross):
    
    if var == 'L':
        display_list = run_multiple_cameras(cam_list,exposure,'L')[1]
        img = display_list[0]
    else:
        display_list = run_multiple_cameras(cam_list,exposure,'R')[1]
        img = display_list[0]
    if cross == True:
        display_image(img, 'Image with cross line')
    else:
        display_image(img,'Capture Image')


    dialog = CustomDialog(root)
    root.wait_window(dialog)

    if dialog.choice == 'save':
        save_img(directory, f"{depth}m", img)
        return
    
    elif dialog.choice == 're_enter':
        return

def setup_input_window(root, cam_list):
    window = root
    window.title("Input Parameters")

    Label(window, text="Enter the project number:").pack(pady=(0, 5))
    project_entry = Entry(window)
    project_entry.insert(0, "Test") 
    project_entry.pack(pady=(0, 10))
    

    Label(window, text="Enter the exposure (in us):").pack(pady=(0, 5))
    exposure_entry = Entry(window)
    exposure_entry.insert(0, "80000") 
    exposure_entry.pack(pady=(0, 10))

    Label(window, text="Enter the depth (in meters):").pack(pady=(0, 5))
    depth_entry = ttk.Combobox(window,width=17)
    #depth_entry['value'] = (0, 1, 1.056, 1.12, 1.19, 1.27, 1.36, 1.47, 1.6, 1.74, 1.92, 2.14, 2.42, 2.78, 3.26, 3.94, 5)
    #depth_entry['value'] = (0.67,0.76,0.87,1.02,1.24,1.57,2.14,3.38,8.00)
    depth_entry['value'] = (0.670,0.791, 0.965, 1.236, 1.722, 2.833, 5.000)#8.00)
    
    depth_entry.current(0)
    depth_entry.pack()  

    var = StringVar()
    r1 = Radiobutton(window, text='Left camera', variable=var, value='L')
    r1.pack()
    r2 = Radiobutton(window, text='Right camera', variable=var, value='R')
    r2.pack()

    var2 = IntVar()
    c1 = Checkbutton(window, text='Cross line',variable=var2, onvalue=1, offvalue=0)  
    c1.pack()

    def close_all_toplevels():
        for widget in root.winfo_children():
            if isinstance(widget, Toplevel):
                widget.destroy()

    def on_submit():
        project_number = project_entry.get()
        depth = float(depth_entry.get())

        exposure = float(exposure_entry.get())
        LorR = var.get()
        Cross = var2.get()
        directory = os.path.join('C:\\Users\\liangxun\\Desktop\\Capture\\PSF', project_number)
        if not os.path.exists(directory):
            os.makedirs(directory)
        close_all_toplevels() 

        main_logic(depth, exposure, cam_list, LorR, directory, root , Cross)
        
        
    #submit_button = Button(window, text="Submit", command=on_submit(root,cam_list,args))
    submit_button = Button(window, text="Submit", command=on_submit)
    submit_button.pack(pady=(0, 10))


def main():

    result = True

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()
    
    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()


    # cam_list.Clear()
    # system.ReleaseInstance()

    # system = PySpin.System.GetInstance()
    # cam_list = system.GetCameras()
    # version = system.GetLibraryVersion()
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

    #result = run_multiple_cameras(cam_list)
    root = Tk()
    root.title("Depth Input")
    setup_input_window(root, cam_list)
    root.mainloop()


    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()

    input('Done! Press Enter to exit...')
    return result

if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)