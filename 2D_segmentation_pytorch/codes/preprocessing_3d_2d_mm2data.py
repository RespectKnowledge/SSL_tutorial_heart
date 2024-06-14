# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 17:18:58 2024

@author: aq22
"""
#%%  dataset conversion from 3D to 2D segmentation
################################preprocessing and resizing function#######################################################

import numpy as np
import SimpleITK as sitk
from skimage.measure import regionprops
from glob import glob
import os

def ResampleXYZAxis(imImage, space=(1., 1., 1.), interp=sitk.sitkLinear):
    identity1 = sitk.Transform(3, sitk.sitkIdentity)
    sp1 = imImage.GetSpacing()
    sz1 = imImage.GetSize()

    sz2 = (int(round(sz1[0]*sp1[0]*1.0/space[0])), int(round(sz1[1]*sp1[1]*1.0/space[1])), int(round(sz1[2]*sp1[2]*1.0/space[2])))

    imRefImage = sitk.Image(sz2, imImage.GetPixelIDValue())
    imRefImage.SetSpacing(space)
    imRefImage.SetOrigin(imImage.GetOrigin())
    imRefImage.SetDirection(imImage.GetDirection())

    imOutImage = sitk.Resample(imImage, imRefImage, identity1, interp)

    return imOutImage

def ResampleLabelToRef(imLabel, imRef, interp=sitk.sitkNearestNeighbor):
    identity1 = sitk.Transform(3, sitk.sitkIdentity)

    imRefImage = sitk.Image(imRef.GetSize(), imLabel.GetPixelIDValue())
    imRefImage.SetSpacing(imRef.GetSpacing())
    imRefImage.SetOrigin(imRef.GetOrigin())
    imRefImage.SetDirection(imRef.GetDirection())
        
    ResampledLabel = sitk.Resample(imLabel, imRefImage, identity1, interp)
    
    return ResampledLabel



def ITKReDirection(itkimg, target_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)):
    # target direction should be orthognal, i.e. (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    # permute axis
    tmp_target_direction = np.abs(np.round(np.array(target_direction))).reshape(3,3).T
    current_direction = np.abs(np.round(itkimg.GetDirection())).reshape(3,3).T
    
    permute_order = []
    if not np.array_equal(tmp_target_direction, current_direction):
        for i in range(3):
            for j in range(3):
                if np.array_equal(tmp_target_direction[i], current_direction[j]):
                    permute_order.append(j)
                    #print(i, j)
                    #print(permute_order)
                    break
        redirect_img = sitk.PermuteAxes(itkimg, permute_order)
    else:
        redirect_img = itkimg
    # flip axis
    current_direction = np.round(np.array(redirect_img.GetDirection())).reshape(3,3).T
    current_direction = np.max(current_direction, axis=1)

    tmp_target_direction = np.array(target_direction).reshape(3,3).T 
    tmp_target_direction = np.max(tmp_target_direction, axis=1)
    flip_order = ((tmp_target_direction * current_direction) != 1)
    fliped_img = sitk.Flip(redirect_img, [bool(flip_order[0]), bool(flip_order[1]), bool(flip_order[2])])
    return fliped_img


def CropForeground(imImage, imLabel, context_size=[10, 30, 30]):
    # the context_size is in numpy indice order: z, y, x
    # Note that SimpleITK use the indice order of: x, y, z
    
    npImg = sitk.GetArrayFromImage(imImage)
    npLab = sitk.GetArrayFromImage(imLabel)

    mask = (npLab>0).astype(np.uint8) # foreground mask
    
    regions = regionprops(mask)
    assert len(regions) == 1

    zz, yy, xx = npImg.shape

    z, y, x = regions[0].centroid

    z_min, y_min, x_min, z_max, y_max, x_max = regions[0].bbox
    print('forground size:', z_max-z_min, y_max-y_min, x_max-x_min)

    z, y, x = int(z), int(y), int(x)

    z_min = max(0, z_min-context_size[0])
    z_max = min(zz, z_max+context_size[0])
    y_min = max(0, y_min-context_size[2])
    y_max = min(yy, y_max+context_size[2])
    x_min = max(0, x_min-context_size[1])
    x_max = min(xx, x_max+context_size[1])

    img = npImg[z_min:z_max, y_min:y_max, x_min:x_max]
    lab = npLab[z_min:z_max, y_min:y_max, x_min:x_max]

    croppedImage = sitk.GetImageFromArray(img)
    croppedLabel = sitk.GetImageFromArray(lab)


    croppedImage.SetSpacing(imImage.GetSpacing())
    croppedLabel.SetSpacing(imImage.GetSpacing())
    
    croppedImage.SetDirection(imImage.GetDirection())
    croppedLabel.SetDirection(imImage.GetDirection())

    return croppedImage, croppedLabel

import numpy as np
import SimpleITK as sitk
#from utils import ResampleXYZAxis, ResampleLabelToRef, CropForeground
import os
import random
import yaml
import copy
import pdb

def ResampleImage(imImage,imLabel,target_spacing=(1., 1., 1.)):

    assert imImage.GetSpacing() == imLabel.GetSpacing()
    assert imImage.GetSize() == imLabel.GetSize()


    spacing = imImage.GetSpacing()
    origin = imImage.GetOrigin()


    npimg = sitk.GetArrayFromImage(imImage).astype(np.int32)
    nplab = sitk.GetArrayFromImage(imLabel).astype(np.uint8)
    z, y, x = npimg.shape

    imImage.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    imLabel.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))


    re_img_xy = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline)
    re_lab_xy = ResampleLabelToRef(imLabel, re_img_xy, interp=sitk.sitkNearestNeighbor)

    re_img_xyz = ResampleXYZAxis(re_img_xy, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkNearestNeighbor)
    re_lab_xyz = ResampleLabelToRef(re_lab_xy, re_img_xyz, interp=sitk.sitkNearestNeighbor)
    

    cropped_img, cropped_lab = CropForeground(re_img_xyz, re_lab_xyz, context_size=[10, 15, 15])
    #print(cropped_img.GetSize())
    #print(cropped_lab.GetSize())
    return cropped_img,cropped_lab
 
def resize_mask(image,new_size):
    # Define image size
    original_size = image.GetSize()
    # Define the resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([sz * (orig / new) for sz, orig, new in zip(image.GetSpacing(), original_size, new_size)])
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Nearest neighbor for segmentation masks
    # Perform the resampling
    resized_mask = resampler.Execute(image)
    return resized_mask

def resize_image(image,new_size):
    # Define image size
    original_size = image.GetSize()
    # Define the resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([sz * (orig / new) for sz, orig, new in zip(image.GetSpacing(), original_size, new_size)])
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)  # Use linear interpolation for intensity images
    # Perform the resampling
    resized_image = resampler.Execute(image)
    return resized_image

############## read image and label from main folder ##########
pathdata='C:/Users/aq22/Desktop/kcl2022/Pytorch_basic_tutorials/heart_dataset/datasets/MandM2_dataset/'
pathimg=glob(os.path.join(pathdata,'images','*.nii.gz'))
pathlabel=glob(os.path.join(pathdata,'labels','*.nii.gz'))


################# save 2d images and 2d masks

pathsaveim='C:/Users/aq22/Desktop/kcl2022/Pytorch_basic_tutorials/heart_dataset/datasets/MandM2_dataset/preprocess_data/image'
pathsavelabel='C:/Users/aq22/Desktop/kcl2022/Pytorch_basic_tutorials/heart_dataset/datasets/MandM2_dataset/preprocess_data/mask'

#from skimage.transform import io
import skimage.io as io
from skimage import io, color
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
################### read data, crop and convert into 2d masks
for i in range(0,len(pathimg)):
    pathim=pathimg[i]
    pathma=pathlabel[i]
    img=sitk.ReadImage(pathim)
    #img_array=img.get_fdata()
    mask=sitk.ReadImage(pathma)
    ############# get interplation data#################################
    cropped_img,cropped_lab=ResampleImage(img, mask, (1.2333333202870196, 1.233333339535614, 9.599999505734397))
    #cropped_img,cropped_lab=ResampleImage1(img, mask, (1.4583333730697632, 1.4583333730697632, 3.0))
    resized_image=resize_image(cropped_img,(224,224,16))
    resized_mask=resize_mask(cropped_lab,(224,224,16))
    print(resized_image.GetSize())
    print(resized_mask.GetSize())
    img_array=sitk.GetArrayFromImage(resized_image)
    mask_array=sitk.GetArrayFromImage(resized_mask)
    patient_name=pathim.split('\\')[-1].split('.')[0]
    ### 16x224x224 3d volume
    for j in range(0,img_array.shape[0]):
        img=img_array[j,:,:] ####[0,16,:,:]
        img = (255 * (img - np.min(img)) / np.ptp(img)).astype(np.uint8)
        # Convert to uint8 (values in the range [0, 255])
        img= color.gray2rgb(img)
        img = img_as_ubyte(img)
        print(img.shape)
        #img= Image.fromarray(img, mode='F')
        msk=mask_array[j,:,:].astype(np.uint8)
        #msk= Image.fromarray(msk,mode='F')
        #print(img.shape)
        #plt.imshow(img)
        #plt.imshow(msk)
        print(np.unique(msk))
        #img.save(os.path.join(pathsaveim,patient_name+'_'+str(j)+'.png'))
        #msk.save(os.path.join(pathsavelabel,patient_name+'_'+str(j)+'.png'))
        io.imsave(os.path.join(pathsaveim,patient_name+'_'+str(j)+'.png'),img)
        io.imsave(os.path.join(pathsavelabel,patient_name+'_'+str(j)+'.png'),msk)
        
    #break
