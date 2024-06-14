# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 17:18:58 2024

@author: aq22
"""

#%%
#%% How to assed spacing and normaization in monai
from monai.data import (
    Dataset,
    DatasetSummary,
    #LoadImaged,
)
from monai.transforms import(
    LoadImaged,
)
from glob import glob
import os
pathdata='C:/Users/aq22/Desktop/kcl2022/Pytorch_basic_tutorials/heart_dataset/datasets/MandM2_dataset/'
pathimg=glob(os.path.join(pathdata,'images','*.nii.gz'))
pathlabel=glob(os.path.join(pathdata,'labels','*.nii.gz'))
train_files=[{'image':image_name,'label':label_name} for image_name, label_name in zip(pathimg, pathlabel)]
dataset = Dataset(data=train_files,transform=LoadImaged(keys=["image", "label"]),)

calculator = DatasetSummary(dataset)
target_spacing = calculator.get_target_spacing()
print("spacing: ", target_spacing)
#this function also prints the mean and std values (used for normalization), 
calculator.calculate_statistics()
print("mean: ", calculator.data_mean, " std: ", calculator.data_std)
#the min (0.5 percentile) and max(99.5 percentile) values (used for clip).
calculator.calculate_percentiles(sampling_flag=True, interval=10, min_percentile=0.5, max_percentile=99.5)
print("min: ",calculator.data_min_percentile," max: ",calculator.data_max_percentile,)

#spacing:  (1.2333333202870196, 1.233333339535614, 9.599999505734397)
#mean:  230.79139709472656  std:  229.6168212890625
#min:  13.0  max:  1341.0
#%%
import splitfolders
input_folder='C:/Users/aq22/Desktop/kcl2022/Pytorch_basic_tutorials/heart_dataset/datasets/MandM2_dataset/dataset'
output='C:/Users/aq22/Desktop/kcl2022/Pytorch_basic_tutorials/heart_dataset/datasets/MandM2_dataset/train_test_dataset'
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio(input_folder, output=output,seed=1337, ratio=(.8, .2), group_prefix=None, move=False) # default values






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

def back_org_seg(image,image_resize):
    #image_resize=sitk.ReadImage(resize)
    # Define the new size (example: resize to half the original size)
    original_size = image.GetSize()
    new_size = [int(sz) for sz in original_size]
    #new_size=[128,128,16]
    # Define the resampler
    resampler = sitk.ResampleImageFilter()
    original_spacing = image.GetSpacing()
    #resampler.SetOutputSpacing([sz * 2 for sz in image.GetSpacing()])  # Adjust spacing accordingly
    #resampler.SetOutputSpacing([sz for sz in image.GetSpacing()])  # Adjust spacing accordingly
    #resampler.SetOutputSpacing([sz * (orig / new) for sz, orig, new in zip(image.GetSpacing(), original_size, new_size)])
    new_spacing = [
       original_spacing[i] * (original_size[i] / new_size[i])
       for i in range(len(new_size))
   ]
    resampler.SetOutputSpacing(new_spacing)
    
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Nearest neighbor for segmentation masks
    #resampler.SetOutputOrigin(image.GetOrigin())
    # Perform the resampling
    resized_image = resampler.Execute(image_resize)
    #print(resized_image.GetSize())
    return resized_image

def back_org_image(image,image_resize):
    #image_resize=sitk.ReadImage(resize)
    # Define the new size (example: resize to half the original size)
    original_size = image.GetSize()
    new_size = [int(sz) for sz in original_size]
    original_spacing = image.GetSpacing()
    #new_size=[128,128,16]
    # Define the resampler
    resampler = sitk.ResampleImageFilter()
    #resampler.SetOutputSpacing([sz * 2 for sz in image.GetSpacing()])  # Adjust spacing accordingly
    #resampler.SetOutputSpacing([sz for sz in image.GetSpacing()])  # Adjust spacing accordingly
    #print([sz for sz in image.GetSpacing()])
    new_spacing = [
       original_spacing[i] * (original_size[i] / new_size[i])
       for i in range(len(new_size))
   ]
    resampler.SetOutputSpacing(new_spacing)
    #resampler.SetOutputSpacing([sz * (orig / new) for sz, orig, new in zip(image.GetSpacing(), original_size, new_size)])
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)  # Nearest neighbor for segmentation masks
    #resampler.SetOutputOrigin(image.GetOrigin())
    # Perform the resampling
    resized_image = resampler.Execute(image_resize)
    #print(resized_image.GetSize())
    return resized_image
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

pathdata='C:/Users/aq22/Desktop/kcl2022/Pytorch_basic_tutorials/heart_dataset/datasets/MandM2_dataset/'
pathimg=glob(os.path.join(pathdata,'images','*.nii.gz'))
pathlabel=glob(os.path.join(pathdata,'labels','*.nii.gz'))
pathsaveim='C:/Users/aq22/Desktop/kcl2022/Pytorch_basic_tutorials/heart_dataset/datasets/MandM2_dataset/preprocess_data/image'
pathsavelabel='C:/Users/aq22/Desktop/kcl2022/Pytorch_basic_tutorials/heart_dataset/datasets/MandM2_dataset/preprocess_data/mask'

#from skimage.transform import io
import skimage.io as io
from skimage import io, color
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
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

#%% Dataset in pytorch

import torch 
import cv2
import os
from torch.utils.data import Dataset
import albumentations as A
import albumentations.augmentations.functional as F
import matplotlib.pyplot as plt
import numpy as np
#from albumentations.pytorch import ToTensorV2
#from albumentations.pytorch.transforms import ToTensorV2
# Data Transform Funtion 
from PIL import Image 
# Dataset Class

class Danlin_First_Dataset(Dataset):
  def __init__(self,image,mask,trans=None):
    self.image=image
    self.mask=mask
    self.trans=trans
    self.num_classes=4

    # List directory for images
    list_image=os.listdir(self.image)
    self.path_image=[]
    for i in list_image:
      pathimg=os.path.join(self.image,i)
      self.path_image.append(pathimg)

    # List directory for mask
    list_masks=os.listdir(self.mask) 
    self.path_mask=[]
    for i in list_masks:
      pathmsk=os.path.join(self.mask,i) 
      self.path_mask.append(pathmsk)

  
  def __getitem__(self,idx):
    imagePath=self.path_image[idx]
    maskPath=self.path_mask[idx]

    # to read and resize an image
    #image=cv2.imread(imagePath)
    image=Image.open(imagePath)
    #image=(np.asarray(image) / 65536).astype('float32')
    image=(np.asarray(image))
    #image=cv2.resize(image,(224,224))
    mask=cv2.imread(maskPath,0)
    #mask=cv2.resize(mask,(224,224))
    # convert into onehot encoding
    #mask_oh = torch.nn.functional.one_hot(torch.from_numpy(mask).long(), self.num_classes).permute(2,0,1).float()

    if self.trans:
      t = self.trans(image=image, mask=mask)
      image = t['image']
      mask = t['mask'] 
    img_t=torch.from_numpy(image).float().permute(2,0,1)
    #img_t=image
    mask_t=mask
    
    # convert into onehot encoding
    mask_t = torch.nn.functional.one_hot(torch.from_numpy(mask_t).long(), self.num_classes).permute(2,0,1).float()

    return img_t,mask_t

  def __len__(self):
    return len(self.path_image)
# Giving Folder paths

image_folder_train="C:/Users/aq22/Desktop/kcl2022/Pytorch_basic_tutorials/heart_dataset/datasets/MandM2_dataset/preprocess_data/image"
mask_folder_train="C:/Users/aq22/Desktop/kcl2022/Pytorch_basic_tutorials/heart_dataset/datasets/MandM2_dataset/preprocess_data/mask"

#image_folder_valid="D:\\FETA Dataset\\Fetadataset\\Fetadataset\\validationdata\\images"
#mask_folder_valid="D:\\FETA Dataset\\Fetadataset\\Fetadataset\\validationdata\\masks"

from albumentations.pytorch import ToTensorV2
import albumentations as A
############ validation transform ######################
trans = A.Compose([
    A.Resize(224, 224),
    ToTensorV2,
])
#################### training transform ############################
transform_train= A.Compose({
        A.Resize(224, 224),
        A.CenterCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        #A.Rotate(limit=(-90, 90)),
        A.VerticalFlip(p=0.5),
        #A.RandomRotate90(p=0.5),
        # A.OneOf([
        # A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        # A.GridDistortion(p=0.5),
        #A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)], p=0.8),
        #A.CLAHE(p=0.8),
        #A.RandomBrightnessContrast(p=0.8),
        #A.RandomGamma(p=0.8),
        #A.GridDistortion(p=0.5),
        #ToTensorV2,
        })

# Creating Class instance

dataset_train=Danlin_First_Dataset(image_folder_train,mask_folder_train,trans=transform_train)
#dataset_valid=Danlin_First_Dataset(image_folder_valid,mask_folder_valid,trans=trans)

# Checking the length of the dataset
len(dataset_train)
image,mask=dataset_train[0]
print(image.shape)
print(mask.shape)
from torch.utils.data import DataLoader

dataset=dataset_train
len(dataset) 
imges,masks=dataset[115]
print(imges.shape)
print(masks.shape)
print(imges.dtype)
print(masks.dtype)
print(imges.max())
print(masks.max())
print(imges.min())
print(masks.min())
#imgesf=imges[:,:,2]
mask1=masks[1,:,:].numpy()
mask2=masks[2,:,:].numpy()
mask3=masks[3,:,:].numpy()
mask4=masks[0,:,:].numpy()
####################### check mask ########################
# import random
# ix = random.randint(0, len(dataset))
# img, mask= dataset[ix]
# fig, ax = plt.subplots(dpi=50)
# ax.imshow(img, cmap="gray")
# ax.axis('off')
# mask = torch.argmax(mask, axis=0).float().numpy()
# mask[mask == 0] = np.nan
# ax.imshow(mask, alpha=0.5)
# plt.show()

#################### take the batch size and prepare dataloader ######
batch_size=25

dataloader=torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)


imgs, masks = next(iter(dataloader))
imgs.shape, masks.shape

import matplotlib.pyplot as plt

r, c = 5, 5
fig = plt.figure(figsize=(5*r, 5*c))
for i in range(r):
    for j in range(c):
        ix = c*i + j
        ax = plt.subplot(r, c, ix + 1)
        ax.imshow(imgs[ix][0,:,:].squeeze(0),cmap="gray")
        #print(imgs[ix].squeeze(0).shape)
        mask = torch.argmax(masks[ix], axis=0).float().numpy()
        mask[mask == 0] = np.nan
        ax.imshow(mask, alpha=0.5)
        ax.axis('off')
plt.tight_layout()
plt.show()