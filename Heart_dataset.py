# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:13:06 2024

@author: aq22
"""
#### steps
# step
#### preprocessing dataset nifiti, dicom, png etc
############ dataset conversion####

###################### torch.tensor, numpy array ###

###########    datalaoder #######################
#2D=BxHxW(100xHXW)
#3D=BxHxWXD(100xHxWXD)
######################  model define ##################

#########  training and testing ######################

############### validation #########################

#%%
import nibabel as nib
import SimpleITK as sitk
path='C:/Users/aq22/Desktop/kcl2022/datasetmiccai2023/All_datasets/ACDC_2017/patient001/patient001_frame01_ED.nii.gz'
############# laod object
img=nib.load(path)
############### convert into array

img_array=img.get_fdata()

img1=img_array[:,:,6] ### HxWxd

#%%
import nibabel as nib
import SimpleITK as sitk
path='C:/Users/aq22/Desktop/kcl2022/datasetmiccai2023/All_datasets/ACDC_2017/patient001/patient001_frame01_ED_gt.nii.gz'
############# laod object
img=nib.load(path)
############### convert into array

img_array=img.get_fdata()

img1=img_array[:,:,6] ### HxWxd

#%% 
import os
from glob import glob
import nibabel as nib
path='C:/Users/aq22/Desktop/kcl2022/datasetmiccai2023/All_datasets/ACDC_2017'
pathimg=glob(os.path.join(path,'*','*_ED.nii.gz'))

for i in range(0,len(pathimg)):
    pathimged=pathimg[i]
    print(pathimged)
    img=nib.load(pathimged).get_fdata()
    print(img.shape)
    #break
    
#%%
import os
from glob import glob
import nibabel as nib
path='C:/Users/aq22/Desktop/kcl2022/datasetmiccai2023/All_datasets/ACDC_2017'
pathimg=glob(os.path.join(path,'*','*_ES.nii.gz'))

for i in range(0,len(pathimg)):
    pathimged=pathimg[i]
    print(pathimged)
    img=nib.load(pathimged).get_fdata()
    print(img.shape)
    #break
    
#%%
import os
from glob import glob
import nibabel as nib
path='C:/Users/aq22/Desktop/kcl2022/datasetmiccai2023/All_datasets/ACDC_2017'
pathimg=glob(os.path.join(path,'*','*_ED_gt.nii.gz'))

for i in range(0,len(pathimg)):
    pathimged=pathimg[i]
    print(pathimged)
    img=nib.load(pathimged).get_fdata()
    print(img.shape)
    #break
#%%
import os
from glob import glob
import nibabel as nib
path='C:\\Users\\aq22\\Desktop\\kcl2022\\datasetmiccai2023\\All_datasets\\Emidec_Database2'
pathimg=glob(os.path.join(path,'*','Images','*.nii.gz'))
pathlabel=glob(os.path.join(path,'*','Contours','*.nii.gz'))



#%%

#%% ################### dataset preprocessing codes ##################################

import os
import nibabel as nib
import glob
################################ ACDC dataset ################################
acdcdataset='C:\\Users\\aq22\\Desktop\\kcl2022\\datasetmiccai2023\\All_datasets\\ACDC_2017'

path_train_volumes_ed = sorted(glob.glob(os.path.join(acdcdataset, "*", "*_ED.nii.gz")))
path_train_volumes_es= sorted(glob.glob(os.path.join(acdcdataset, "*", "*_ES.nii.gz")))

path_train_segmentation_ed = sorted(glob.glob(os.path.join(acdcdataset, "*", "*_ED_gt.nii.gz")))
path_train_segmentation_es = sorted(glob.glob(os.path.join(acdcdataset, "*", "*_ES_gt.nii.gz")))

path_train_volumesacdc=path_train_volumes_ed+path_train_volumes_es

path_train_segmentationacdc=path_train_segmentation_ed+path_train_segmentation_es

#%% ################### MandM1 dataset ###################################
import os
import nibabel as nib
import glob
pathmm1='C:/Users/aq22/Desktop/kcl2022/datasetmiccai2023/All_datasets/MandM2_dataset'
path_train_volumes_mm1= sorted(glob.glob(os.path.join(pathmm1, "imagesTr", "*.nii.gz")))
path_train_segmentation_mm1= sorted(glob.glob(os.path.join(pathmm1, "labelsTr", "*.nii.gz")))
#%% ###################### CMRX dataset #################################
import os
import glob
pathcmrx='C:\\Users\\aq22\\Desktop\\kcl2022\\datasetmiccai2023\\All_datasets\\CMRxMotion Training Datasetmain\\updated_data\\data_update'
path_train_volumes_edc=sorted(glob.glob(os.path.join(pathcmrx,'*','*-ED.nii.gz')))
path_train_volumes_esc=sorted(glob.glob(os.path.join(pathcmrx,'*','*-ES.nii.gz')))

path_train_segmentation_edc=sorted(glob.glob(os.path.join(pathcmrx,'*','*-ED-label.nii.gz')))
path_train_segmentation_esc=sorted(glob.glob(os.path.join(pathcmrx,'*','*-ES-label.nii.gz')))

path_train_volumecmrx=path_train_volumes_edc+path_train_volumes_esc

path_train_segmentationcmrx=path_train_segmentation_edc+path_train_segmentation_esc


###################### total datasets lists#########################################
total_images_list=path_train_volumesacdc+path_train_volumes_mm1+path_train_volumecmrx
total_labels_list=path_train_segmentationacdc+path_train_segmentation_mm1+path_train_segmentationcmrx


#%% creat list of training path for training segmentation models ########################
train_files=[{'image':image_name,'label':label_name} for image_name,label_name in zip(total_images_list,total_labels_list)]
train_files[0]['image']
train_files[0]['label']


#%%#training and validation list 
import random
#random.seed(0)
def Trian_val(data_list,test_size=0.15):
    n=len(data_list)
    m=int(n*test_size)
    test_item=random.sample(data_list,m)
    train_item=list(set(data_list)-set(test_item))
    return train_item,test_item
tr_list,test_list=Trian_val(total_images_list,test_size=0.20)
#%% only images for SSL part
train_data=[{'image':image_name} for image_name in tr_list]
val_data=[{'image':image_name} for image_name in test_list]



#%%                    training and validation dataset loader for SSL
#%               training and validation datasets prepartion for SSL part

#https://github.com/Project-MONAI/tutorials/blob/main/self_supervised_pretraining/vit_unetr_ssl/ssl_train.ipynb
import os
import json
import time
import torch
import matplotlib.pyplot as plt

from torch.nn import L1Loss
from monai.utils import set_determinism, first
from monai.networks.nets import ViTAutoEnc
from monai.losses import ContrastiveLoss
from monai.data import DataLoader, Dataset
from monai.config import print_config
from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    CopyItemsd,
    SpatialPadd,
    EnsureChannelFirstd,
    Spacingd,
    OneOf,
    ScaleIntensityRanged,
    RandSpatialCropSamplesd,
    RandCoarseDropoutd,ToTensord,
    RandCoarseShuffled,AddChanneld
)

print_config()
# Define Training Transforms
train_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.2499995173568585, 1.2499998886073935, 9.599999507298296), mode=("bilinear")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=224.209,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=(96, 96, 96)),
        RandSpatialCropSamplesd(keys=["image"], roi_size=(96, 96, 96), random_size=False, num_samples=2),
        CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),
        OneOf(
            transforms=[
                RandCoarseDropoutd(
                    keys=["image"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True, max_spatial_size=32
                ),
                RandCoarseDropoutd(
                    keys=["image"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False, max_spatial_size=64
                ),
            ]
        ),
        RandCoarseShuffled(keys=["image"], prob=0.8, holes=10, spatial_size=8),
        # Please note that that if image, image_2 are called via the same transform call because of the determinism
        # they will get augmented the exact same way which is not the required case here, hence two calls are made
        OneOf(
            transforms=[
                RandCoarseDropoutd(
                    keys=["image_2"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True, max_spatial_size=32
                ),
                RandCoarseDropoutd(
                    keys=["image_2"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False, max_spatial_size=64
                ),
            ]
        ),
        RandCoarseShuffled(keys=["image_2"], prob=0.8, holes=10, spatial_size=8),
    ]
)
############################ orginal tranforms##################
my_transform_org=Compose([LoadImaged(keys=["image"]),AddChanneld(keys=["image"]),
                          ToTensord(keys=["image"])])


#%% ########################### training and validation dataset loader ############
batch_size=2
# Define DataLoader using MONAI, CacheDataset needs to be used
train_ds = Dataset(data=train_data, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

val_ds = Dataset(data=val_data, transform=train_transforms)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=0)


#%% ########################### visulization ################################

################## orginal tranform ###############

original_ds = Dataset(data=train_data, transform=my_transform_org)
original_loader = DataLoader(original_ds, batch_size=1)
original_patient = first(original_loader)

import matplotlib.pyplot as plt
number_slice = 6
plt.figure("display", (12, 6))
plt.subplot(1, 3, 1)
plt.title(f"Original patient slice {number_slice}")
plt.imshow(original_patient["image"][0, 0, :, :, number_slice], cmap="gray")


#%% training tranform data visluzation ##################################
################ check dataset with tranformation ###########################
check_ds = Dataset(data=train_data, transform=train_transforms)
check_loader = DataLoader(check_ds, batch_size=1)
check_data = first(check_loader)
image = check_data["image"][0][0]
print(f"image shape: {image.shape}")
import matplotlib.pyplot as plt
number_slice = 50
plt.figure("display", (12, 6))
plt.subplot(1, 3, 1)
plt.title(f"Original patient slice {number_slice}")
plt.imshow(check_data["gt_image"][0, 0, :, :, number_slice], cmap="gray")
plt.subplot(1, 3, 2)
plt.title(f"Generated patient slice {number_slice}")
plt.imshow(check_data["image"][0, 0, :, :, number_slice], cmap="gray")
plt.subplot(1, 3, 3)
plt.title(f"Generated patient slice {number_slice}")
plt.imshow(check_data["image_2"][0, 0, :, :, number_slice], cmap="gray")




