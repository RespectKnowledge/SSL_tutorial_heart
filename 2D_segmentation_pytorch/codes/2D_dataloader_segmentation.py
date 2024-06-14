# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 19:51:03 2024

@author: aq22
"""

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

class Segmentation_Dataset(Dataset):
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

#from albumentations.pytorch import ToTensorV2
import albumentations as A
############ validation transform ######################
trans = A.Compose([
    A.Resize(224, 224),
    #ToTensorV2,
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

dataset_train=Segmentation_Dataset(image_folder_train,mask_folder_train,trans=transform_train)
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