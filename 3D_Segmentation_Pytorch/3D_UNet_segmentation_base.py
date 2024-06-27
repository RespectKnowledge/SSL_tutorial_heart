# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:04:08 2024

@author: aq22
"""
#%% dataloader
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
    #print('forground size:', z_max-z_min, y_max-y_min, x_max-x_min)

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

#%
import torch
from torch.utils.data import Dataset
class myseg(Dataset):
    def __init__(self,pathdata,trans=None):
        self.pathdata=pathdata
        self.trans=trans
        self.pathimg=glob(os.path.join(self.pathdata,'images','*.nii.gz'))
        self.pathlabel=glob(os.path.join(self.pathdata,'labels','*.nii.gz'))
        
        
        
    def __getitem__(self, index):
        
        imgpath=self.pathimg[index]
        imglabel=self.pathlabel[index]
        img=sitk.ReadImage(imgpath)
        #img_array=img.get_fdata()
        mask=sitk.ReadImage(imglabel)
        #mask_array=mask.get_fdata()
        ############# get interplation data#################################
        cropped_img,cropped_lab=ResampleImage(img, mask, (1.2333333202870196, 1.233333339535614, 9.599999505734397))
        #cropped_img,cropped_lab=ResampleImage1(img, mask, (1.4583333730697632, 1.4583333730697632, 3.0))
        
        resized_image=resize_image(cropped_img,(224,224,32))
        resized_mask=resize_mask(cropped_lab,(224,224,32))
        
        img_array=sitk.GetArrayFromImage(resized_image)
        mask_array=sitk.GetArrayFromImage(resized_mask)
        #img_array=img_array.from_numpy
        ### add channel############
        img_array_t=np.expand_dims(img_array,0)
        #mask_array_t=np.expand_dims(mask_array,0)
        mask_array_t=mask_array
        ############# convert into torch tensor #######
        img_array_t=torch.from_numpy(img_array_t)
        #### convert into one hot based on number of classes
        mask_array_o=torch.nn.functional.one_hot(torch.from_numpy(mask_array_t).long(),4).permute(3,0,1,2).float()

        return img_array_t,mask_array_o
    
    
    def __len__(self):
        
        return len(self.pathimg)
    
pathdata='/home/aqayyum/Segmentation_3d_pytorch/train_test_dataset/train'
#pathdata='C:/Users/aq22/Desktop/kcl2022/Pytorch_basic_tutorials/heart_dataset/datasets/MandM2_dataset/train_test_dataset/train'
datast_train=myseg(pathdata,trans=None)

########### test single sample ####################
img,labl=datast_train[10]
#print(img.shape)
#print(labl.shape)
# import matplotlib.pyplot as plt
# plt.imshow(img[0,6,:,:])
# plt.imshow(labl[0,0,6,:,:]) ### all with background
# plt.imshow(labl[0,1,6,:,:]) ## blood-pool
# plt.imshow(labl[0,2,6,:,:]) ### myo
# plt.imshow(labl[0,3,6,:,:]) ### Right venricle

pathval='/home/aqayyum/Segmentation_3d_pytorch/train_test_dataset/val'
#pathval='C:/Users/aq22/Desktop/kcl2022/Pytorch_basic_tutorials/heart_dataset/datasets/MandM2_dataset/train_test_dataset/val'
datast_val=myseg(pathval,trans=None)

############## train and val dataloaders
from torch.utils.data import DataLoader
train_dataloader=DataLoader(datast_train,batch_size=2,shuffle=True,num_workers=4)
valid_dataloader=DataLoader(datast_val,batch_size=2,shuffle=False,num_workers=4)

#%% 3D segmentation model
######### more compact representation of 3D model
import torch.nn as nn
def conv_3d_block(cin,cout):
    conv_block=nn.Sequential(nn.Conv3d(cin,cout,3,1,1),
                             nn.ReLU(inplace=True),
                             nn.BatchNorm3d(cout),)
    return conv_block

class decode(nn.Module):
    def __init__(self,cin,cout):
        super(decode,self).__init__()
        self.decode=nn.ConvTranspose3d(cin,cout,2,2)
        self.conv_block=conv_3d_block(cin,cout)

    def forward(self,x1,x2):
        x1=self.decode(x1)
        #print(x1.shape)
        #decod=self.conv_block(concat)
        concat=torch.concat([x1,x2],dim=1)
        decod=self.conv_block(concat)
        return decod


class my_3D_UNet(nn.Module):
    def __init__(self):
        super(my_3D_UNet,self).__init__()
        ######### encoder block #######
        self.encod1=conv_3d_block(1,16) ### 16x64x128x128
        self.maxpool=nn.MaxPool3d(2)
        self.encod2=conv_3d_block(16,32)  ### 32x32x64x64
        self.encod3=conv_3d_block(32,64) ### 64x16x32x32
        self.botom=nn.Conv3d(64,128,3,1,1) ##### 128x8x16x16
        #### decoder block
        self.decod1=decode(128,64)
        self.decod2=decode(64,32)
        self.decod3=decode(32,16)
        self.final=nn.Conv3d(16,4,1)

    def forward(self,x):
        encod1=self.encod1(x)
        maxpool1=self.maxpool(encod1)
        #print(f'ecode1: {encod1.shape}')
        encod2=self.encod2(maxpool1)
        maxpool2=self.maxpool(encod2)
        #print(f'ecode2: {encod2.shape}')
        encod3=self.encod3(maxpool2)
        maxpool3=self.maxpool(encod3)
        #print(f'ecode3: {encod3.shape}')
        bottom=self.botom(maxpool3)
        #print(bottom.shape)
        #print(f'bottom: {bottom.shape}')
        ########### decode1 ###########
        decod1=self.decod1(bottom,encod3)
        decod2=self.decod2(decod1,encod2)
        decod3=self.decod3(decod2,encod1)
        out=self.final(decod3)
        return out
model=my_3D_UNet()
# out=model(torch.rand(1,1,32,224,224))
# print(out.shape)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=model.to(device)
#%% training and validation functions

from torch.nn import BCEWithLogitsLoss
loss_f=BCEWithLogitsLoss()

################## define iou for 3D segmentation assesment ####
def iou(outputs, labels):
    # apply sigmoid and convert to binary
    outputs, labels = torch.sigmoid(outputs) > 0.5, labels > 0.5
    SMOOTH = 1e-6
    # BATCH x num_classes x D x H x W
    B, N, D, H, W = outputs.shape
    ious = []
    for i in range(N-1): # we skip the background
        _out, _labs = outputs[:,i,:,:,:], labels[:,i,:,:,:]
        #print(_out.shape)
        #print((_out & _labs).float().shape)
        
        intersection = (_out & _labs).float().sum((1,2,3))  
        union = (_out | _labs).float().sum((1,2,3))         
        iou = (intersection + SMOOTH) / (union + SMOOTH)  
        ious.append(iou.mean().item())
    return np.mean(ious) 

best_iou=0
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)

path_save='/home/aqayyum/Segmentation_3d_pytorch/base_3D_UNet.pth'
epochs=100
########################## save training and testing losses and ious ###
history={'train_loss':[],'train_ious':[],'val_loss':[],'val_ious':[]}
for epoch in range(epochs):
    model.train()
    model.to(device)
    train_loss=[]
    train_ious=[]
    for i,d in enumerate(train_dataloader):
        imges,labels=d
        imges=imges.to(device).float()
        labels=labels.to(device).float()
        outputs=model(imges)
        loss=loss_f(outputs,labels)
        ############ zero_grad, loss backward,update gradients ##
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ################ compute ios ########################
        ious=iou(outputs,labels)
        train_loss.append(loss.item())
        train_ious.append(ious.item())
    
    history['train_loss'].append(np.mean(train_loss))
    history['train_ious'].append(np.mean(train_ious))
    print(f'epoch:{epoch},train_loss:{np.mean(train_loss)},train_iou:{np.mean(train_ious)}')
    val_loss=[]
    val_ious=[]
    model.eval()
    with torch.no_grad():
        for i,d in enumerate(valid_dataloader):
            imges_v,labels_v=d
            imges_v=imges_v.to(device).float()
            labels_v=labels_v.to(device).float()
            outputs=model(imges_v)
            loss=loss_f(outputs,labels_v)
            ious=iou(outputs,labels_v)
            val_loss.append(loss.item())
            val_ious.append(ious.item())
    history['val_loss'].append(np.mean(val_loss))
    history['val_ious'].append(np.mean(val_ious))
    print(f'epoch:{epoch},val_loss:{np.mean(val_loss)},val_iou:{np.mean(val_ious)}')
    val_iou_epoch=np.mean(val_ious)
    #print(val_iou_epoch)
    if val_iou_epoch > best_iou:
        best_iou=val_iou_epoch
        torch.save(model.state_dict(),path_save)
        print(f'best_iou_model_saved: {best_iou}')
            


