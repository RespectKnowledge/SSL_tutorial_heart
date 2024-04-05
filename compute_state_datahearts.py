# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:48:52 2024

@author: aq22
"""

################################# Compute voxel spacing, mean and max values ##########

#%% dataset preprocessing codes ##################################

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

#%% ####################### compute spacing,min,max values in complete datasets
import monai
import os
import glob
from monai.transforms import (
    Compose,
    LoadImaged,   
)
from monai.data import (
    Dataset,
    DatasetSummary,
    load_decathlon_datalist,
    load_decathlon_properties,
)
####################### dataset from datalist ###############################
dataset = Dataset(data=train_files,transform=LoadImaged(keys=["image", "label"]),)
############################## compute statictics of dataset ####################
calculator = DatasetSummary(dataset)
target_spacing = calculator.get_target_spacing()
print("spacing: ", target_spacing)
#this function also prints the mean and std values (used for normalization), 
calculator.calculate_statistics()
print("mean: ", calculator.data_mean, " std: ", calculator.data_std)
calculator.calculate_percentiles(sampling_flag=True, interval=10, min_percentile=0.5, max_percentile=99.5)
print("min: ",calculator.data_min_percentile," max: ",calculator.data_max_percentile,)
############################## computed from all dataset ##########################
#spacing:  (1.2499995173568585, 1.2499998886073935, 9.599999507298296)
#mean:  224.2091827392578  std:  192.90354919433594