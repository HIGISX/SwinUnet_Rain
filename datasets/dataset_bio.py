'''
Author: DC
Date: 2024-11-13 16:20:07
LastEditTime: 2024-12-28 11:03:20
LastEditors: DC
Description: 
FilePath: /swin_unet_co2_bio180360_6var_train_test_dataloader/datasets/dataset_bio.py
Never lose my passion
'''

import os
import random
import netCDF4 as nc
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


import torchvision.transforms as transforms

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        '''
        DataAugment
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        '''
        # print(image.shape)
        x, y, z = image.shape
        # print(image.sum())
        # print(label.sum())
        # print()
        if x != self.output_size[0] or y != self.output_size[1]:
            # print(x)
            # print(self.output_size[0])
            # print(self.output_size[0] / x)
            # why not 3?
            image = zoom(
                image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
            label = zoom(
                label, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
        # print(image.sum())
        # print(label.sum())
        # print()
        image = torch.from_numpy(image.astype(np.float32))  # .unsqueeze(0)
        # print(image.sum())
        # print(label.sum())
        # print()
        label = torch.from_numpy(label.astype(np.float32))
        
        
        sample = {'image': image, 'label': label}
        # print(image.sum())
        # print(label.sum())
        # print()
        return sample


class bio_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split

        self.data_dir = base_dir
        self.sample_list,self.test_list = self.get_sample_list()
        # print(self.sample_list[0])
        # print("-----")
        # print(self.label_list[0])
        # print()
            

    def get_sample_list(self):

        sample_list = []
        test_list=[]

        for filename in os.listdir(self.data_dir):
            if "2019" in filename:
                
                test_list.append(filename)
            else:
                sample_list.append(filename)
        # print(sample_list)
        return sample_list,test_list

    def __len__(self):
        if self.split == "train":
            return len(self.sample_list)-1
        else:
            return len(self.test_list)-1

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            slice_name_label = self.sample_list[idx+1].strip('\n')

            train_dataset = nc.Dataset(os.path.join(self.data_dir,slice_name), 'r')

            bio_image = np.array(train_dataset.variables['bio_flux_opt'][:])
            fire_image = np.array(train_dataset.variables['fire_flux_imp'][:])
            fossil_image = np.array(train_dataset.variables['fossil_flux_imp'][:])
            ocn_image = np.array(train_dataset.variables['ocn_flux_opt'][:])


            # # Normalize the variables
            bio_image = (bio_image - np.min(bio_image)) / (np.max(bio_image) - np.min(bio_image))
            fire_image = (fire_image - np.min(fire_image)) / (np.max(fire_image) - np.min(fire_image))
            fossil_image = (fossil_image - np.min(fossil_image)) / (np.max(fossil_image) - np.min(fossil_image))
            ocn_image = (ocn_image - np.min(ocn_image)) / (np.max(ocn_image) - np.min(ocn_image))

            
            
            label_dataset = nc.Dataset(os.path.join(self.data_dir,slice_name_label), 'r')

            bio_label = np.array(label_dataset.variables['bio_flux_opt'][:])
            fire_label = np.array(label_dataset.variables['fire_flux_imp'][:])
            fossil_label = np.array(label_dataset.variables['fossil_flux_imp'][:])
            ocn_label = np.array(label_dataset.variables['ocn_flux_opt'][:])


            # Normalize the labels
            bio_label = (bio_label - np.min(bio_label)) / (np.max(bio_label) - np.min(bio_label))
            fire_label = (fire_label - np.min(fire_label)) / (np.max(fire_label) - np.min(fire_label))
            fossil_label = (fossil_label - np.min(fossil_label)) / (np.max(fossil_label) - np.min(fossil_label))
            ocn_label = (ocn_label - np.min(ocn_label)) / (np.max(ocn_label) - np.min(ocn_label))
        else:
            slice_name = self.test_list[idx].strip('\n')
            slice_name_label = self.test_list[idx+1].strip('\n')

            train_dataset = nc.Dataset(os.path.join(self.data_dir,slice_name), 'r')

            bio_image = np.array(train_dataset.variables['bio_flux_opt'][:])
            fire_image = np.array(train_dataset.variables['fire_flux_imp'][:])
            fossil_image = np.array(train_dataset.variables['fossil_flux_imp'][:])
            ocn_image = np.array(train_dataset.variables['ocn_flux_opt'][:])


            # # Normalize the variables
            bio_image = (bio_image - np.min(bio_image)) / (np.max(bio_image) - np.min(bio_image))
            fire_image = (fire_image - np.min(fire_image)) / (np.max(fire_image) - np.min(fire_image))
            fossil_image = (fossil_image - np.min(fossil_image)) / (np.max(fossil_image) - np.min(fossil_image))
            ocn_image = (ocn_image - np.min(ocn_image)) / (np.max(ocn_image) - np.min(ocn_image))

            
            
            label_dataset = nc.Dataset(os.path.join(self.data_dir,slice_name_label), 'r')

            bio_label = np.array(label_dataset.variables['bio_flux_opt'][:])
            fire_label = np.array(label_dataset.variables['fire_flux_imp'][:])
            fossil_label = np.array(label_dataset.variables['fossil_flux_imp'][:])
            ocn_label = np.array(label_dataset.variables['ocn_flux_opt'][:])


            # Normalize the labels
            bio_label = (bio_label - np.min(bio_label)) / (np.max(bio_label) - np.min(bio_label))
            fire_label = (fire_label - np.min(fire_label)) / (np.max(fire_label) - np.min(fire_label))
            fossil_label = (fossil_label - np.min(fossil_label)) / (np.max(fossil_label) - np.min(fossil_label))
            ocn_label = (ocn_label - np.min(ocn_label)) / (np.max(ocn_label) - np.min(ocn_label))


        bio_image = bio_image[..., np.newaxis]
        fire_image = fire_image[..., np.newaxis]
        fossil_image = fossil_image[..., np.newaxis]
        ocn_image = ocn_image[..., np.newaxis]
        
        bio_label = bio_label[..., np.newaxis]
        fire_label = fire_label[..., np.newaxis]
        fossil_label = fossil_label[..., np.newaxis]
        ocn_label = ocn_label[..., np.newaxis]
        image = np.concatenate((bio_image, fire_image, fossil_image, ocn_image), axis=-1)
        label = np.concatenate((bio_label, fire_label, fossil_label, ocn_label), axis=-1)
        # print(image.shape)
        # print(label.shape)
        # image=image[:,:,np.newaxis]
        # label=label[:,:,np.newaxis]
        sample = {'image': image, 'label': label}

        # print(image.shape)
        # print()
        # print(label.shape)
        # print(image.sum())
        # print(label.sum())
        # print()

        # if self.transform:
        #     sample = self.transform(sample)
        if self.split == "train":
            sample['case_name'] = self.sample_list[idx].strip('\n')
        else:
            sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample