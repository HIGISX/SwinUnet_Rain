'''
Author: DC
Date: 2025-01-10 14:49:59
LastEditTime: 2025-01-13 16:07:56
LastEditors: DC
Description: 
FilePath: /swin_unet1/datasets/dataset_rain.py
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
    def __init__(self, output_size1,output_size2):
        self.output_size1 = output_size1
        self.output_size2 = output_size2

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
        if x != self.output_size1[0] or y != self.output_size1[1]:
            # print(x)
            # print(self.output_size[0])
            # print(self.output_size[0] / x)
            # why not 3?
            image = zoom(
                image, (self.output_size1[0] / x, self.output_size1[1] / y, 1), order=3)
        if x != self.output_size2[0] or y != self.output_size2[1]:
            label = zoom(
                label, (self.output_size2[0] / x, self.output_size2[1] / y, 1), order=3)
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


class rain_dataset(Dataset):
    def __init__(self, base_dir,list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.data_dir = base_dir
        self.sample_list_cwv = self.get_sample_list("cwv")
        self.sample_list_geo_200 = self.get_sample_list("geopotential/200")
        self.sample_list_geo_850 = self.get_sample_list("geopotential/850")
        self.sample_list_u_200 = self.get_sample_list("u/200")
        self.sample_list_u_850 = self.get_sample_list("u/850")
        self.sample_list_v_200 = self.get_sample_list("v/200")
        self.sample_list_v_850 = self.get_sample_list("v/850")
        self.label_list_extreme = self.get_sample_list("label2")
        self.label_list_pre = self.get_sample_list("label/precipitation")

    def get_sample_list(self, sub_dir):
        arr = []
        for root, dirs, files in os.walk(os.path.join(self.data_dir, sub_dir)):
            for file in files:
                p1 = os.path.join(root, file)
                if '2022' in p1 or '2001' in p1:
                    continue
                arr.append(p1)
        arr = sorted(arr)
        return arr

    def __len__(self):
        if self.split == "train":
            return len(self.sample_list_cwv[:len(self.sample_list_cwv)-365])
        else:
            return len(self.sample_list_cwv[len(self.sample_list_cwv)-365:])

    def __getitem__(self, idx):
        if self.split == "train":
            cwv=nc.Dataset(self.sample_list_cwv[idx], 'r')
            geo_200=nc.Dataset(self.sample_list_geo_200[idx], 'r')
            geo_850=nc.Dataset(self.sample_list_geo_850[idx], 'r')
            u_200=nc.Dataset(self.sample_list_u_200[idx], 'r')
            u_850=nc.Dataset(self.sample_list_u_850[idx], 'r')
            v_200=nc.Dataset(self.sample_list_v_200[idx], 'r')
            v_850=nc.Dataset(self.sample_list_v_850[idx], 'r')
            # label_extreme=nc.Dataset(self.label_list_extreme[idx], 'r')
            label_extreme_light=np.load(os.path.join(self.data_dir,'label2/light')+"/"+str(idx)+".npy")
            label_extreme_dark=np.load(os.path.join(self.data_dir,'label2/dark')+"/"+str(idx)+".npy")
            label_pre=nc.Dataset(self.label_list_pre[idx], 'r')
        else:
            None
            cwv=nc.Dataset(self.sample_list_cwv[len(self.sample_list_cwv)-365+idx], 'r')
            geo_200=nc.Dataset(self.sample_list_geo_200[len(self.sample_list_cwv)-365+idx], 'r')
            geo_850=nc.Dataset(self.sample_list_geo_850[len(self.sample_list_cwv)-365+idx], 'r')
            u_200=nc.Dataset(self.sample_list_u_200[len(self.sample_list_cwv)-365+idx], 'r')
            u_850=nc.Dataset(self.sample_list_u_850[len(self.sample_list_cwv)-365+idx], 'r')
            v_200=nc.Dataset(self.sample_list_v_200[len(self.sample_list_cwv)-365+idx], 'r')
            v_850=nc.Dataset(self.sample_list_v_850[len(self.sample_list_cwv)-365+idx], 'r')
            label_extreme_light=np.load(os.path.join(self.data_dir,'label2/light')+"/"+str(len(self.sample_list_cwv)-365+idx)+".npy")
            label_extreme_dark=np.load(os.path.join(self.data_dir,'label2/dark')+"/"+str(len(self.sample_list_cwv)-365+idx)+".npy")
            label_pre=nc.Dataset(self.label_list_pre[len(self.sample_list_cwv)-365+idx], 'r')
            


        
        cwv_light = np.array(cwv['lightmeanwatervapour'], copy=True)
        cwv_dark = np.array(cwv['darkmeanwatervapour'], copy=True)
        geo_200_light = np.array(geo_200['z_light'], copy=True)
        geo_200_dark = np.array(geo_200['z_dark'], copy=True)
        geo_850_light = np.array(geo_850['z_light'], copy=True)
        geo_850_dark = np.array(geo_850['z_dark'], copy=True)
        u_200_light = np.array(u_200['u_light'], copy=True)
        u_200_dark = np.array(u_200['u_dark'], copy=True)
        u_850_light = np.array(u_850['u_light'], copy=True)
        u_850_dark = np.array(u_850['u_dark'], copy=True)
        v_200_light = np.array(v_200['v_light'], copy=True)
        v_200_dark = np.array(v_200['v_dark'], copy=True)
        v_850_light = np.array(v_850['v_light'], copy=True)
        v_850_dark = np.array(v_850['v_dark'], copy=True)



        # print(cwv_light.shape)
        # print(cwv_dark.shape)
        # print(geo_200_light.shape)
        # print(geo_200_dark.shape)
        # print(geo_850_light.shape)
        # print(geo_850_dark.shape)
        # print(u_200_light.shape)
        # print(u_200_dark.shape)
        # print(u_850_light.shape)
        # print(u_850_dark.shape)
        # print(v_200_light.shape)
        # print(v_200_dark.shape)
        # print(v_850_light.shape)
        # print(v_850_dark.shape)

        cwv_light = zoom(cwv_light, (480 / cwv_light.shape[0], 1440 / cwv_light.shape[1]), order=3)
        cwv_dark = zoom(cwv_dark, (480 / cwv_dark.shape[0], 1440 / cwv_dark.shape[1]), order=3)
        geo_200_light = zoom(geo_200_light, (480 / geo_200_light.shape[0], 1440 / geo_200_light.shape[1]), order=3)
        geo_200_dark = zoom(geo_200_dark, (480 / geo_200_dark.shape[0], 1440 / geo_200_dark.shape[1]), order=3)
        geo_850_light = zoom(geo_850_light, (480 / geo_850_light.shape[0], 1440 / geo_850_light.shape[1]), order=3)
        geo_850_dark = zoom(geo_850_dark, (480 / geo_850_dark.shape[0], 1440 / geo_850_dark.shape[1]), order=3)
        u_200_light = zoom(u_200_light, (480 / u_200_light.shape[0], 1440 / u_200_light.shape[1]), order=3)
        u_200_dark = zoom(u_200_dark, (480 / u_200_dark.shape[0], 1440 / u_200_dark.shape[1]), order=3)
        u_850_light = zoom(u_850_light, (480 / u_850_light.shape[0], 1440 / u_850_light.shape[1]), order=3)
        u_850_dark = zoom(u_850_dark, (480 / u_850_dark.shape[0], 1440 / u_850_dark.shape[1]), order=3)
        v_200_light = zoom(v_200_light, (480 / v_200_light.shape[0], 1440 / v_200_light.shape[1]), order=3)
        v_200_dark = zoom(v_200_dark, (480 / v_200_dark.shape[0], 1440 / v_200_dark.shape[1]), order=3)
        v_850_light = zoom(v_850_light, (480 / v_850_light.shape[0], 1440 / v_850_light.shape[1]), order=3)
        v_850_dark = zoom(v_850_dark, (480 / v_850_dark.shape[0], 1440 / v_850_dark.shape[1]), order=3)

        image = np.stack([cwv_light, cwv_dark, geo_200_light, geo_200_dark, geo_850_light, geo_850_dark, 
                  u_200_light, u_200_dark, u_850_light, u_850_dark, v_200_light, v_200_dark, 
                  v_850_light, v_850_dark], axis=2)

        
        # print("======================================================")
        # print(np.array(cwv['darkmeanwatervapour']).shape)
        # print(np.array(geo_200['z_light']).shape)
        # print(np.array(geo_200['z_dark']).shape)
        # print(np.array(geo_850['z_light']).shape)
        # print(np.array(geo_850['z_dark']).shape)
        # print(np.array(u_200['u_light']).shape)
        # print(np.array(u_200['u_dark']).shape)
        # print(np.array(u_850['u_dark']).shape)
        # print(np.array(u_850['u_light']).shape)
        # print(np.array(v_200['v_light']).shape)
        # print(np.array(v_200['v_dark']).shape)
        # print(np.array(v_850['v_dark']).shape)
        # print(np.array(v_850['v_light']).shape)
        # print("####======================================================")
        
        # label_extreme_light = np.swapaxes(label_extreme_light, 0, 1)
        # label_extreme_dark = np.swapaxes(label_extreme_dark, 0, 1)
        # light_hourly_precip_rate = np.swapaxes(np.array(label_pre['lightHourlyPrecipRate']), 0, 1)
        # dark_hourly_precip_rate = np.swapaxes(np.array(label_pre['darkHourlyPrecipRate']), 0, 1)
        light_hourly_precip_rate=label_pre['lightHourlyPrecipRate']
        dark_hourly_precip_rate=label_pre['darkHourlyPrecipRate']

        
        label = np.stack([label_extreme_light, label_extreme_dark, light_hourly_precip_rate, dark_hourly_precip_rate], axis=2)

        cwv.close()
        geo_200.close()
        geo_850.close()
        u_200.close()
        u_850.close()
        v_200.close()
        v_850.close()
        label_pre.close()
        

        # print(image.shape)
        # print()
        # print(label.shape)
        # print()
        # # print(image.sum())
        # # print(label.sum())
        # print()

        image[np.isnan(image)] = 0
        label[np.isnan(label)] = 0
        sample = {'image': image, 'label': label}

        # if self.transform:
        #     sample = self.transform(sample)
        if self.split == "train":
            sample['case_name'] = self.sample_list_cwv[idx].strip('\n')
        else:
            sample['case_name'] = self.sample_list_cwv[idx].strip('\n')
        return sample