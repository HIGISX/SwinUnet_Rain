'''
Author: DC
Date: 2024-11-21 16:23:06
LastEditTime: 2025-01-13 16:01:59
LastEditors: DC
Description: 
FilePath: /swin_unet1/train_rain.py
Never lose my passion
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import logging

import random
from networks.vision_transformer import SwinUnet as ViT_seg
import numpy as np
import torch

import torch.backends.cudnn as cudnn
# from networks.vit_seg_modeling import VisionTransformer as ViT_seg
# from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
# from lib.networks import MERIT_Cascaded
# from trainer_potsdam import trainer_synapse
# from torchviz import make_dot
from trainer_rain import trainer_rain1
from torchvision import models

from config import get_config
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Vai', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=6, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=1, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=(180,360), help='input patch size of network input')
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
# parser.add_argument("--dataset_name", default="datasets")
parser.add_argument("--n_class", default=4, type=int)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--eval_interval", default=1, type=int)

parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file', default=r"/home/dc/Documents/vscode/jiangyu/swin_unet/swin_unet1/configs/swin_tiny_patch4_window7_224_lite2.yaml")
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--att-type', type=str, choices=['BAM', 'CBAM'], default=None)
parser.add_argument('--volume_path', type=str,
                    default='/private/data/Vai256_npz/test_npz', help='root dir for validation volume data')
parser.add_argument('--test_save_dir', type=str,
                    default='../predictions', help='saving prediction as nii!')
args = parser.parse_args()
config = get_config(args)

if __name__ == "__main__":
    
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #args.vit_patches_size=16
    # args.img_size=256
    args.batch_size=1
    dataset_name = 'rain' #args.dataset
    dataset_config = {
        'Vai_256': {
            'root_path': '/private/hexin/data/Vai_256_npz/train_npz',
            'list_dir': './lists/lists_Vai_256',
            'num_classes': 6,
        },
        'Pots_256': {
            'root_path': '/mnt/d/dataset/newpotsdam3',
            'list_dir': './lists/lists_Pots_256',
            'num_classes': 6,
        },
        'bio': {
            'root_path': r'/home/dc/dc_data/co2_data/LowRes/Carbontracker/CT2022_flux/',
            'list_dir': './lists/lists_Pots_256',
            'num_classes': 1,
        },
        'rain': {
            'root_path': r'/home/dc/dc_data/jiangyu/',
            'list_dir': './lists/lists_Pots_256',
            'num_classes': 1,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = False #True
    args.exp = 'STUNet_' + dataset_name + str(args.img_size)
    snapshot_path = r"/home/dc/Documents/vscode/jiangyu/swin_unet/swin_unet1/snapshot_path2/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    print('-------------------------------------------')
    print(snapshot_path)
    print('------------------------------------------')
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # net = MERIT_Cascaded(n_class=args.num_classes, img_size_s2=(args.img_size,args.img_size), img_size_s1=(256,256), model_scale='small', decoder_aggregation='additive', interpolation='bilinear').cuda()
    net = ViT_seg(config, img_size=args.img_size, num_classes=1).cuda()

    trainer = {dataset_name: trainer_rain1}
    trainer[dataset_name](args, net, snapshot_path)