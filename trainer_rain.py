'''
Author: DC
Date: 2024-09-02 15:04:05
LastEditTime: 2025-01-13 16:09:04
LastEditors: DC
Description: 
FilePath: /swin_unet1/trainer_rain.py
Never lose my passion
'''

# from utils_potsdam import test_single_volume

from torchvision import transforms
# from utils_potsdam import DiceLoss
# from utils_potsdam import lovasz_softmax
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from tensorboardX import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch
import numpy as np
import os
import glob
from tqdm import tqdm
# import tifffile
import time
import sys
import random
import argparse
import logging


def inference2(args, model, test_save_path=None):
    from datasets.dataset_bio import bio_dataset, RandomGenerator
    db_test = bio_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="test",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    # db_test = args.Dataset(base_dir='/mnt/e/BaiduNetdiskDownload/Dataset',
    #                        split="test", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1,
                            shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()

    NUM_CATEGORIES = 5
    tp = np.zeros(NUM_CATEGORIES)
    fp = np.zeros(NUM_CATEGORIES)
    fn = np.zeros(NUM_CATEGORIES)
    tn = np.zeros(NUM_CATEGORIES)
    Time = 0
    # for i_batch, sampled_batch in tqdm(enumerate(testloader)):
    #      None
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):

        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        image_batch, label_batch = image_batch.cpu(), label_batch.cpu()
        image_batch = image_batch.cpu()
        image_batch = image_batch.numpy()
        image_batch = image_batch.transpose([0, 3, 1, 2])
        # print('image_batch', type(image_batch))
        # print('label_batch', label_batch.shape)
        image_batch = torch.from_numpy(image_batch)
        image_batch = image_batch.cpu()
        image, label, case_name = image_batch, label_batch, sampled_batch['case_name'][0]
        try:
            prediction, time = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[
                                                args.img_size, args.img_size], test_save_path=None, case=case_name, z_spacing=1)
            Time = time + Time
            label_batch = label_batch.cpu()
            label = label_batch.numpy()
            for cat in range(NUM_CATEGORIES):
                # tp[cat] += ((prediction == cat) & (label == cat) & (label < NUM_CATEGORIES)).sum()
                # fp[cat] += ((prediction == cat) & (label != cat) & (label < NUM_CATEGORIES)).sum()
                # fn[cat] += ((prediction != cat) & (label == cat) & (label < NUM_CATEGORIES)).sum()

                tp[cat] += ((prediction == cat) & (label == cat)
                            & (label < NUM_CATEGORIES)).sum()
                fp[cat] += ((prediction == cat) & (label != cat)
                            & (label < NUM_CATEGORIES)).sum()
                fn[cat] += ((prediction != cat) & (label == cat)
                            & (label < NUM_CATEGORIES)).sum()
                tn[cat] += ((prediction != cat) & (label != cat)
                            & (label < NUM_CATEGORIES)).sum()
        except Exception as e:
            continue
        # accumulate statistics for IOU-3

        # compute IOU-3
    nfiles = len(testloader)
    logging.info('Generated segmentations in %s ms/per -- %s FPS' %
          (Time / nfiles * 1000, nfiles / Time))
    logging.info('Generated segmentations in %s seconds' % (Time))
    np.seterr(divide='ignore', invalid='ignore')
    iou = np.divide(tp, tp + fp + fn)

    pre = np.divide(tp, (tp + fp))
    recall = np.divide(tp, (tp + fn))
    f1 = np.divide(2 * pre * recall, (pre + recall))
    acc = np.divide((tp+tn).sum(), (tp + fn+fp+tn).sum())
    logging.info('---------------------------------------------------')
    logging.info('IOU:  '+ str(iou))
    logging.info('mIOU: '+str(iou.mean()))
    logging.info('---------------------------------------------------')
    # print('recall',recall)
    logging.info('F1'+ str(f1))
    logging.info('Ave.F1'+ str(f1.mean()))
    logging.info('Acc'+ str(acc))
    logging.info('---------------------------------------------------')

    return iou.mean()


def cal(pred, label):

    NUM_CATEGORIES = 5
    tp = np.zeros(NUM_CATEGORIES)
    fp = np.zeros(NUM_CATEGORIES)
    fn = np.zeros(NUM_CATEGORIES)
    out = torch.argmax(torch.softmax(pred, dim=1), dim=1).squeeze(0)

    prediction = out.cpu().detach().numpy()

    label = label.cpu()
    label = label.numpy()
    for cat in range(NUM_CATEGORIES):
        tp[cat] += ((prediction == cat) & (label == cat)
                    & (label < NUM_CATEGORIES)).sum()
        fp[cat] += ((prediction == cat) & (label != cat)
                    & (label < NUM_CATEGORIES)).sum()
        fn[cat] += ((prediction != cat) & (label == cat)
                    & (label < NUM_CATEGORIES)).sum()

    np.seterr(divide='ignore', invalid='ignore')
    iou = np.divide(tp, tp + fp + fn)

    m = iou.mean()
    return m


def test_bio1(args,model,writer,sum_num):
    from datasets.dataset_rain import rain_dataset, RandomGenerator
    db_train = rain_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="test")
    batch_size = args.batch_size * args.n_gpu
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False,
                             worker_init_fn=worker_init_fn)
    loss1=nn.MSELoss()
    sum_loss=0
    day=1
    # sum_num=0
    # iterator = tqdm(range(max_epoch), ncols=70)
    for i_batch, sampled_batch in tqdm(enumerate(trainloader)):
            # print('iter_num',iter_num)
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            
            image_batch, label_batch = image_batch.float().cuda(), label_batch.float().cuda()

            image_batch = image_batch.cpu()

            image_batch = image_batch.numpy()
            image_batch = image_batch.transpose([0, 3, 1, 2])

            image_batch = torch.from_numpy(image_batch)
            # image_batch = image_batch.float().cuda()

            label_batch = label_batch.cpu()

            label_batch = label_batch.numpy()
            label_batch = label_batch.transpose([0, 3, 1, 2])

            label_batch = torch.from_numpy(label_batch)
            
            # label_batch = label_batch.float().cuda()
            model = model.cpu()
            outputs = model(image_batch)
            daily_loss = loss1(outputs, label_batch).sum().item()
            sum_loss+=daily_loss
            # if day%1==0:
            #     print(daily_loss)
            writer.add_scalar('info/test_daily_loss', daily_loss,day)
            day+=1
    sum_num+=1
    writer.add_scalar('info/sum_loss', sum_loss,sum_num)
    model = model.cuda()
    return sum_loss

def trainer_rain1(args, model, snapshot_path):
    from datasets.dataset_rain import rain_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    # print(args.img_size)
    db_train = rain_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train")
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    # print(batch_size)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    # fen hao zu le 758iterations per epoch 758ge batch
    # if args.n_gpu > 1:
    #    model = nn.DataParallel(model)
    model.train()
    loss1=nn.L1Loss()

    optimizer = optim.AdamW(model.parameters(), )  # yohua SGD

    writer = SummaryWriter(snapshot_path + '/log')
    '''
    snapshot ='x'
    checkpoint = torch.load(snapshot, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = 80
    print('load epoch{} succeed!'.format(start_epoch))
    '''
    # print(snapshot_path)
    iter_num = 0
    max_epoch = args.max_epochs

    max_iterations = args.max_epochs * len(trainloader)
    # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(
        len(trainloader), max_iterations))
    best_performance = 0.0
    print(max_epoch)

    iterator = tqdm(range(max_epoch), ncols=70)
    loss=0
    sum_loss_num=0
    scaler = torch.GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    for epoch_num in iterator:

        epoch_loss = 0
        if epoch_num %1 ==0 and epoch_num!=0:
            model.eval()
            sum_loss=test_bio1(args,model,writer,sum_loss_num)
            sum_loss_num+=1
            scheduler.step(sum_loss)
        model.train()
        
        for i_batch, sampled_batch in enumerate(trainloader):
            # print('iter_num',iter_num)
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            
            image_batch, label_batch = image_batch.float().cuda(), label_batch.float().cuda()
            image_batch = image_batch.cpu()
            image_batch = image_batch.numpy()
            image_batch = image_batch.transpose([0, 3, 1, 2])

            image_batch = torch.from_numpy(image_batch)
            image_batch = image_batch.float().cuda()

            label_batch = label_batch.cpu()

            label_batch = label_batch.numpy()
            label_batch = label_batch.transpose([0, 3, 1, 2])

            label_batch = torch.from_numpy(label_batch)
            label_batch = label_batch.float().cuda()
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                outputs = model(image_batch)
                # print(outputs.shape)
                # print(label_batch.shape)
                # print()
                loss = loss1(outputs, label_batch).sum()
            scale_loss=scaler.scale(loss)
            # print(scale_loss)
            # scaler.scale(loss).backward()
            scale_loss.backward()
            scaler.step(optimizer)
            scaler.update()

            # outputs = model(image_batch)
            # loss = loss1(outputs, image_batch).sum()
            # loss.backward()
            # optimizer.step()

            
            # if iter_num % 10 == 0:
            #     loss.backward()
            #     optimizer.step()
            #     loss=0
            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            
            for param_group in optimizer.param_groups:
                 lr_=param_group['lr']

            # lr = optimizer.param_groups[0]['lr']
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            # print(scale_loss)
            writer.add_scalar('info/total_loss', scale_loss.item(), iter_num)
            # writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            # writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            # writer.add_scalar('info/miou', miou, iter_num)
            if iter_num% 100 ==0:
            # if iter_num% 1 ==0:

                logging.info('iteration %d : lr : %f, loss : %f' % (
                    iter_num, lr_, scale_loss.item() ))
        if epoch_num%1==0:
            # print(image_batch.shape)
            # print(outputs.shape)
            for i in range(outputs.shape[1]):
                pred1=outputs[0,i,:,:].unsqueeze(0)
                image1=image_batch[0,i,:,:].unsqueeze(0)
                image2=label_batch[0,i,:,:].unsqueeze(0)
                image1 = (image1 - image1.min()) / (image1.max() - image1.min())
                image2 = (image2 - image2.min()) / (image2.max() - image2.min())
                pred1 = (pred1 - pred1.min()) / (pred1.max() - pred1.min())
                writer.add_image('train/image1_%d'%(i), image1*255, iter_num)
                writer.add_image('train/image2_%d'%(i), image2*255, iter_num)
                writer.add_image('train/pred1_%d'%(i), pred1*255, iter_num)
            save_mode_path = os.path.join(snapshot_path, 'last_model.pth')
            torch.save(model.state_dict(), save_mode_path)
        

    writer.close()
    return "Training Finished!"