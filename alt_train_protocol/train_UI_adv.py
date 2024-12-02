from __future__ import print_function
import os
import logging
import time
import numpy as np
import random
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import shutil
from shutil import copyfile
from datetime import datetime
# from tensorboardX import SummaryWriter
from model_UI import WideResNet
# from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, Sampler
from utils import *
import glob
import torchvision.utils as vutils
import torch.autograd.profiler as profiler
from torchvision.utils import save_image
import pandas as pd
from collections import defaultdict
from datetime import timedelta


############### alternative training method! reconstructing from pixel noise
############### not reconstructing non uniform to uniform 


class CustomDataset(Dataset):
    '''
    Takes folder structure:

    data/
    └── class/
    |    └──  uniform/
    |        │   ├── nparray0
    |        |   ├── nparray1
    |        |   ...
    |        └── nonuniform/
    |            ├── nparray0
    |            ├── nparray1
    |            ...
    └── class/
    |   └── uniform/
    |       │   ├── nparray0
    |       |   ├── nparray1
    |       |   ...
    |       └── nonuniform/
    |           ├── nparray0
    |           ├── nparray1
    |           ...
    └── class/....
    
    This code gives a uniform or nonuniform target tensor + class tensors for each image
    and returns each image in the uniform or nonuniform folders.
    
    If TEST mode is enabled, it selects only one uniform and one nonuniform image per class.
    '''
    
    def __init__(self, data_dir, TEST=False):
        self.data_dir = data_dir
        self.TEST = TEST
        self.class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        self.samples = self._get_samples()
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_dirs)}
        print(self.class_to_idx)

    def _get_samples(self):
        samples = []
        for class_dir in self.class_dirs:
            class_path = os.path.join(self.data_dir, class_dir)
            uni_files = sorted(glob.glob(os.path.join(class_path, 'uniform', '*.npy')))
            nonuni_files = sorted(glob.glob(os.path.join(class_path, 'nonuniform', '*.npy')))
            
            if self.TEST:
                if len(uni_files) > 0:
                    samples.append((uni_files[0], class_dir, 1))  # 1 = uniform
                if len(nonuni_files) > 0:
                    samples.append((nonuni_files[0], class_dir, 0))  # 0 = nonuniform
            else:
                samples.extend([(file_path, class_dir, 1) for file_path in uni_files])
                samples.extend([(file_path, class_dir, 0) for file_path in nonuni_files])
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, class_name, is_uniform = self.samples[idx]
        img = np.loadtxt(file_path).astype(np.float32)
        img = torch.from_numpy(img).unsqueeze(0)

        target = torch.tensor(self.class_to_idx[class_name], dtype=torch.long)
        uni_target = torch.tensor([is_uniform], dtype=torch.float32)

        return img, target, uni_target

    def get_num_classes(self):
        return len(self.class_dirs)


def add_peripheral_noise(tensor):

    noise_levels = [40, 50, 60]
    weights = [1, 2, 1] 

    noise_level = random.choices(noise_levels, weights=weights, k=1)[0]

    device = tensor.device
    B, C, H, W = tensor.shape

    central_size = H // 2
    mid_size = (H - central_size) // 2
    outer_size = H - central_size + mid_size

    # # mask for centre
    # mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    # start = (H - central_size) // 2
    # end = start + central_size
    # mask[start:end, start:end] = 1

    # outer mask
    out_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    out_start = (H - outer_size) // 2
    out_end = out_start + outer_size
    out_mask[out_start:out_end, out_start:out_end] = 1
    # mid mask
    mid_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    mid_start = (H - mid_size) // 2
    mid_end = mid_start + mid_size
    mid_mask[mid_start:mid_end, mid_start:mid_end] = 1
    
    # mask = mask.unsqueeze(0).expand(C, -1, -1)  # to (C, H, W)
    out_mask = out_mask.unsqueeze(0).expand(C, -1, -1) 
    mid_mask = mid_mask.unsqueeze(0).expand(C, -1, -1) 
    
    binary_noise = torch.randint(0,2, (C, H, W), dtype=torch.float32, device=device)
    
    mid_noise_level = noise_level
    outer_noise_level = noise_level * 4

    mid_noise = binary_noise * mid_noise_level
    outer_noise = binary_noise * outer_noise_level
    
    noisy_tensor = tensor.clone()
    for i in range(B):
        noisy_tensor[i][~out_mask] += mid_noise[~out_mask]
        noisy_tensor[i][~mid_mask] += outer_noise[~mid_mask]
        
    return noisy_tensor


#############
### TRAIN ###
#############
def train_adv(args, model, device, train_loader, optimizer, uni_optimizer, epoch,
          cycles, mse_parameter, clean_parameter, clean='supclean', uni_adv='useadv'):
    
    model.train()

    # train loss for classes 
    train_loss = 0.0
    # the total loss for the uniformity classification task 
    uni_train_loss = 0.0

    train_data = {
        'epoch':[],
        'batch index': [],
        'class targets': [],
        'uni targets': [],
        # class preds
        'clean class predictions': [],
        'adv class predictions': [],
        # uni preds
        'clean uni predictions': [],
        'adv uni predictions': [],
    }

    # model.module.reset()
    model.reset()
    
    for batch_idx, (images, targets, uni_target) in enumerate(train_loader):
        optimizer.zero_grad()
        uni_optimizer.zero_grad()
        # SEE customdataset class to see how images and lables are processed: 
        # targets are only the feature type
        images, targets = images.to(device), targets.to(device)
        # uni nonuni targets are the uniformity label (uniform: 1 or not: 0)
        # if target is 1 then save as uni target, if 0 then save as nonuni target
        uni_target = uni_target.to(device)

        # # checks for corrupted data in batch 
        # for i in range(images.shape[0]):
        #     assert images[i].shape == torch.Size([1, 128, 128]), f"uni image {i} shape is not [1, 128, 128]"
        model.reset()

        # add peripheral noise to all images periphery as adversarial images
        adv_images = add_peripheral_noise(images)

        # concatenate uniform and non-uniform images for processing
        images_all = torch.cat((images, adv_images), 0)

        model.reset() 

        # FIRST FORWARD PASS
        logits, uni_fc_logits, orig_feature_all, \
        block1_all, block2_all, block3_all = model(images_all, first=True, inter=True)
        
        ff_prev = orig_feature_all

        # extract the features from clean images
        orig_feature, _ = torch.split(orig_feature_all, images.size(0))
        block1_clean, _ = torch.split(block1_all, images.size(0))
        block2_clean, _ = torch.split(block2_all, images.size(0))
        block3_clean, _ = torch.split(block3_all, images.size(0))

        # make a copy of it to save the forward pass image 
        orig_feature_b4_cycle = orig_feature
        ff_prev_b4_cycle = ff_prev

        # logits from the first forward pass
        logits_clean, logits_adv = torch.split(logits, images.size(0))

        if not ('no' in clean):
            loss = (clean_parameter * F.cross_entropy(logits_clean, targets) + F.cross_entropy(logits_adv, targets)) / (2*(cycles+1))
        else:        
            loss = F.cross_entropy(logits_adv, targets) / (cycles+1) 

        for i_cycle in range(cycles):
            recon, block1_recon, block2_recon, block3_recon = model(logits, step='backward', inter_recon=True)
            # split for clean and adversarial reconstructions from DGM
            recon_clean, recon_adv = torch.split(recon, images.size(0))
            recon_block1_clean, recon_block1_adv = torch.split(block1_recon, images.size(0))
            recon_block2_clean, recon_block2_adv = torch.split(block2_recon, images.size(0))
            recon_block3_clean, recon_block3_adv = torch.split(block3_recon, images.size(0))

            # reconstruction loss 
            loss += (F.mse_loss(recon_adv, orig_feature) 
                    + F.mse_loss(recon_block1_adv, block1_clean) 
                    + F.mse_loss(recon_block2_adv, block2_clean) 
                    + F.mse_loss(recon_block3_adv, block3_clean)) * mse_parameter / (4*cycles)

            # feedforward again 
            ff_current = ff_prev + args.res_parameter * (recon - ff_prev)

            if (epoch) % 2 == 0 or epoch == 1:
                epoch_dir = f"{args.model_dir}/recon_imgs/epoch_{epoch}"
                cycle_dir = os.path.join(epoch_dir, f"cycle_{i_cycle}")
                clean_dir = os.path.join(cycle_dir, "clean")
                adv_dir = os.path.join(cycle_dir, "adv")
                os.makedirs(clean_dir, exist_ok=True)
                os.makedirs(adv_dir, exist_ok=True)
            
                # input imgs saved for comparison
                clean_input_img = norm_t(images[0])
                adv_input_img = norm_t(adv_images[0])
                save_image(clean_input_img, os.path.join(clean_dir, "clean_input.png"))
                save_image(adv_input_img, os.path.join(adv_dir, "adv_input.png"))

                # ff current
                ff_current_img = norm_t(ff_current[0].mean(dim=0, keepdim=True))
                save_image(ff_current_img, os.path.join(cycle_dir,  f"ff_current.png"))
                # ff_prev
                ff_prev_img = norm_t(ff_prev[0].mean(dim=0, keepdim=True))
                save_image(ff_prev_img, os.path.join(cycle_dir,  f"ff_prev.png"))
                # ff_prev b4 cycle (otig feature all)
                ff_prevb4_cycle__img = norm_t(ff_prev_b4_cycle[0].mean(dim=0, keepdim=True))
                save_image(ff_prevb4_cycle__img, os.path.join(cycle_dir,  f"ff_prevb4_cycle__img.png"))       
                # original feedforward fature map
                orig_feature_img = norm_t(orig_feature_b4_cycle[0].mean(dim=0, keepdim=True))
                save_image(orig_feature_img, os.path.join(cycle_dir,  f"orig_feature.png"))

                # final layer recon images
                reconimg_clean = norm_t(recon_clean[0].mean(dim=0))
                reconimg_adv = norm_t(recon_adv[0].mean(dim=0))
                save_image(reconimg_clean, os.path.join(clean_dir, "final_clean.png"))
                save_image(reconimg_adv, os.path.join(adv_dir, "final_adv.png"))

                # block 1 reconstructions
                reconimgb1_clean = norm_t(recon_block1_clean[0].mean(dim=0))
                reconimgb1_adv = norm_t(recon_block1_adv[0].mean(dim=0))
                save_image(reconimgb1_clean, os.path.join(clean_dir, "b1_clean.png"))
                save_image(reconimgb1_adv, os.path.join(adv_dir, "b1_adv.png"))

                # block 2 reconstructions
                reconimgb2_clean = norm_t(recon_block2_clean[0].mean(dim=0))
                reconimgb2_adv = norm_t(recon_block2_adv[0].mean(dim=0))
                save_image(reconimgb2_clean, os.path.join(clean_dir, "b2_clean.png"))
                save_image(reconimgb2_adv, os.path.join(adv_dir, "b2_adv.png"))

                # block 3 reconstructions
                reconimgb3_clean = norm_t(recon_block3_clean[0].mean(dim=0))
                reconimgb3_adv = norm_t(recon_block3_adv[0].mean(dim=0))
                save_image(reconimgb3_clean, os.path.join(clean_dir, "b3_clean.png"))
                save_image(reconimgb3_adv, os.path.join(adv_dir, "b3_adv.png"))


            # the new logits are obtained, now this is no longer the first forward pass
            logits = model(ff_current, first=False)

            # the new ff previous weights now include information from reconstructions
            ff_prev = ff_current
            # get the new logits for clean and adv images 
            logits_clean, logits_adv = torch.split(logits, images.size(0)) 

            if not ('no' in clean):
                loss += (clean_parameter * F.cross_entropy(logits_clean, targets) 
                         + F.cross_entropy(logits_adv, targets)) / (2*(cycles+1))
            else:
                loss += F.cross_entropy(logits_adv, targets) / (cycles+1)

        ### UNIFORMITY LOSS
        criterion = nn.BCELoss()  # binary cross entropy
        # get logits for clean and adv images
        clean_uni_logits, adv_uni_logits = torch.split(uni_fc_logits, images.size(0))

        clean_uni_loss = criterion(clean_uni_logits, uni_target)
        adv_uni_loss = criterion(adv_uni_logits, uni_target)

        '''
        if uni_adv is set to include adv uniformity loss or NOT

        this will combine the clean and adv uni losses so the model learns 
        to rate uniformity of adv examples too 
        '''
        if not ('no' in uni_adv):
            uni_total_loss = clean_uni_loss + adv_uni_loss
        else:
            uni_total_loss = clean_uni_loss
        
        ### FEATURE CLASSIFICATION
        # feature predictions for clean images (uniform and nonuniform combined) 
        clean_class_pred = logits_clean.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        # feature predictions for adv images (uniform and nonuniform combined) 
        adv_class_pred = logits_adv.argmax(dim=1, keepdim=True)  
        
        ### UNIFORMITY CLASSIFICATION
        # uniformity predictions for clean images (uniform and nonuniform combined) 
        clean_uni_pred = (clean_uni_logits > 0.5).float()
        # uniformity predictions for adv images (uniform and nonuniform combined) 
        adv_uni_pred = (adv_uni_logits > 0.5).float()

        # save batch data for clean images
        train_data['epoch'].append(epoch)
        train_data['batch index'].append(batch_idx)
        train_data['class targets'].append(targets.cpu().numpy().tolist())
        train_data['uni targets'].append(uni_target.cpu().numpy().tolist())
        train_data['clean class predictions'].append(clean_class_pred.cpu().numpy().tolist())
        train_data['adv class predictions'].append(adv_class_pred.cpu().numpy().tolist())
        train_data['clean uni predictions'].append(clean_uni_pred.cpu().numpy().tolist())
        train_data['adv uni predictions'].append(adv_uni_pred.cpu().numpy().tolist())

        # update 
        # what happens if you swap them around? class loss first the uniformity loss
        uni_total_loss.backward(retain_graph=True)
        loss.backward(retain_graph=False)

        uni_optimizer.step()
        optimizer.step()

    # update loss for epoch 
    train_loss += loss.item()
    uni_train_loss += uni_total_loss.item()

    train_loss /= len(train_loader)
    uni_train_loss /= len(train_loader)

    return train_loss, uni_train_loss, train_data


############
### TEST ###
############
def test(args, model, device, test_loader, cycles, epoch):
    model.eval()

    clean_test_loss = 0.0
    adv_test_loss = 0.0
    uni_test_loss = 0.0

    test_data = {
        'epoch':[],
        'batch index': [],
        'class targets': [],
        'uni targets': [],
        # feature preds
        'clean class predictions': [],
        'adv class predictions': [],
        # uni preds
        'clean uni predictions': [],
        'adv uni predictions': [],
    }

    with torch.no_grad():
        for batch_idx, (images, targets, uni_target) in enumerate(test_loader):

            # targets are only the feature type
            images, targets = images.to(device), targets.to(device)
            # uni nonuni targets are the UNIFORMITY label (uniform: 1 or not: 0)
            uni_target = uni_target.to(device)

            adv_images = add_peripheral_noise(images)
            images_all = torch.cat((images, adv_images), 0)

            # model.module.reset()
            model.reset()

            logits, uni_fc_logits, orig_feature_all, _, _, _ = model(images_all, first=True, inter=True)
            ff_prev = orig_feature_all

            for i_cycle in range(cycles):
                recon, _, _, _ = model(logits, step='backward', inter_recon=True)
                ff_current = ff_prev + args.res_parameter * (recon - ff_prev)
                logits = model(ff_current, first=False)
                ff_prev = ff_current

            ########## CLASS CLASSIFICATION TASK
            # class logits
            logits_clean, logits_adv = torch.split(logits, images.size(0))
            # for class acuuracy 
            clean_test_loss += F.cross_entropy(logits_clean, targets, reduction='sum').item()
            adv_test_loss += F.cross_entropy(logits_adv, targets, reduction='sum').item()

            # class predictions
            clean_class_pred = logits_clean.argmax(dim=1, keepdim=True)
            adv_class_pred = logits_adv.argmax(dim=1, keepdim=True)

            ####### UNIFORMITY TASK 
            # uniformity logits 
            clean_uni_logits, adv_uni_logits = torch.split(uni_fc_logits, images.size(0))

            criterion = nn.BCELoss()
            clean_uni_loss = criterion(clean_uni_logits, uni_target)
            adv_uni_loss = criterion(adv_uni_logits, uni_target)
            # add to total uniformity loss 
            uni_test_loss += (clean_uni_loss + adv_uni_loss)

            clean_uni_pred = (clean_uni_logits > 0.5).float()
            adv_uni_pred = (adv_uni_logits > 0.5).float()

            test_data['epoch'].append(epoch)
            test_data['batch index'].append(batch_idx)
            test_data['class targets'].append(targets.cpu().numpy().tolist())
            test_data['uni targets'].append(uni_target.cpu().numpy().tolist())
            test_data['clean class predictions'].append(clean_class_pred.cpu().numpy().tolist())
            test_data['adv class predictions'].append(adv_class_pred.cpu().numpy().tolist())
            test_data['clean uni predictions'].append(clean_uni_pred.cpu().numpy().tolist())
            test_data['adv uni predictions'].append(adv_uni_pred.cpu().numpy().tolist())

        # update loss for epoch 
        clean_test_loss /= len(test_loader)
        adv_test_loss /= len(test_loader)
        uni_test_loss /= len(test_loader)

        return clean_test_loss, adv_test_loss, uni_test_loss, test_data

#####################
######## EXP ########
#####################
def exp(args, model, device, test_loader, cycles, epoch, folder):
    model.eval()

    class_losses = [0.0] * (cycles + 1)
    uni_losses = [0.0] * (cycles + 1)
    UI_cycle_losses = [0.0] * (cycles + 1)

    ## NOTE clean and adv uniform and nonuniform 
    exp_data = [{
        'epoch': [],
        'batch index': [],
        'cycle': [],
        'class targets': [],
        'uni targets': [],

        # outside AND inside cycles 
        'clean class predictions': [],
        'adv class predictions': [],

        # not in cycles
        'clean uni predictions': [],
        'adv uni predictions': [],
        'clean uni logits': [], 
        'adv uni logits': [],   

    } for _ in range(cycles + 1)]
    
    with torch.no_grad():
        for batch_idx, (images, targets, uni_target) in enumerate(test_loader):
            # targets are only the feature type
            images, targets = images.to(device), targets.to(device)
            # uni nonuni targets are the UNIFORMITY label (uniform: 1 or not: 0)
            uni_target = uni_target.to(device)

            adv_images = add_peripheral_noise(images)
            images_all = torch.cat((images, adv_images), 0)

            model.reset()

            output, uni_fc_logits, orig_feature, _, _, _ = model(images_all, first=True, inter=True)
            ff_prev = orig_feature

            ########## CLASS CLASSIFICATION TASK
            # class logits
            logits_clean, logits_adv = torch.split(output, images.size(0))
            # for class accuracy 
            clean_test_loss = F.cross_entropy(logits_clean, targets, reduction='sum').item()
            adv_test_loss = F.cross_entropy(logits_adv, targets, reduction='sum').item()
            class_losses[0] += (clean_test_loss + adv_test_loss)

            # class predictions
            clean_class_pred = logits_clean.argmax(dim=1, keepdim=True)
            adv_class_pred = logits_adv.argmax(dim=1, keepdim=True)

            ####### UNIFORMITY TASK 
            # uniformity logits
            clean_uni_logits, adv_uni_logits = torch.split(uni_fc_logits, images.size(0))

            criterion = nn.BCELoss()
            clean_uni_loss = criterion(clean_uni_logits, uni_target)
            adv_uni_loss = criterion(adv_uni_logits, uni_target)
            # add to total uniformity loss 
            uni_losses[0] += (clean_uni_loss + adv_uni_loss)
            UI_cycle_losses[0] = 666666666

            clean_uni_pred = (clean_uni_logits > 0.5).float()
            adv_uni_pred = (adv_uni_logits > 0.5).float()

            exp_data[0]['epoch'].append(epoch)
            exp_data[0]['batch index'].append(batch_idx)
            exp_data[0]['cycle'].append(0)
            exp_data[0]['class targets'].append(targets.cpu().numpy().tolist())
            exp_data[0]['uni targets'].append(uni_target.cpu().numpy().tolist())

            exp_data[0]['clean class predictions'].append(clean_class_pred.cpu().numpy().tolist())
            exp_data[0]['adv class predictions'].append(adv_class_pred.cpu().numpy().tolist())

            exp_data[0]['clean uni predictions'].append(clean_uni_pred.cpu().numpy().tolist())
            exp_data[0]['adv uni predictions'].append(adv_uni_pred.cpu().numpy().tolist())
            exp_data[0]['clean uni logits'].append(clean_uni_logits.cpu().numpy().tolist()) 
            exp_data[0]['adv uni logits'].append(adv_uni_logits.cpu().numpy().tolist())     

            for i_cycle in range(1, cycles + 1):
                # cycles start at 1 not 0 
                recon = model(output, step='backward')
                ff_current = ff_prev + args.res_parameter * (recon - ff_prev)
                # mean of channels in ff_current to use for uni logits within cycle
                ff_current_mean = ff_current.mean(dim=1, keepdim=True) 
                output = model(ff_current, first=False)
                ff_prev = ff_current

                ########## CLASS CLASSIFICATION TASK
                # class logits
                logits_clean, logits_adv = torch.split(output, images.size(0))
                # for class accuracy 
                clean_test_loss = F.cross_entropy(logits_clean, targets, reduction='sum').item()
                adv_test_loss = F.cross_entropy(logits_adv, targets, reduction='sum').item()
                class_losses[i_cycle] += (clean_test_loss + adv_test_loss)

                # class predictions
                clean_class_pred = logits_clean.argmax(dim=1, keepdim=True)
                adv_class_pred = logits_adv.argmax(dim=1, keepdim=True)

                ####### UNIFORMITY TASK 
                # uniformity logits 
                clean_uni_logits, adv_uni_logits = torch.split(uni_fc_logits, images.size(0))

                criterion = nn.BCELoss()
                clean_uni_loss = criterion(clean_uni_logits, uni_target)
                adv_uni_loss = criterion(adv_uni_logits, uni_target)
                # add to total uniformity loss
                uni_losses[i_cycle] += (clean_uni_loss + adv_uni_loss)

                clean_uni_pred = (clean_uni_logits > 0.5).float()
                adv_uni_pred = (adv_uni_logits > 0.5).float()

                # obtain UI task logits on reconstructed inputs
                # by passing ff_current back into the models first forward pass
                _, cycle_uni_fc_logits, _, _, _, _ = model(ff_current_mean, first=True, inter=True)
                
                # split for uni and nonuni images
                clean_cycle_logits, adv_cycle_logits = torch.split(cycle_uni_fc_logits, images.size(0))
                clean_cycle_loss = criterion(clean_cycle_logits, uni_target).item()
                adv_cycle_loss = criterion(adv_cycle_logits, uni_target).item()

                UI_cycle_losses[i_cycle] += (clean_cycle_loss + adv_cycle_loss)
                
                clean_cycle_uni_pred = (clean_cycle_logits > 0.5).float()
                adv_cycle_uni_pred = (adv_cycle_logits > 0.5).float()

                exp_data[i_cycle]['epoch'].append(epoch)
                exp_data[i_cycle]['batch index'].append(batch_idx)
                exp_data[i_cycle]['cycle'].append(i_cycle)
                exp_data[i_cycle]['class targets'].append(targets.cpu().numpy().tolist())
                exp_data[i_cycle]['uni targets'].append(uni_target.cpu().numpy().tolist())

                exp_data[i_cycle]['clean class predictions'].append(clean_class_pred.cpu().numpy().tolist())
                exp_data[i_cycle]['adv class predictions'].append(adv_class_pred.cpu().numpy().tolist())

                exp_data[i_cycle]['clean uni predictions'].append(clean_cycle_uni_pred.cpu().numpy().tolist())
                exp_data[i_cycle]['adv uni predictions'].append(adv_cycle_uni_pred.cpu().numpy().tolist())
                exp_data[i_cycle]['clean uni logits'].append(clean_cycle_logits.cpu().numpy().tolist()) 
                exp_data[i_cycle]['adv uni logits'].append(adv_cycle_logits.cpu().numpy().tolist())    

                all_cycle_test = True
                if all_cycle_test:
                    exp_cycle_dir = f"{args.model_dir}/exp_recon_arrays_{folder}/cycle_{i_cycle}"
                    batch_dir = os.path.join(exp_cycle_dir, str(f'batch_{batch_idx}'))
                    os.makedirs(batch_dir, exist_ok=True)
                    # for every image in the batch
                    for img_idx in range(args.batch_size):
                        # save sample, every 10th image
                        if img_idx % 10 == 0: 
                            img_dir = os.path.join(batch_dir, f"img_{img_idx}")
                            os.makedirs(img_dir, exist_ok=True)
                            np.save(os.path.join(img_dir, "clean_input.npy"), images[img_idx].cpu().numpy())
                            np.save(os.path.join(img_dir, "adv_input.npy"), adv_images[img_idx].cpu().numpy())
                            np.save(os.path.join(img_dir, "ff_current.npy"), ff_current[img_idx].cpu().numpy())
                            np.save(os.path.join(img_dir, "recon.npy"), recon[img_idx].cpu().numpy())

        for i_cycle in range(cycles + 1):
            class_losses[i_cycle] /= len(test_loader)
            uni_losses[i_cycle] /= len(test_loader)
            UI_cycle_losses[i_cycle] /= len(test_loader)

        return class_losses, uni_losses, UI_cycle_losses, exp_data


def main():
    parser = argparse.ArgumentParser(description='CNNF training')

    # disable cuda 
    parser.add_argument('--no-cuda', action='store_true', 
                        help='disables CUDA training')
    # num workers same as cpu cores 
    parser.add_argument('--n-workers', type=int, default=18)
    # data set
    parser.add_argument('--datadir', default='data/UI128', 
                        help='the dataset for training the model')

    # experimental data dir
    parser.add_argument('--exp-data-dir', default='exp_data', 
                        help='the dataset for running the experiment')
    # number fo cycles to run EXPERIMENT
    parser.add_argument('--exp-cycles', type=int, default=10, 
                        help='the number of cycles to run for the EXPERIMENT')

    # optimization parameters
    # batch size
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 128)')
    # val batch size 
    parser.add_argument('--val-batch-size', type=int, default=2, metavar='N',
                        help='input batch size for valing (default: 1000)')
    # epochs 
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train')
    # lr
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.05 for SGD)')
    # power 4 poly
    parser.add_argument('--power', type=float, default=0.9, metavar='LR',
                        help='learning rate for poly scheduling')
    # momentum
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    # weight decay
    parser.add_argument('--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
    # grad-clip
    parser.add_argument('--grad-clip', action='store_true', default=False,
                        help='enable gradient clipping')
    # scheduler
    parser.add_argument('--schedule', choices=['poly', 'cos', None],
                        default=None, help='scheduling for learning rate')
    # seed
    parser.add_argument('--seed', type=int, default=666, metavar='S',
                        help='random seed')
    
    # adversarial training parameters
    parser.add_argument('--clean', choices=['no', 'supclean'],
                        default='supclean', help='whether to use uniform data in nonuni training')
    
    # hyper-parameters
    # mse weight for reconstruction loss 
    parser.add_argument('--mse-parameter', type=float, default=1.0,
                        help='weight of the reconstruction loss')
    # weight for xentropy loss for uni imgs
    parser.add_argument('--clean-parameter', type=float, default=0.5,
                        help='weight of the clean Xentropy loss')
    # res step size 
    parser.add_argument('--res-parameter', type=float, default=0.1,
                        help='step size for residuals')
    
    # model parameters wideresnet
    parser.add_argument('--layers', default=40, type=int, help='total number of layers for WRN')
    parser.add_argument('--widen-factor', default=2, type=int, help='Widen factor for WRN')
    parser.add_argument('--droprate', default=0.0, type=float, help='Dropout probability')
    parser.add_argument('--ind', type=int, default=5,
                        help='index of the intermediate layer to reconstruct to')
    parser.add_argument('--train-cycles', type=int, default=2,
                        help='the maximum cycles that the CNN-F uses')
    
    # save model
    parser.add_argument('--model-name', default='TEST',
                        help='Name for Saving the current Model')
    parser.add_argument('--model-dir', default='TEST',
                        help='Directory for Saving the current Model')
    
    args = parser.parse_args()

    ### DIRS 
    if not os.path.exists(args.model_dir) and args.model_dir:
        os.makedirs(args.model_dir)
        # dir to save the reconstructed imgs and tensors to 
        os.makedirs(f'{args.model_dir}/recon_imgs', exist_ok=True)
     
    seed_torch(args.seed) 
    
    # logging config
    logging.basicConfig(filename=f'{args.model_dir}/log_{args.model_name}.txt', 
                        level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Arguments:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")

    # cuda
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        # available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        current_gpu = torch.cuda.current_device()
        print(f"Current GPU being used: {current_gpu}") 

    ### DIRS 
    if not os.path.exists(args.model_dir) and args.model_dir:
        os.makedirs(args.model_dir)
        # dir to save the reconstructed imgs and tensors to 
        os.makedirs(f'{args.model_dir}/recon_imgs', exist_ok=True)
     
    seed_torch(args.seed) 
    
    # logging config
    logging.basicConfig(filename=f'{args.model_dir}/log_{args.model_name}.txt', 
                        level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Arguments:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")

    # cuda
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        # available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        current_gpu = torch.cuda.current_device()
        print(f"Current GPU being used: {current_gpu}") 

    # num workers same as cores per gpu
    # kwargs = {'num_workers': 18, 'pin_memory': True} if use_cuda else {}
    
    # # train, test data dirs
    train_data = CustomDataset(data_dir=f"{args.datadir}/train")
    val_data = CustomDataset(data_dir=f"{args.datadir}/test")

    # # # ### FOR TESTING CODE
    # train_data = CustomDataset(data_dir=f"{args.datadir}/train", TEST=True)
    # val_data = CustomDataset(data_dir=f"{args.datadir}/test", TEST=True)

    num_classes = train_data.get_num_classes()
    print('n classes ', num_classes)
    ### NOTE: this will be half the uniform and nonuniform combined, because
    ### of how the custom dataset gets the matching images, 
    ### so expect half the data here
    print('len train uni and nonuni', len(train_data)*2)
    print('len val uni and nonuni', len(val_data)*2)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, 
        shuffle=True, num_workers=args.n_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.val_batch_size,
        shuffle=True, num_workers=args.n_workers, pin_memory=True)

    # config model 
    # change custom scale to match the size of the imgs (original model for 32x32)
    # imgs are 128x128 so scale = 4
    model = WideResNet(args.layers, num_classes, args.widen_factor, 
                       args.droprate, args.ind, args.train_cycles, 
                       args.res_parameter, custom_scale=4)

    model = model.to(device)

    # model = torch.nn.DataParallel(model)
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])

    optimizer = torch.optim.SGD(
          model.parameters(),
          args.lr,
          momentum=args.momentum,
          weight_decay=args.wd)

    uni_optimizer = optim.Adam(model.uniformity_fc.parameters(), lr=0.0001)

    # if(args.schedule == 'cos'):        
    #     scheduler = torch.optim.lr_scheduler.LambdaLR(
    #       optimizer, lr_lambda=lambda step: get_lr(step, args.epochs * len(train_loader), 1.0, 1e-5))
    # elif(args.schedule == 'poly'):
    #     scheduler = torch.optim.lr_scheduler.LambdaLR(
    #       optimizer, lr_lambda=lambda step: lr_poly(1.0, step, args.epochs * len(train_loader), args.power))

    # begin training
    best_overall_acc = 0.0
    epoch_train_times = []

    # storage for training, validation and experimental data
    train_df = pd.DataFrame(columns=[
        'epoch', 'batch index','class targets','uni targets',
        # class predications
        'clean class predictions','adv class predictions',
        # uni predictions
        'clean uni predictions','adv uni predictions'
    ])

    test_df = pd.DataFrame(columns=[
        'epoch', 'batch index', 'class targets', 'uni targets',
        # class predictions
        'clean class predictions', 'adv class predictions',
        # uniformity predictions
        'clean uni predictions', 'adv uni predictions'
    ])

    exp_df = pd.DataFrame(columns=[
        'data folder', 'epoch', 'batch index', 'cycle',

        'class targets', 'clean class predictions', 'adv class predictions',

        'uni targets', 'clean uni predictions', 'adv uni predictions',

        'clean uni logits', 'adv uni logits',
    ])

    train_losses = {
        'epoch': [],
        'class loss': [], 
        'uniformity loss': [] 
    }

    test_losses = {
        'epoch': [],
        'clean class loss': [], 
        'adv class loss': [], 
        'uniformity loss': []
    }

    exp_losses = {
        'epoch': [],
        'data folder': [],
        'class_losses': [], 
        'uni_losses': [],   
        'UI_cycle_losses': [] 
    }

    for epoch in range(1,args.epochs+1):
        # torch.cuda.empty_cache()

        # time start epoch:
        epoch_start = time.time()

        #############
        ### TRAIN ###
        #############
        class_train_loss, \
        uni_train_loss, \
        epoch_train_data = train_adv(args, model, device, train_loader, optimizer, uni_optimizer, epoch,
                            cycles=args.train_cycles, mse_parameter=args.mse_parameter, clean_parameter=args.clean_parameter, 
                            clean=args.clean, uni_adv='useadv')
        
        # time end epoch:
        epoch_end = time.time()
        epoch_train_time = epoch_end - epoch_start
        epoch_train_time_for = str(timedelta(seconds=epoch_train_time))
        epoch_train_times.append(epoch_train_time)

        # convert to list
        class_train_loss = tensor_to_list(class_train_loss)
        uni_train_loss = tensor_to_list(uni_train_loss)
        epoch_train_data = {key: tensor_to_list(value) for key, value in epoch_train_data.items()}

        # append training losses
        train_losses['epoch'].append(epoch)
        train_losses['class loss'].append(class_train_loss)
        train_losses['uniformity loss'].append(uni_train_loss)

        # save data to dataframes
        for batch_idx in range(len(epoch_train_data['class targets'])):
            train_batch_data = {
                'epoch': [epoch] * len(epoch_train_data['class targets'][batch_idx]),
                'batch index': [batch_idx] * len(epoch_train_data['class targets'][batch_idx]),
                'class targets': epoch_train_data['class targets'][batch_idx],
                'uni targets': [item[0] for item in epoch_train_data['uni targets'][batch_idx]],
                # feature predictions
                'clean class predictions': [item[0] for item in epoch_train_data['clean class predictions'][batch_idx]],
                'adv class predictions': [item[0] for item in epoch_train_data['adv class predictions'][batch_idx]],
                # uni predictions
                'clean uni predictions': [item[0] for item in epoch_train_data['clean uni predictions'][batch_idx]],
                'adv uni predictions': [item[0] for item in epoch_train_data['adv uni predictions'][batch_idx]],
            }
            train_df = pd.concat([train_df, pd.DataFrame(train_batch_data)], ignore_index=True)

        #############
        ### TEST  ###
        #############
        clean_test_loss, adv_test_loss, uni_test_loss, epoch_test_data = test(
            args, model, device, test_loader, cycles=args.train_cycles, epoch=epoch
        )

        # Convert to list
        clean_test_loss = tensor_to_list(clean_test_loss)
        adv_test_loss = tensor_to_list(adv_test_loss)
        uni_test_loss = tensor_to_list(uni_test_loss)
        epoch_test_data = {key: tensor_to_list(value) for key, value in epoch_test_data.items()}

        # Append testing losses
        test_losses['epoch'].append(epoch)
        test_losses['clean class loss'].append(clean_test_loss)
        test_losses['adv class loss'].append(adv_test_loss)
        test_losses['uniformity loss'].append(uni_test_loss)

        # Save data to DataFrame
        for batch_idx in range(len(epoch_test_data['class targets'])):
            test_batch_data = {
                'epoch': [epoch] * len(epoch_test_data['class targets'][batch_idx]),
                'batch index': [batch_idx] * len(epoch_test_data['class targets'][batch_idx]),
                'class targets': epoch_test_data['class targets'][batch_idx],
                'uni targets': [item[0] for item in epoch_test_data['uni targets'][batch_idx]],
                # Class predictions
                'clean class predictions': [item[0] for item in epoch_test_data['clean class predictions'][batch_idx]],
                'adv class predictions': [item[0] for item in epoch_test_data['adv class predictions'][batch_idx]],
                # Uniformity predictions
                'clean uni predictions': [item[0] for item in epoch_test_data['clean uni predictions'][batch_idx]],
                'adv uni predictions': [item[0] for item in epoch_test_data['adv uni predictions'][batch_idx]],
            }
        test_df = pd.concat([test_df, pd.DataFrame(test_batch_data)], ignore_index=True)

        #### ACCURACIES FOR LOG
        ## TRAIN
        train_clean_class_acc = calculate_accuracy(train_df[train_df['epoch'] == epoch]['class targets'], train_df[train_df['epoch'] == epoch]['clean class predictions'])
        train_adv_class_acc = calculate_accuracy(train_df[train_df['epoch'] == epoch]['class targets'], train_df[train_df['epoch'] == epoch]['adv class predictions'])
        train_clean_uni_acc = calculate_accuracy(train_df[train_df['epoch'] == epoch]['uni targets'], train_df[train_df['epoch'] == epoch]['clean uni predictions'])
        train_adv_uni_acc = calculate_accuracy(train_df[train_df['epoch'] == epoch]['uni targets'], train_df[train_df['epoch'] == epoch]['adv uni predictions'])
        ## TEST
        test_clean_class_acc = calculate_accuracy(test_df[test_df['epoch'] == epoch]['class targets'], test_df[test_df['epoch'] == epoch]['clean class predictions'])
        test_adv_class_acc = calculate_accuracy(test_df[test_df['epoch'] == epoch]['class targets'], test_df[test_df['epoch'] == epoch]['adv class predictions'])
        test_clean_uni_acc = calculate_accuracy(test_df[test_df['epoch'] == epoch]['uni targets'], test_df[test_df['epoch'] == epoch]['clean uni predictions'])
        test_adv_uni_acc = calculate_accuracy(test_df[test_df['epoch'] == epoch]['uni targets'], test_df[test_df['epoch'] == epoch]['adv uni predictions'])

        current_overall_acc = (
            test_clean_class_acc + test_adv_class_acc +
            test_clean_uni_acc + test_adv_uni_acc) / 4.0

        if current_overall_acc > best_overall_acc:
            best_overall_acc = current_overall_acc
            torch.save(model.state_dict(), os.path.join(
                args.model_dir, f"{args.model_name}_best_model.pt"
            ))
            logging.info("""
                            ,---.   .--.    .-''-.  .--.      .--.         _______       .-''-.     .-'''-. ,---------.  
                            |    \  |  |  .'_ _   \ |  |_     |  |        \  ____  \   .'_ _   \   / _     \\          \ 
                            |  ,  \ |  | / ( ` )   '| _( )_   |  |        | |    \ |  / ( ` )   ' (`' )/`--' `--.  ,---' 
                            |  |\_ \|  |. (_ o _)  ||(_ o _)  |  |        | |____/ / . (_ o _)  |(_ o _).       |   \    
                            |  _( )_\  ||  (_,_)___|| (_,_) \ |  |        |   _ _ '. |  (_,_)___| (_,_). '.     :_ _:    
                            | (_ o _)  |'  \   .---.|  |/    \|  |        |  ( ' )  \'  \   .---..---.  \  :    (_I_)    
                            |  (_,_)\  | \  `-'    /|  '  /\  `  |        | (_{;}_) | \  `-'    /\    `-'  |   (_(=)_)   
                            |  |    |  |  \       / |    /  \    |        |  (_,_)  /  \       /  \       /     (_I_)    
                            '--'    '--'   `'-..-'  `---'    `---`        /_______.'    `'-..-'    `-...-'      '---'    
                                                                                              
                         """)
            logging.info(f"New best model saved with overall accuracy: {best_overall_acc:.2f}")
            logging.info("New best model saved with overall accuracy")

        logging.info(f'𓍯𓂃𓏧♡ ₍ᐢ._.ᐢ₎♡ ༘ ꒰ᐢ. .ᐢ꒱ ₊˚⊹ *ੈ✩‧₊˚༺☆༻*ੈ✩‧₊˚')
        logging.info(f'')
        logging.info(f"Epoch {epoch} Results:")
        logging.info(f"~~~~~~~~~~~~~~~~~~~~~~~~~~")
        logging.info(f"Train Time: {epoch_train_time_for}")
        logging.info(f"♡")
        logging.info(f"Class Accuracy:")
        logging.info(f"  Clean - Train: {train_clean_class_acc:.2f} | Val: {test_clean_class_acc:.2f}")
        logging.info(f"  Adversarial - Train: {train_adv_class_acc:.2f} | Val: {test_adv_class_acc:.2f}")
        logging.info(f"  TOTAL - Train: {((train_adv_class_acc+train_clean_class_acc)/2):.2f} | Val: {((test_clean_class_acc+test_adv_class_acc)/2):.2f}")
        logging.info(f'✩')
        logging.info(f"Uniformity Accuracy:")
        logging.info(f"  Clean - Train: {train_clean_uni_acc:.2f} | Val: {test_clean_uni_acc:.2f}")
        logging.info(f"  Adversarial - Train: {train_adv_uni_acc:.2f} | Val: {test_adv_uni_acc:.2f}")
        logging.info(f"  TOTAL - Train: {((train_clean_uni_acc + train_adv_uni_acc)/2):.2f} | Val: {((test_clean_uni_acc + test_adv_uni_acc)/2):.2f}")
        logging.info(f'')
        logging.info(f"𖦹")
        logging.info(f'')

        ###############
        ##### EXP #####
        ###############
        exp_start_time = time.time()
        if epoch >= args.epochs-1:
            data_folders = [d for d in os.listdir(args.exp_data_dir) if os.path.isdir(os.path.join(args.exp_data_dir, d))]
            for folder in data_folders:
                print(f"Processing folder: {folder}")
                exp_data = CustomDataset(f"{args.exp_data_dir}/{folder}")

                exp_loader = torch.utils.data.DataLoader(
                    exp_data, batch_size=args.batch_size,
                    shuffle=True, num_workers=18, pin_memory=True,
                    drop_last=True
                )
                print(f"Length of exp loader for {folder}: {len(exp_loader.dataset)}")

                exp_class_losses, exp_uni_losses, exp_UI_cycle_losses, epoch_exp_data = exp(
                    args, model, device, exp_loader, cycles=args.exp_cycles, epoch=epoch, folder=folder
                )

                exp_rows = [] 

                for cycle_idx, cycle_data in enumerate(epoch_exp_data):
                    for i in range(len(cycle_data['batch index'])):
                        for j in range(len(cycle_data['class targets'][i])):

                            row_data = {
                                'data folder': folder,
                                'epoch': cycle_data['epoch'][i],
                                'batch index': cycle_data['batch index'][i],
                                'cycle': cycle_data['cycle'][i],
                                'class targets': cycle_data['class targets'][i][j],
                                'uni targets': cycle_data['uni targets'][i][j],
                                # Class predictions
                                'clean class predictions': cycle_data['clean class predictions'][i][j],
                                'adv class predictions': cycle_data['adv class predictions'][i][j],
                                # Uniformity predictions
                                'clean uni predictions': cycle_data['clean uni predictions'][i][j],
                                'adv uni predictions': cycle_data['adv uni predictions'][i][j],
                                # Logits
                                'clean uni logits': cycle_data['clean uni logits'][i][j],
                                'adv uni logits': cycle_data['adv uni logits'][i][j],
                            }
                            exp_rows.append(row_data)

                if exp_rows:
                    exp_rows_df = pd.DataFrame(exp_rows)
                    exp_df = pd.concat([exp_df, exp_rows_df], ignore_index=True)

                exp_losses['data folder'].append(folder)
                exp_losses['epoch'].append(epoch)
                exp_losses['class_losses'].append(exp_class_losses)
                exp_losses['uni_losses'].append(exp_uni_losses)
                exp_losses['UI_cycle_losses'].append(exp_UI_cycle_losses)
        
        exp_end_time = time.time()
        for_exp_time = str(timedelta(seconds=(exp_end_time - exp_start_time)))
        logging.info(f"Total Experiment Time: {for_exp_time}")
        
        ### SAVE DATA
        # losses to df
        train_losses_df = pd.DataFrame(train_losses)
        test_losses_df = pd.DataFrame(test_losses)
        exp_losses_df = pd.DataFrame(exp_losses)

        # train data
        train_data_dir = f'{args.model_dir}/train_data'
        os.makedirs(train_data_dir, exist_ok=True)
        train_df.to_csv(os.path.join(train_data_dir, 'train_data.csv'), index=False)
        train_losses_df.to_csv(os.path.join(train_data_dir, 'train_losses.csv'), index=False)

        # val data
        test_data_dir = f'{args.model_dir}/test_data'
        os.makedirs(test_data_dir, exist_ok=True)
        test_df.to_csv(os.path.join(test_data_dir, 'test_data.csv'), index=False)
        test_losses_df.to_csv(os.path.join(test_data_dir, 'test_losses.csv'), index=False)

        # experiment data
        if epoch >= args.epochs-1:
        # test_all_epochs = True
        # if test_all_epochs:
            exp_data_dir = f'{args.model_dir}/exp_data'
            os.makedirs(exp_data_dir, exist_ok=True)
            exp_df.to_csv(os.path.join(exp_data_dir, 'exp_data.csv'), index=False)
            exp_losses_df.to_csv(os.path.join(exp_data_dir, 'exp_losses.csv'), index=False)

    # if epoch >= 10:
    #     torch.save(model.state_dict(), args.model_dir + f"/{args.model_name}{epoch}_model.pt")

if __name__ == '__main__':
    main()
