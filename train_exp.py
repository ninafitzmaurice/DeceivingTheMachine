from __future__ import print_function
import os
import logging
import time
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from tensorboardX import SummaryWriter
from model_UI import WideResNet
from torch.utils.data import Dataset
from utils import *
import glob
from torchvision.utils import save_image
import pandas as pd
from datetime import timedelta

########### THIS IS THE TRAINING TESTING AND EXP CODE I USED FOR MY REPORT

#### remeber the model had issues saving and re-loading, so I run the experiment for certain epochs
## this way you can also see how the UI task is learned over epochs if you run the experiment on
## all training epochs! interesting 

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
    
    Converts nparray to tensor
    Converts class folder to target tensor
    Converts uniform and nonuniform folder to target tensors (with 1 class) for binary cross entropy loss
    '''
    def __init__(self, data_dir,TEST=False):
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
                # select only one uniform and one nonuniform sample per class
                if len(uni_files) > 0 and len(nonuni_files) > 0:
                    samples.append((uni_files[0], nonuni_files[0], class_dir))
            else:
                samples.extend(list(zip(uni_files, nonuni_files, [class_dir]*len(uni_files))))
        return samples
            
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uni_file, nonuni_file, class_name = self.samples[idx]
        uni_img = np.loadtxt(uni_file).astype(np.float32)
        nonuni_img = np.loadtxt(nonuni_file).astype(np.float32)
        
        uni_img = torch.from_numpy(uni_img).unsqueeze(0) 
        nonuni_img = torch.from_numpy(nonuni_img).unsqueeze(0)

        target = torch.tensor(self.class_to_idx[class_name], dtype=torch.long)
        uni_target = torch.tensor([1], dtype=torch.float32)
        nonuni_target = torch.tensor([0], dtype=torch.float32)
        
        return [uni_img, nonuni_img, target, uni_target, nonuni_target]
    
    def get_num_classes(self):
        return len(self.class_dirs)


def train_UI(args, model, device, train_loader, optimizer, uni_optimizer, epoch,
          cycles, mse_parameter, clean_parameter, clean='supclean'):
    
    model.train()

    train_loss = 0.0
    uni_train_loss = 0.0

    # in this training protocol, 'clean' images are all uniform
    train_clean_data = {
        'epoch':[],
        'batch index': [],
        'class targets': [],
        'uni targets': [], # these are ALWAYS 1 because here clean data is uniform
        # class preds
        'class predictions': [],
        # uni preds
        'uni predictions': [],
    }

    # in this training protocol, 'adv' images are all NONuniform
    train_adv_data = {
        'epoch':[],
        'batch index': [],
        'class targets': [],
        'nonuni targets': [], # these are ALWAYS 0 because here adv data is nonuniform
        # class preds
        'class predictions': [],
        # uni preds
        'nonuni predictions': [], # always nonuni prediction
    }

    # model.module.reset()
    model.reset()

    for batch_idx, (uni_images, nonuni_images, targets, uni_target, nonuni_target) in enumerate(train_loader):
        optimizer.zero_grad()
        uni_optimizer.zero_grad()

        # targets are only the feature type
        images, nonuni_images, targets = uni_images.to(device), nonuni_images.to(device), targets.to(device)
        # uni nonuni targets are the uniformity label (uniform: 1 or not: 0)
        uni_target, nonuni_target = uni_target.to(device), nonuni_target.to(device)

        # # checks for corrupted data in batch 
        # for i in range(images.shape[0]):
        #     assert images[i].shape == torch.Size([1, 128, 128]), f"uni image {i} shape is not [1, 128, 128]"
        #     assert nonuni_images[i].shape == torch.Size([1, 128, 128]), f"nonuni image {i} shape is not [1, 128, 128]"

        # concatenate uniform and non-uniform images for processing
        images_all = torch.cat((images, nonuni_images), 0)
        
        # needed otherwise model blows up 
        model.reset()

        '''
        Notes:

        The targets only contain information about the class of features, not whether they are uniform or not.
        Both images, uniform and nonuniform, have the same target.  
        this is because the targets must always be consistent in uniform and non uniform trials for the loss
        computation that allows the generative feedback to learn. 

        The uniformity targets for images are different, and they are used for the model to learn
        to differentiate between uniform and nonuniform images. The loss for this is computed seperately 
        and a seperate fully connected layer in the model is used for this classification task.

        '''
        # FIRST FORWARD PASS
        logits, uni_fc_logits, orig_feature_all, block1_all, block2_all, block3_all = model(images_all, first=True, inter=True)
        ff_prev = orig_feature_all
        # print(f"GPU memory allocated after first forward: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        '''
        Notes: 

        The first feedforward pass gets the original features and logits for ALL IMAGES
        adv and clean (aka nonuniform and uniform). 

        orig_feature_all = final layer features extracted for all images 

        The uni_fc_logits are for computing the loss of the binary output node that ONLY learns the UI task,
        so just learns whether the input is uniform or not. It is only part of the feedforward arcitecture. 
        
        '''
        # extract the features from clean images
        orig_feature, _ = torch.split(orig_feature_all, images.size(0))
        block1_clean, _ = torch.split(block1_all, images.size(0))
        block2_clean, _ = torch.split(block2_all, images.size(0))
        block3_clean, _ = torch.split(block3_all, images.size(0))

        # logits from the first forward pass
        logits_clean, logits_adv = torch.split(logits, images.size(0))

        '''
        initial loss is computed using only logits from this first forward pass

        include clean images to prevent overfitting to adv perturbations:
            supclean = include loss of clean images in total loss 
            no = only use adversarial loss
        
        '''
        if not ('no' in clean):
            loss = (clean_parameter * F.cross_entropy(logits_clean, targets) + F.cross_entropy(logits_adv, targets)) / (2*(cycles+1))
        else:        
            loss = F.cross_entropy(logits_adv, targets) / (cycles+1) 
        '''
        a feedback cycle is ran for every forward pass made by the model during training
        to extract the reconstructions of every single forward pass
        '''
        for i_cycle in range(1,cycles+1):
            # cycles start at 1 not 0 
            """
            set model to backward steps with intermediate reconstruction at conv layers (inter_recon = True)

            extract reconstructions by making custom backwards steps
            (this is the DGM: deconvolution at each conv layer and unpooling etc)
                
            recon: reconstructed features at the final layer of the network, reconstructions of the 
            original features (orig_feature) that were extracted in the initial forward pass

            """
            # print(f"GPU memory allocated before recon cycle {i_cycle}: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            recon, block1_recon, block2_recon, block3_recon = model(logits, step='backward', inter_recon=True)
            # print(f"GPU memory allocated after recon cycle {i_cycle}: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            # split for clean and adversarial reconstructions from DGM
            recon_clean, recon_adv = torch.split(recon, images.size(0))
            recon_block1_clean, recon_block1_adv = torch.split(block1_recon, images.size(0))
            recon_block2_clean, recon_block2_adv = torch.split(block2_recon, images.size(0))
            recon_block3_clean, recon_block3_adv = torch.split(block3_recon, images.size(0))
            '''
                ASCII bunny sends blessings

                             _...._
                         .-'       '-.
                        /              
                        /      (\__/)   
                       |       (-ㅅo)    |
                       |      /  <3 \   |
                        \    /   <3  \  /
                         '-._\       /_.-'
                             '-.___.-'

                     -  /|
                     \'o.0'   - nyauuu
                    =(____)=
                       Uu
            '''
            loss += (F.mse_loss(recon_adv, orig_feature) 
                     + F.mse_loss(recon_block1_adv, block1_clean) 
                     + F.mse_loss(recon_block2_adv, block2_clean) 
                     + F.mse_loss(recon_block3_adv, block3_clean)) * mse_parameter / (4*cycles)
            # feedforward again 
            '''
            sad smooth brain moment :c
            ff current is updated using the clean features extracted at the final layer (ff_prev)
            + a weighted difference between these clean features from the first pass against 
            the reconstructed data at this layer
            '''
            ff_current = ff_prev + args.res_parameter * (recon - ff_prev)

            # saves reconstructions from all blocks and final layer
            # also saves the original input image for comparison
            # takes the first image of the batch and average over all
            # channels
            # saves the ff_current and ff_prev as images for comparison, 
            # averaged over all channels too
            # modify if needed 
            if (epoch) % 2 == 0 or epoch == 1:
                epoch_dir = f"{args.model_dir}/recon_imgs/epoch_{epoch}"
                cycle_dir = os.path.join(epoch_dir, f"cycle_{i_cycle}")
                clean_dir = os.path.join(cycle_dir, "clean")
                adv_dir = os.path.join(cycle_dir, "adv")
                os.makedirs(clean_dir, exist_ok=True)
                os.makedirs(adv_dir, exist_ok=True)
            
                # input imgs saved for comparison
                clean_input_img = norm_t(images[0])
                adv_input_img = norm_t(nonuni_images[0])
                save_image(clean_input_img, os.path.join(clean_dir, "clean_input.png"))
                save_image(adv_input_img, os.path.join(adv_dir, "adv_input.png"))

                # ff current
                ff_current_img = norm_t(ff_current[0].mean(dim=0, keepdim=True))
                save_image(ff_current_img, os.path.join(cycle_dir,  f"ff_current.png"))
                # ff_prev
                ff_prev_img = norm_t(ff_prev[0].mean(dim=0, keepdim=True))
                save_image(ff_prev_img, os.path.join(cycle_dir,  f"ff_prev.png"))
                # original feedforward fature map
                orig_feature_img = norm_t(orig_feature[0].mean(dim=0, keepdim=True))
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

            ########################### NIKLAS:
            # cycle_logits, _, _, _, _, _ = model(logits, first=True, inter=True)
            # cycle_uni_logits, cycle_nonuni_logits = torch.split(cycle_uni_fc_logits, images.size(0))
            # criterion = nn.BCELoss()
            # cycle_uni_loss = criterion(cycle_uni_logits, uni_target)
            # cycle_nonuni_loss = criterion(cycle_nonuni_logits, uni_target) # <------------ This should now be uni_target!!! ??? !!!
            # collect all the uni_losses and nonuni_losses over cycles and add together in the same way as for the reconstruction/class loss
            ###########################

            if not ('no' in clean):
                loss += (clean_parameter * F.cross_entropy(logits_clean, targets) 
                         + F.cross_entropy(logits_adv, targets)) / (2*(cycles+1))
            else:
                loss += F.cross_entropy(logits_adv, targets) / (cycles+1)

        ## predictions for class targets
        ## clean images ONLY === only uniform images
        clean_class_pred = logits_clean.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        ## now for adversarial images too 
        adv_class_pred = logits_adv.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

        '''
        The loss for the uni fc linear: uni_fc_logits is from the FIRST forward pass only.
        The uni_fc_logits has one class

        uni targets = 1
        nonuni targets = 0
        '''
        criterion = nn.BCELoss()  # binary cross entropy
        # get uni and nonuni logits
        uni_logits, nonuni_logits = torch.split(uni_fc_logits, images.size(0))

        uni_loss = criterion(uni_logits, uni_target)
        nonuni_loss = criterion(nonuni_logits, nonuni_target)
        # correct for uni and non uni 
        uni_pred = (uni_logits > 0.5).float()
        nonuni_pred = (nonuni_logits > 0.5).float()

        # save batch data for clean images
        train_clean_data['epoch'].append(epoch)
        train_clean_data['batch index'].append(batch_idx)
        train_clean_data['class targets'].append(targets.cpu().numpy().tolist())
        train_clean_data['uni targets'].append(uni_target.cpu().numpy().tolist())
        train_clean_data['class predictions'].append(clean_class_pred.cpu().numpy().tolist())
        train_clean_data['uni predictions'].append(uni_pred.cpu().numpy().tolist())

        # save batch data for adversarial images
        train_adv_data['epoch'].append(epoch)
        train_adv_data['batch index'].append(batch_idx)
        train_adv_data['class targets'].append(targets.cpu().numpy().tolist())
        train_adv_data['nonuni targets'].append(nonuni_target.cpu().numpy().tolist())
        train_adv_data['class predictions'].append(adv_class_pred.cpu().numpy().tolist())
        train_adv_data['nonuni predictions'].append(nonuni_pred.cpu().numpy().tolist())

        # total loss of uni and non uni
        uni_total_loss = uni_loss + nonuni_loss

        # compute gradients
        # print(f"GPU memory allocated before total_loss.backward(): {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        uni_total_loss.backward(retain_graph=True)
        loss.backward(retain_graph=False)
        # total_loss.backward()
        # print(f"GPU memory allocated after total_loss.backward(): {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        uni_optimizer.step()
        optimizer.step()

    # update losses
    train_loss += loss.item()
    train_loss /= len(train_loader.dataset)
    uni_train_loss += uni_total_loss.item()
    uni_train_loss /= len(train_loader.dataset)
    
    # scheduler.step()
    return train_loss, uni_train_loss, train_clean_data, train_adv_data


def val(args, model, device, val_loader, cycles, epoch):
    '''
    vals the performance of the model on 4 different elements:

    1. clean (aka uniform) predictions against class targets 
    2. adversarial (aka nonuniform) predictions against class targets 
    3. clean (uniform) predictions against uniformity target 
    4. adversarial (nonuiform) predictions against uniformity target 

    note: 1 and 2 have the same class target, imgs belong to the same class. 
    3 & 4 have a different uniformity target but this is represented by a single binary node. 

    1 & 2 the outputs are from the class logits, 3 & 4 are obtained from the single node
    of the uniformity_fc layer in the model

    '''
    model.eval()

    clean_val_loss = 0.0
    adv_val_loss = 0.0
    uni_val_loss = 0.0

    val_clean_data = {
        'epoch':[],
        'batch index': [],
        'class targets': [],
        'uni targets': [], # these are ALWAYS 1 because here clean data is uniform
        # class preds
        'class predictions': [],
        # uni preds
        'uni predictions': [],
    }

    # in this training protocol, 'adv' images are all NONuniform
    val_adv_data = {
        'epoch':[],
        'batch index': [],
        'class targets': [],
        'nonuni targets': [], # these are ALWAYS 0 because here adv data is nonuniform
        # class preds
        'class predictions': [],
        # uni preds
        'nonuni predictions': [], # always nonuni prediction
    }

    with torch.no_grad():
        for batch_idx, (uni_images, nonuni_images, targets, uni_target, nonuni_target) in enumerate(val_loader):

            # targets are only the feature type
            images, nonuni_images, targets = uni_images.to(device), nonuni_images.to(device), targets.to(device)
            # uni nonuni targets are the UNIFORMITY label (uniform:0 or not:1)
            uni_target, nonuni_target = uni_target.to(device), nonuni_target.to(device)

            images_all = torch.cat((images, nonuni_images), 0)
            # model.module.reset()
            model.reset()

            # first forward
            output, uni_fc_logits, orig_feature, _, _, _ = model(images_all, first=True, inter=True)
            ff_prev = orig_feature

            for i_cycle in range(1,cycles+1):
                # cycles start at 1 not 0 
                recon = model(output, step='backward')
                ff_current = ff_prev + args.res_parameter * (recon - ff_prev)
                output = model(ff_current, first=False)
                ff_prev = ff_current

            # split for clean and adv loss calculations 
            clean_output, adv_output = torch.split(output, images.size(0))

            # classification loss for feature type
            clean_val_loss += F.cross_entropy(clean_output, targets, reduction='sum').item()
            adv_val_loss += F.cross_entropy(adv_output, targets, reduction='sum').item()
            # accuracy clean and adv imgs
            clean_class_pred = clean_output.argmax(dim=1, keepdim=True)
            adv_class_pred = adv_output.argmax(dim=1, keepdim=True)

            # split logits for uni and non uni imgs 
            uni_logits, nonuni_logits = torch.split(uni_fc_logits, images.size(0))

            criterion = nn.BCELoss()
            uni_loss = criterion(uni_logits, uni_target).item()
            nonuni_loss = criterion(nonuni_logits, nonuni_target).item()
            uni_val_loss += (uni_loss + nonuni_loss)

            # acc for uniformity 
            uni_pred = (uni_logits > 0.5).float()
            nonuni_pred = (nonuni_logits > 0.5).float()

            # save batch data for clean images
            val_clean_data['epoch'].append(epoch)
            val_clean_data['batch index'].append(batch_idx)
            val_clean_data['class targets'].append(targets.cpu().numpy().tolist())
            val_clean_data['uni targets'].append(uni_target.cpu().numpy().tolist())
            val_clean_data['class predictions'].append(clean_class_pred.cpu().numpy().tolist())
            val_clean_data['uni predictions'].append(uni_pred.cpu().numpy().tolist())

            # save batch data for adversarial images
            val_adv_data['epoch'].append(epoch)
            val_adv_data['batch index'].append(batch_idx)
            val_adv_data['class targets'].append(targets.cpu().numpy().tolist())
            val_adv_data['nonuni targets'].append(nonuni_target.cpu().numpy().tolist())
            val_adv_data['class predictions'].append(adv_class_pred.cpu().numpy().tolist())
            val_adv_data['nonuni predictions'].append(nonuni_pred.cpu().numpy().tolist())

        # update loss for epoch 
        clean_val_loss /= len(val_loader)
        adv_val_loss /= len(val_loader)
        uni_val_loss /= len(val_loader)

        return clean_val_loss, adv_val_loss, uni_val_loss, val_clean_data, val_adv_data
    

def exp(args, model, device, exp_loader, cycles, epoch, folder):
    model.eval()

    class_losses = [0.0] * (cycles + 1)
    uni_losses = [0.0] * (cycles + 1)
    UI_cycle_losses = [0.0] * (cycles + 1)

    ## NOTE clean means uniform 
    exp_clean_data = [{
        'epoch': [],
        'batch index': [],
        'cycle': [],
        'class targets': [],
        'uni targets': [],
        'class predictions': [],
        'uni predictions': [],
        'uni cycle predictions': [], # the predictions from the logits obtained within the cycles
        'uni logits': [],
        'uni cycle logits': []
    } for _ in range(cycles + 1)]
    
    ## NOTE adv means nonuniform 
    exp_adv_data = [{
        'epoch': [],
        'batch index': [],
        'cycle': [],
        'class targets': [],
        'nonuni targets': [],
        'class predictions': [],
        'nonuni predictions': [],
        'nonuni cycle predictions': [], # the predictions from the logits obtained within the cycles
        'nonuni logits': [],
        'nonuni cycle logits': []
    } for _ in range(cycles + 1)]

    with torch.no_grad():
        for batch_idx, (uni_images, nonuni_images, targets, uni_target, nonuni_target) in enumerate(exp_loader):
            images, nonuni_images, targets = uni_images.to(device), nonuni_images.to(device), targets.to(device)
            uni_target, nonuni_target = uni_target.to(device), nonuni_target.to(device)

            images_all = torch.cat((images, nonuni_images), 0)
            model.reset()

            output, uni_fc_logits, orig_feature, _, _, _ = model(images_all, first=True, inter=True)
            ff_prev = orig_feature

            ## CLASS TASK
            clean_output, adv_output = torch.split(output, images.size(0))
            clean_exp_loss = F.cross_entropy(clean_output, targets, reduction='sum').item()
            adv_exp_loss = F.cross_entropy(adv_output, targets, reduction='sum').item()
            class_losses[0] += (clean_exp_loss + adv_exp_loss)

            clean_class_pred = clean_output.argmax(dim=1, keepdim=True)
            adv_class_pred = adv_output.argmax(dim=1, keepdim=True)

            ## UI TASK 
            uni_logits, nonuni_logits = torch.split(uni_fc_logits, images.size(0))
            criterion = nn.BCELoss()
            uni_loss = criterion(uni_logits, uni_target).item()
            nonuni_loss = criterion(nonuni_logits, nonuni_target).item()
            uni_losses[0] +=  (uni_loss + nonuni_loss)
            UI_cycle_losses[0] = 666666666

            uni_pred = (uni_logits > 0.5).float()
            nonuni_pred = (nonuni_logits > 0.5).float()

            exp_clean_data[0]['epoch'].append(epoch)
            exp_clean_data[0]['batch index'].append(batch_idx)
            exp_clean_data[0]['cycle'].append(0)
            exp_clean_data[0]['class targets'].append(targets.cpu().numpy().tolist())
            exp_clean_data[0]['uni targets'].append(uni_target.cpu().numpy().tolist())
            exp_clean_data[0]['class predictions'].append(clean_class_pred.cpu().numpy().tolist())
            exp_clean_data[0]['uni predictions'].append(uni_pred.cpu().numpy().tolist())
            exp_clean_data[0]['uni cycle predictions'].append(666)
            exp_clean_data[0]['uni logits'].append(uni_logits.cpu().numpy().tolist())
            exp_clean_data[0]['uni cycle logits'].append(666)

            exp_adv_data[0]['epoch'].append(epoch)
            exp_adv_data[0]['batch index'].append(batch_idx)
            exp_adv_data[0]['cycle'].append(0)
            exp_adv_data[0]['class targets'].append(targets.cpu().numpy().tolist())
            exp_adv_data[0]['nonuni targets'].append(nonuni_target.cpu().numpy().tolist())
            exp_adv_data[0]['class predictions'].append(adv_class_pred.cpu().numpy().tolist())
            exp_adv_data[0]['nonuni predictions'].append(nonuni_pred.cpu().numpy().tolist())
            exp_adv_data[0]['nonuni cycle predictions'].append(666)
            exp_adv_data[0]['nonuni logits'].append(nonuni_logits.cpu().numpy().tolist())
            exp_adv_data[0]['nonuni cycle logits'].append(666)
            
            for i_cycle in range(1,cycles+1):
                # cycles start at 1 not 0 
                recon = model(output, step='backward')
                ff_current = ff_prev + args.res_parameter * (recon - ff_prev)
                # mean of channels in ff_current to use for uni logits within cycle
                ff_current_mean = ff_current.mean(dim=1, keepdim=True) 
                output = model(ff_current, first=False)
                ff_prev = ff_current

                ### CLASS TASK
                clean_output, adv_output = torch.split(output, images.size(0))
                clean_exp_loss = F.cross_entropy(clean_output, targets, reduction='sum').item()
                adv_exp_loss = F.cross_entropy(adv_output, targets, reduction='sum').item()
                class_losses[i_cycle] += (clean_exp_loss + adv_exp_loss)

                clean_class_pred = clean_output.argmax(dim=1, keepdim=True)
                adv_class_pred = adv_output.argmax(dim=1, keepdim=True)

                ### UI TASK 
                uni_logits, nonuni_logits = torch.split(uni_fc_logits, images.size(0))
                criterion = nn.BCELoss()
                uni_loss = criterion(uni_logits, uni_target)
                nonuni_loss = criterion(nonuni_logits, nonuni_target)
                uni_losses[i_cycle] += (uni_loss + nonuni_loss).item()

                uni_pred = (uni_logits > 0.5).float()
                nonuni_pred = (nonuni_logits > 0.5).float()

                # obtain UI task logits on reconstructed inputs
                # by passing ff_current back into the models first forward pass
                _, cycle_uni_fc_logits, _, _, _, _ = model(ff_current_mean, first=True, inter=True)
                # split for uni and nonuni images
                cycle_uni_logits, cycle_nonuni_logits = torch.split(cycle_uni_fc_logits, images.size(0))
                uni_cycle_loss = criterion(cycle_uni_logits, uni_target).item()
                nonuni_cycle_loss = criterion(cycle_nonuni_logits, nonuni_target).item()

                UI_cycle_losses[i_cycle] += (uni_cycle_loss + nonuni_cycle_loss)
                
                cycle_uni_pred = (cycle_uni_logits > 0.5).float()
                cycle_nonuni_pred = (cycle_nonuni_logits > 0.5).float()

                exp_clean_data[i_cycle]['epoch'].append(epoch)
                exp_clean_data[i_cycle]['batch index'].append(batch_idx)
                exp_clean_data[i_cycle]['cycle'].append(i_cycle)
                exp_clean_data[i_cycle]['class targets'].append(targets.cpu().numpy().tolist())
                exp_clean_data[i_cycle]['uni targets'].append(uni_target.cpu().numpy().tolist())
                exp_clean_data[i_cycle]['class predictions'].append(clean_class_pred.cpu().numpy().tolist())
                exp_clean_data[i_cycle]['uni predictions'].append(uni_pred.cpu().numpy().tolist())
                exp_clean_data[i_cycle]['uni cycle predictions'].append(cycle_uni_pred.cpu().numpy().tolist())
                exp_clean_data[i_cycle]['uni logits'].append(uni_logits.cpu().numpy().tolist())
                exp_clean_data[i_cycle]['uni cycle logits'].append(cycle_uni_logits.cpu().numpy().tolist())
                
                exp_adv_data[i_cycle]['epoch'].append(epoch)
                exp_adv_data[i_cycle]['batch index'].append(batch_idx)
                exp_adv_data[i_cycle]['cycle'].append(i_cycle)
                exp_adv_data[i_cycle]['class targets'].append(targets.cpu().numpy().tolist())
                exp_adv_data[i_cycle]['nonuni targets'].append(nonuni_target.cpu().numpy().tolist())
                exp_adv_data[i_cycle]['class predictions'].append(adv_class_pred.cpu().numpy().tolist())
                exp_adv_data[i_cycle]['nonuni predictions'].append(cycle_nonuni_pred.cpu().numpy().tolist())
                exp_adv_data[i_cycle]['nonuni cycle predictions'].append(cycle_nonuni_pred.cpu().numpy().tolist())
                exp_adv_data[i_cycle]['nonuni logits'].append(nonuni_logits.cpu().numpy().tolist())
                exp_adv_data[i_cycle]['nonuni cycle logits'].append(cycle_nonuni_logits.cpu().numpy().tolist())
                
                ### saving all reconstructions as npy arrays
                ### for every image in the batch
                # <model_dir>/
                # └── exp_recon_arrays_<folder>/
                #     └── cycle_<i_cycle>/
                #         └── <batch_idx>/
                #             └── img_<img_idx>/
                #                 ├── clean_input.npy
                #                 ├── adv_input.npy
                #                 ├── ff_current.npy
                #                 └── recon.npy 
                # if i_cycle == 1 or i_cycle == cycles + 1 or i_cycle % 2 == 0:
                all_cycle_test = True
                if all_cycle_test:
                    exp_cycle_dir = f"{args.model_dir}/exp_recon_arrays_{folder}/cycle_{i_cycle}"
                    batch_dir = os.path.join(exp_cycle_dir, str(f'batch_{batch_idx}'))
                    os.makedirs(batch_dir, exist_ok=True)
                    # for every image in the batch
                    for img_idx in range(args.batch_size):
                        img_dir = os.path.join(batch_dir, f"img_{img_idx}")
                        os.makedirs(img_dir, exist_ok=True)
                        np.save(os.path.join(img_dir, "clean_input.npy"), images[img_idx].cpu().numpy())
                        np.save(os.path.join(img_dir, "adv_input.npy"), nonuni_images[img_idx].cpu().numpy())
                        np.save(os.path.join(img_dir, "ff_current.npy"), ff_current[img_idx].cpu().numpy())
                        np.save(os.path.join(img_dir, "recon.npy"), recon[img_idx].cpu().numpy())

        for i_cycle in range(cycles + 1):
            class_losses[i_cycle] /= len(exp_loader)
            uni_losses[i_cycle] /= len(exp_loader)
            UI_cycle_losses[i_cycle] /= len(exp_loader)

        return class_losses, uni_losses, UI_cycle_losses, exp_clean_data, exp_adv_data


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

    # num workers same as cores per gpu
    # kwargs = {'num_workers': 18, 'pin_memory': True} if use_cuda else {}
    
    # # train, test data dirs
    train_data = CustomDataset(data_dir=f"{args.datadir}/train")
    val_data = CustomDataset(data_dir=f"{args.datadir}/test")

    # ### FOR TESTING CODE
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

    val_loader = torch.utils.data.DataLoader(
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
    # best_acc = 0
    # best_uni_acc = 0
    # overall_best = 0
    epoch_train_times = []

    # storage for training, validation and experimental data
    train_clean_df = pd.DataFrame(columns=[
        'epoch', 'batch index', 'class targets', 'uni targets', 'class predictions', 'uni predictions'
    ])
    train_adv_df = pd.DataFrame(columns=[
        'epoch', 'batch index', 'class targets', 'nonuni targets', 'class predictions', 'nonuni predictions'
    ])
    val_clean_df = pd.DataFrame(columns=[
        'epoch', 'batch index', 'class targets', 'uni targets', 'class predictions', 'uni predictions'
    ])
    val_adv_df = pd.DataFrame(columns=[
        'epoch', 'batch index', 'class targets', 'nonuni targets', 'class predictions', 'nonuni predictions'
    ])
    exp_clean_df = pd.DataFrame(columns=[
        'data folder',

        'epoch', 'batch index', 'cycle', 'class targets', 'uni targets', 
        
        'class predictions', 'uni predictions', 'uni cycle predictions',

        'uni logits', 'uni cycle logits'
    ])
    exp_adv_df = pd.DataFrame(columns=[
        'data folder',

        'epoch', 'batch index', 'cycle', 'class targets', 'nonuni targets', 
        
        'class predictions', 'nonuni predictions', 'nonuni cycle predictions',

        'nonuni logits', 'nonuni cycle logits'
    ])

    train_losses = {'epoch': [], 'class loss': [], 'uniformity loss': []}
    val_losses = {'epoch': [], 'clean class loss': [],'adv class loss': [], 'uniformity loss': []}
    exp_losses = {'epoch': [], 'data folder': [], 'class_losses': [], 'uni_losses': [], 'UI_cycle_losses': [],}

    for epoch in range(1,args.epochs+1):
        # torch.cuda.empty_cache()

        # time start epoch:
        epoch_start = time.time()

        #############
        ### TRAIN ###
        #############
        train_loss, uni_train_loss, \
        epoch_train_clean_data, epoch_train_adv_data = train_UI(args, model, device, train_loader, optimizer, uni_optimizer, epoch,
            cycles=args.train_cycles, mse_parameter=args.mse_parameter, clean_parameter=args.clean_parameter, 
            clean=args.clean)

        # time end epoch:
        epoch_end = time.time()
        epoch_train_time = epoch_end - epoch_start
        epoch_train_time_for = str(timedelta(seconds=epoch_train_time))
        epoch_train_times.append(epoch_train_time)

        # save every model for all epochs and check evaulation code
        # theres an issue with saving and/or loading models so slip for now 
        # torch.save(model.state_dict(), args.model_dir + f"/{epoch}_model.pt")

        # convert to list
        train_loss = tensor_to_list(train_loss)
        uni_train_loss = tensor_to_list(uni_train_loss)
        epoch_train_clean_data = {key: tensor_to_list(value) for key, value in epoch_train_clean_data.items()}
        epoch_train_adv_data = {key: tensor_to_list(value) for key, value in epoch_train_adv_data.items()}

        # append training losses
        train_losses['epoch'].append(epoch)
        train_losses['class loss'].append(train_loss)
        train_losses['uniformity loss'].append(uni_train_loss)

        # save data to dataframes
        for batch_idx in range(len(epoch_train_clean_data['class targets'])):
            train_clean_batch_data = {
                'epoch': [epoch] * len(epoch_train_clean_data['class targets'][batch_idx]),
                'batch index': [batch_idx] * len(epoch_train_clean_data['class targets'][batch_idx]),
                'class targets': epoch_train_clean_data['class targets'][batch_idx],
                'uni targets': [item[0] for item in epoch_train_clean_data['uni targets'][batch_idx]],
                'class predictions': [item[0] for item in epoch_train_clean_data['class predictions'][batch_idx]],
                'uni predictions': [item[0] for item in epoch_train_clean_data['uni predictions'][batch_idx]]
            }
            train_clean_df = pd.concat([train_clean_df, pd.DataFrame(train_clean_batch_data)], ignore_index=True)

        for batch_idx in range(len(epoch_train_adv_data['class targets'])):
            train_adv_batch_data = {
                'epoch': [epoch] * len(epoch_train_adv_data['class targets'][batch_idx]),
                'batch index': [batch_idx] * len(epoch_train_adv_data['class targets'][batch_idx]),
                'class targets': epoch_train_adv_data['class targets'][batch_idx],
                'nonuni targets': [item[0] for item in epoch_train_adv_data['nonuni targets'][batch_idx]],
                'class predictions': [item[0] for item in epoch_train_adv_data['class predictions'][batch_idx]],
                'nonuni predictions': [item[0] for item in epoch_train_adv_data['nonuni predictions'][batch_idx]]
            }
            train_adv_df = pd.concat([train_adv_df, pd.DataFrame(train_adv_batch_data)], ignore_index=True)

        ###################
        #### VALIDATION ###
        ###################
        clean_val_loss, adv_val_loss, uni_val_loss, \
        epoch_val_clean_data, epoch_val_adv_data= val(args, model, device, val_loader, 
                                             cycles=args.train_cycles, epoch=epoch)
        
        # convert to list
        clean_val_loss = tensor_to_list(clean_val_loss)
        adv_val_loss = tensor_to_list(adv_val_loss)
        uni_val_loss = tensor_to_list(uni_val_loss)
        epoch_val_clean_data = {key: tensor_to_list(value) for key, value in epoch_val_clean_data.items()}
        epoch_val_adv_data = {key: tensor_to_list(value) for key, value in epoch_val_adv_data.items()}
        
        # append training losses
        val_losses['epoch'].append(epoch)
        val_losses['clean class loss'].append(clean_val_loss)
        val_losses['adv class loss'].append(adv_val_loss)
        val_losses['uniformity loss'].append(uni_val_loss)
        
        for batch_idx in range(len(epoch_val_clean_data['class targets'])):
            val_clean_batch_data = {
                'epoch': [epoch] * len(epoch_val_clean_data['class targets'][batch_idx]),
                'batch index': [batch_idx] * len(epoch_val_clean_data['class targets'][batch_idx]),
                'class targets': epoch_val_clean_data['class targets'][batch_idx],
                'uni targets': [item[0] for item in epoch_val_clean_data['uni targets'][batch_idx]],
                'class predictions': [item[0] for item in epoch_val_clean_data['class predictions'][batch_idx]],
                'uni predictions': [item[0] for item in epoch_val_clean_data['uni predictions'][batch_idx]]
            }
            val_clean_df = pd.concat([val_clean_df, pd.DataFrame(val_clean_batch_data)], ignore_index=True)

        for batch_idx in range(len(epoch_val_adv_data['class targets'])):
            val_adv_batch_data = {
                'epoch': [epoch] * len(epoch_val_adv_data['class targets'][batch_idx]),
                'batch index': [batch_idx] * len(epoch_val_adv_data['class targets'][batch_idx]),
                'class targets': epoch_val_adv_data['class targets'][batch_idx],
                'nonuni targets': [item[0] for item in epoch_val_adv_data['nonuni targets'][batch_idx]],
                'class predictions': [item[0] for item in epoch_val_adv_data['class predictions'][batch_idx]],
                'nonuni predictions': [item[0] for item in epoch_val_adv_data['nonuni predictions'][batch_idx]]
            }
            val_adv_df = pd.concat([val_adv_df, pd.DataFrame(val_adv_batch_data)], ignore_index=True)

        ### calculate accuracies! 
        train_clean_class_acc = calculate_accuracy(train_clean_df[train_clean_df['epoch'] == epoch]['class targets'], train_clean_df[train_clean_df['epoch'] == epoch]['class predictions'])
        train_adv_class_acc = calculate_accuracy(train_adv_df[train_adv_df['epoch'] == epoch]['class targets'], train_adv_df[train_adv_df['epoch'] == epoch]['class predictions'])
        val_clean_class_acc = calculate_accuracy(val_clean_df[val_clean_df['epoch'] == epoch]['class targets'], val_clean_df[val_clean_df['epoch'] == epoch]['class predictions'])
        val_adv_class_acc = calculate_accuracy(val_adv_df[val_adv_df['epoch'] == epoch]['class targets'], val_adv_df[val_adv_df['epoch'] == epoch]['class predictions'])

        train_clean_uni_acc = calculate_accuracy(train_clean_df[train_clean_df['epoch'] == epoch]['uni targets'], train_clean_df[train_clean_df['epoch'] == epoch]['uni predictions'])
        train_adv_uni_acc = calculate_accuracy(train_adv_df[train_adv_df['epoch'] == epoch]['nonuni targets'], train_adv_df[train_adv_df['epoch'] == epoch]['nonuni predictions'])
        val_clean_uni_acc = calculate_accuracy(val_clean_df[val_clean_df['epoch'] == epoch]['uni targets'], val_clean_df[val_clean_df['epoch'] == epoch]['uni predictions'])
        val_adv_uni_acc = calculate_accuracy(val_adv_df[val_adv_df['epoch'] == epoch]['nonuni targets'], val_adv_df[val_adv_df['epoch'] == epoch]['nonuni predictions'])

        logging.info(f'𓍯𓂃𓏧♡ ₍ᐢ._.ᐢ₎♡ ༘ ꒰ᐢ. .ᐢ꒱ ₊˚⊹ *ੈ✩‧₊˚༺☆༻*ੈ✩‧₊˚')
        logging.info(f'')
        logging.info(f"Epoch {epoch} Results:")
        logging.info(f"~~~~~~~~~~~~~~~~~~~~~~~~~~")
        logging.info(f"Train Time: {epoch_train_time_for}")
        logging.info(f"Class Loss - Train: {train_loss:.4f} | Val: clean {clean_val_loss:.4f}, adv {adv_val_loss:.4f}")
        logging.info(f"Uniformity Loss - Train: {uni_train_loss:.4f} | Val: {uni_val_loss:.4f}")
        logging.info(f"♡")
        logging.info(f"Class Accuracy:")
        logging.info(f"  Clean - Train: {train_clean_class_acc:.2f} | Val: {val_clean_class_acc:.2f}")
        logging.info(f"  Adversarial - Train: {train_adv_class_acc:.2f} | Val: {val_adv_class_acc:.2f}")
        logging.info(f"  TOTAL - Train: {((train_adv_class_acc+train_clean_class_acc)/2):.2f} | Val: {((val_clean_class_acc+val_adv_class_acc)/2):.2f}")
        logging.info(f'✩')
        logging.info(f"Uniformity Accuracy:")
        logging.info(f"  Clean - Train: {train_clean_uni_acc:.2f} | Val: {val_clean_uni_acc:.2f}")
        logging.info(f"  Adversarial - Train: {train_adv_uni_acc:.2f} | Val: {val_adv_uni_acc:.2f}")
        logging.info(f"  TOTAL - Train: {((train_clean_uni_acc + train_adv_uni_acc)/2):.2f} | Val: {((val_clean_uni_acc + val_adv_uni_acc)/2):.2f}")
        logging.info(f'')
        logging.info(f"𖦹")
        logging.info(f'')

        # losses to df
        train_losses_df = pd.DataFrame(train_losses)
        val_losses_df = pd.DataFrame(val_losses)

        ############################
        ######## EXPERIMENT ########
        ############################
        ## on last 3 epochs
        if epoch >= args.epochs-2:
        # test_all_epochs = True
        # if test_all_epochs:
            data_folders = [d for d in os.listdir(args.exp_data_dir) if os.path.isdir(os.path.join(args.exp_data_dir, d))]
            print(data_folders)
            for folder in data_folders:
                print(folder)
                exp_data = CustomDataset(f"{args.exp_data_dir}/{folder}")

                exp_loader = torch.utils.data.DataLoader(
                    exp_data, batch_size=args.batch_size,
                    shuffle=True, num_workers=18, pin_memory=True,
                    drop_last=True)
                print('length exp loader',len(exp_loader.dataset))

                exp_class_losses, exp_uni_losses, exp_UI_cycle_losses, \
                exp_clean_data, exp_adv_data = exp(args, model, device, exp_loader, cycles=args.exp_cycles, epoch=epoch, folder=folder)
                
                clean_exp_rows = []
                for cycle_data in exp_clean_data:
                    for i in range(len(cycle_data['batch index'])):  # iterate over batches
                        for j in range(len(cycle_data['class targets'][i])):  # iterate over images within the batch
                            if cycle_data['cycle'][i] >= 1:
                                uni_cycle_prediction = cycle_data['uni cycle predictions'][i][j]
                                uni_cycle_logit = cycle_data['uni cycle logits'][i][j]
                            elif cycle_data['cycle'][i] == 0:
                                uni_cycle_prediction = cycle_data['uni cycle predictions'][i]
                                uni_cycle_logit = cycle_data['uni cycle logits'][i]
                                    
                            row_data = {
                                'data folder': folder,
                                'epoch': cycle_data['epoch'][i],
                                'batch index': cycle_data['batch index'][i],
                                'cycle': cycle_data['cycle'][i], 
                                'class targets': cycle_data['class targets'][i][j],
                                'uni targets': cycle_data['uni targets'][i][j], 
                                'class predictions': cycle_data['class predictions'][i][j], 
                                'uni predictions': cycle_data['uni predictions'][i][j], 
                                'uni cycle predictions': uni_cycle_prediction,
                                'uni logits': cycle_data['uni logits'][i][j],
                                'uni cycle logits': uni_cycle_logit
                            }
                            clean_exp_rows.append(row_data)

                adv_exp_rows = []
                for cycle_data in exp_adv_data:
                    for i in range(len(cycle_data['batch index'])):  # iterate over batches
                        for j in range(len(cycle_data['class targets'][i])):  # iterate over images within the batch
                            if cycle_data['cycle'][i] >= 1:
                                nonuni_cycle_prediction = cycle_data['nonuni cycle predictions'][i][j]
                                nonuni_cycle_logit = cycle_data['nonuni cycle logits'][i][j]
                            elif cycle_data['cycle'][i] == 0:
                                nonuni_cycle_prediction = cycle_data['nonuni cycle predictions'][i]
                                nonuni_cycle_logit = cycle_data['nonuni cycle logits'][i]
                            
                            row_data = {
                                'data folder': folder,
                                'epoch': cycle_data['epoch'][i],
                                'batch index': cycle_data['batch index'][i],
                                'cycle': cycle_data['cycle'][i], 
                                'class targets': cycle_data['class targets'][i][j],
                                'nonuni targets': cycle_data['nonuni targets'][i][j], 
                                'class predictions': cycle_data['class predictions'][i][j], 
                                'nonuni predictions': cycle_data['nonuni predictions'][i][j], 
                                'nonuni cycle predictions': nonuni_cycle_prediction,
                                'nonuni logits': cycle_data['nonuni logits'][i][j],
                                'nonuni cycle logits': nonuni_cycle_logit
                            }
                            adv_exp_rows.append(row_data)

                if clean_exp_rows:
                    clean_exp_rows_df = pd.DataFrame(clean_exp_rows)
                    exp_clean_df = pd.concat([exp_clean_df, clean_exp_rows_df], ignore_index=True)
                
                if adv_exp_rows:
                    adv_exp_rows_df = pd.DataFrame(adv_exp_rows)
                    exp_adv_df = pd.concat([exp_adv_df, adv_exp_rows_df], ignore_index=True)

                exp_losses['data folder'].append(folder)
                exp_losses['epoch'].append(epoch)
                exp_losses['class_losses'].append(exp_class_losses)
                exp_losses['uni_losses'].append(exp_uni_losses)
                exp_losses['UI_cycle_losses'].append(exp_UI_cycle_losses)

                exp_losses_df = pd.DataFrame(exp_losses)

        # save as csv
        # train data
        train_data_dir = f'{args.model_dir}/train_data'
        os.makedirs(train_data_dir, exist_ok=True)
        train_clean_df.to_csv(os.path.join(train_data_dir, 'train_clean_data.csv'), index=False)
        train_adv_df.to_csv(os.path.join(train_data_dir, 'train_adv_data.csv'), index=False)
        train_losses_df.to_csv(os.path.join(train_data_dir, 'train_losses.csv'), index=False)

        # val data
        val_data_dir = f'{args.model_dir}/val_data'
        os.makedirs(val_data_dir, exist_ok=True)
        val_clean_df.to_csv(os.path.join(val_data_dir, 'val_clean_data.csv'), index=False)
        val_adv_df.to_csv(os.path.join(val_data_dir, 'val_adv_data.csv'), index=False)
        val_losses_df.to_csv(os.path.join(val_data_dir, 'val_losses.csv'), index=False)

        # experiment data
        if epoch >= args.epochs-2:
        # if test_all_epochs:
            exp_data_dir = f'{args.model_dir}/exp_data'
            os.makedirs(exp_data_dir, exist_ok=True)
            exp_clean_df.to_csv(os.path.join(exp_data_dir, 'exp_clean_data.csv'), index=False)
            exp_adv_df.to_csv(os.path.join(exp_data_dir, 'exp_adv_data.csv'), index=False)
            exp_losses_df.to_csv(os.path.join(exp_data_dir, 'exp_losses.csv'), index=False)

        if epoch >= 10:
            torch.save(model.state_dict(), args.model_dir + f"/{args.model_name}{epoch}_model.pt")

if __name__ == '__main__':
    main()