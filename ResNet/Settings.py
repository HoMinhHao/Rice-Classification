import setGPU
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from PIL import Image
import timm
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.onnx
from Tools import EarlyStopping
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import os
import random
from albumentations.pytorch import ToTensorV2
from albumentations import (Resize, Compose, Flip, ColorJitter, Normalize, CoarseDropout, CenterCrop, Sharpen, Transpose, RandomResizedCrop)

CFG = {
    'seed': 194,
    'img_size': 224,
    'epochs': 10000,
    'train_bs': 64,
    'valid_bs': 64,
    'num_workers': 10,
    'accum_iter': 2, 
    'verbose_step': 1,
}
MapToClass = {'1121': 0,
              '1401': 1,
              '1509': 2,
              'pr-11': 3,
              'rh-10': 4,
              'sharbati': 5,
              'sona-masoori': 6,
              'sugandha': 7}

class Dataset():
    def __init__(self, path, transforms=None):
        super().__init__()
        self.path = path
        self.transforms = transforms

        self.ListImg = []
        self.ListLabel = []
        for classes in os.listdir(self.path):
            subdir = os.path.join(self.path, classes)
            for img in os.listdir(subdir):
                ImgPath = os.path.join(subdir, img)
                self.ListImg.append(ImgPath)
                self.ListLabel.append(MapToClass[classes])
                
    def __len__(self):
        return len(self.ListImg)

    def get_img(path ):
        im_bgr = cv2.imread(path)
        im_rgb = im_bgr[:, :, [2, 1, 0]]
        return im_rgb

    def __getitem__(self, index: int):
        target = self.ListLabel[index]
        
        img  = Dataset.get_img(self.ListImg[index])
        img = self.transforms(image=img)['image']
        return img, target

def get_train_transforms():
    interpolation = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC][1]
    Range = (0, 0.2)
    return Compose([
            Resize(CFG['img_size'] * 2, CFG['img_size'] * 2, interpolation = interpolation),
            RandomResizedCrop(CFG['img_size'], CFG['img_size']),
            Sharpen(alpha=1, always_apply=True),
            Transpose(p=0.01),
            Flip(p=0.01),
            ColorJitter(brightness=Range, contrast=Range, saturation=Range, hue=Range, p=0.01),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.01),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_valid_transforms():
    interpolation = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC][1]
    return Compose([
            Resize(CFG['img_size'] * 2, CFG['img_size'] * 2, interpolation = interpolation),
            CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
            Sharpen(alpha=1, always_apply=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

def prepare_dataloader():
    training_set = Dataset(path='/home/jupyter-iec_haoquy/iec_hai/original/MobileNetV2/Dataset_2/train',
                           transforms=get_train_transforms())
    validation_set = Dataset(path='/home/jupyter-iec_haoquy/iec_hai/original/MobileNetV2/Dataset_2/valid',
                           transforms=get_valid_transforms())
    
    training_loader = torch.utils.data.DataLoader(training_set, 
                                                  batch_size=CFG['train_bs'], 
                                                  num_workers=CFG['num_workers'],
                                                  shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set,
                                                    batch_size=CFG['train_bs'], 
                                                    num_workers=CFG['num_workers'],
                                                    shuffle=True)
    print("Load data successfully!")
    return training_loader, validation_loader
    

def TrainModel(epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False):
        model.train()

        running_loss = None

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (imgs, image_labels) in pbar:
            imgs = imgs.to(device).float()
            image_labels = image_labels.to(device).long()

            scaler = GradScaler()
            with autocast():
                image_preds = model(imgs)

                loss = loss_fn(image_preds, image_labels)
                
                scaler.scale(loss).backward()

                if running_loss is None:
                    running_loss = loss.item()
                else:
                    running_loss = running_loss * .99 + loss.item() * .01

                if ((step + 1) %  CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad() 
                    
                    if scheduler is not None and schd_batch_update:
                        scheduler.step()

                if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
                    description = f'Epoch {epoch} loss: {running_loss:.4f}'
                    
                    pbar.set_description(description)
                  
        if scheduler is not None and not schd_batch_update:
           scheduler.step()
    
def EvalModel(epoch, model, loss_fn, val_loader, device, early_stopping, scheduler=None, schd_loss_update=False):
        model.eval()

        loss_sum = 0
        sample_num = 0
        image_preds_all = []
        image_targets_all = []
        
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        for step, (imgs, image_labels) in pbar:
            imgs = imgs.to(device).float()
            image_labels = image_labels.to(device).long()
            
            image_preds = model(imgs)
            image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
            image_targets_all += [image_labels.detach().cpu().numpy()]
            
            loss = loss_fn(image_preds, image_labels)
            
            loss_sum += loss.item() * image_labels.shape[0]
            sample_num += image_labels.shape[0]  

            if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
                description = f'Epoch {epoch} loss: {loss_sum/sample_num:.4f}'
                pbar.set_description(description)
        
        image_preds_all = np.concatenate(image_preds_all)
        image_targets_all = np.concatenate(image_targets_all)
        report = classification_report(image_targets_all, image_preds_all, digits=4)
        
        print ("Classification report: ", report)
        print ("F1 micro averaging:",(f1_score(image_targets_all, image_preds_all, average='micro')))
            
        print('Validation loss', loss_sum/sample_num)
        print('Validation accuracy', (image_preds_all==image_targets_all).mean())
        early_stopping(loss_sum/sample_num, model, report)
          
        if scheduler is not None:
            if schd_loss_update:
                scheduler.step(loss_sum/sample_num)
            else:
                scheduler.step()
        return (image_preds_all==image_targets_all).mean(), loss_sum/sample_num

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True