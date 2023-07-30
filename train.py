import argparse
import numpy as np
import pandas as pd
import os
import math
import cv2
from tqdm import tqdm
import gc
import torch
import datetime
import torch.nn as nn
import torch.nn.functional as F
from h_transformer_1d import HTransformer1D
from torch import optim
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
from monai.transforms import ScaleIntensity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data import DistributedSampler

import random
import albumentations
from image_enhancement import image_enhancement


from albumentations.pytorch import ToTensorV2
from transformers import get_cosine_schedule_with_warmup
import timm
from timm.utils import AverageMeter
import glob
import sys
import time
import pydicom
from skimage.filters import threshold_otsu

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

from helper import *
from tabnet.tab_network import *
from transformer.transformers import Transformer as T

sys.path.append("models")
from models.nextvit.classification import utils
from models.nextvit.classification.nextvit import nextvit_base, nextvit_large
from models.metaformer import convformer_b36, caformer_b36, caformer_s36_384_in21ft1k, convformer_s18
from models.TinyViT.models.tiny_vit import tiny_vit_21m_512
import itertools
from ignite.distributed import DistributedProxySampler
from torch.nn.parallel import DistributedDataParallel as ddp
import torch.distributed as dist


def pfbeta(labels, predictions, beta=1.):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0

def optimal_f1(labels, predictions):
    thres = np.linspace(0, 1, 101)
    f1s = [pfbeta(labels, predictions > thr) for thr in thres]
    idx = np.argmax(f1s)
    return f1s[idx], thres[idx]

def validate(model, loader, df, device, criterion, get_output=False):
    
    model.eval()
    val_loss = []
    LOGITS = []
    PREDS = []
    TARGETS = []
    IDS = []
    with torch.no_grad():
        for data, target, ids, tab in tqdm(loader, total=len(loader)):

            data, target= data.float().to(device), target.to(device)
            logits = model(data)

            loss = criterion(logits, target)
            PREDS.append(torch.sigmoid(logits).squeeze())
            TARGETS.append(target.squeeze().cpu().numpy())
            #PREDS_OUT.append(pred_out.cpu().squeeze().numpy())
            IDS.append(np.asarray(ids))

            val_loss.append(loss.detach().cpu().numpy())
        val_loss = np.mean(val_loss)

    LOGITS = []    
    PREDS = torch.concat(PREDS).cpu().numpy()
    TARGETS = np.concatenate(TARGETS)

    IDS = list(itertools.chain(*IDS))
    
    val_df = pd.DataFrame({'pred_id':IDS, 'preds':PREDS.tolist(), 'target' :TARGETS.tolist()})
    val_df_mean = val_df.groupby('pred_id').mean().reset_index()
    val_df_max = val_df.groupby('pred_id').max().reset_index()
    roc_mean = roc_auc_score(val_df_mean['target'].values.tolist(), val_df_mean['preds'].values.tolist())
    f1_mean = f1_score(val_df_mean['target'].values.tolist(), val_df_mean['preds'].round().values.tolist())
    pf1_mean, thr_mean= optimal_f1(val_df_mean['target'].values.tolist(), val_df_mean['preds'].values.tolist())
    
    roc_max = roc_auc_score(val_df_max['target'].values.tolist(), val_df_max['preds'].values.tolist())
    f1_max = f1_score(val_df_max['target'].values.tolist(), val_df_max['preds'].round().values.tolist())
    pf1_max, thr_max= optimal_f1(val_df_max['target'].values.tolist(), val_df_max['preds'].values.tolist())

    #print(pf1)
    
    ret={'loss' : val_loss,
        'roc_mean' : roc_mean,
        'roc_max' : roc_max,
        'f1_mean' : f1_mean,
        'f1_max' : f1_max,
        'pf1_mean' : pf1_mean,
        'pf1_max' : pf1_max,
        'th_mean' : thr_mean,
        'th_max' : thr_max,}
    
    if get_output:
        return LOGITS
    
    else:
        return ret

def get_values(value):
    return value.values.reshape(-1, 1)
    
class rDataset(torch.utils.data.Dataset):
    def __init__(self, df, tab, image_size, transform = None):
        self.df = df
        self.tab = tab
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        patient_id = self.df['patient_id'].iloc[i]
        image_id = self.df['image_id'].iloc[i]
        target = self.df['cancer'].iloc[i]
        ptarget = self.df['patient_target'].iloc[i]
        img = cv2.imread(f'crop_1536/{patient_id}_{image_id}.png', cv2.IMREAD_GRAYSCALE)    
        img = np.stack([(img), (img), (img)], axis=2)
        if self.transform is not None:
            img = self.transform(image=img)['image']
        img = img/255.
        img = img.transpose(2,0,1)
        img = torch.tensor(img, dtype=float)
        tab = self.tab.drop('fold', axis=1)
        tab = tab.iloc[i]
        label = torch.tensor(np.stack([target]), dtype=float)
        return img, label, ptarget, torch.as_tensor(tab)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, help="set fold")
    parser.add_argument("--image-size", type=int, default=320, help="set image size")
    args = parser.parse_args()
    dist.init_process_group("nccl", timeout = datetime.timedelta(hours=3))
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    
    folds = 4
    seed = 1437
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    df = pd.read_csv('my_train_group_fix.csv')
    tabd= pd.read_csv('tabular.csv')
    print(len(df), len(tabd))
    
    # hyperparameters
    learning_rate = 3e-5
    num_epoch = 4
    batch_size = 2
    weight_decay = 1e-5
    
    # training
    train_transform = albumentations.Compose([
            albumentations.LongestMaxSize(max_size=1536,p=1),
            albumentations.PadIfNeeded(min_width = 930, p=1),
            albumentations.ShiftScaleRotate(     
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=20, p=0.3),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomRotate90(p=0.2),
            albumentations.OneOf([
                albumentations.GridDistortion(p=0.3),
                albumentations.OpticalDistortion(p=0.3)],p=1.0),          
            albumentations.RandomBrightnessContrast(brightness_limit=0.03, contrast_limit = 0.03, p=0.2),
            albumentations.CLAHE(clip_limit=0.03, tile_grid_size=(8, 8), p=0.2),
            albumentations.Perspective((0.05,0.09), p=0.3),
            albumentations.CoarseDropout(max_holes=4, max_height=8, max_width=8, fill_value=0, always_apply=False, p=0.3),
    ])
    test_transform = albumentations.Compose([
        albumentations.LongestMaxSize(max_size=1536,p=1),
        albumentations.PadIfNeeded(min_width=930,p=1),
    ])
    for fold in tqdm(range(folds)):
        print('*' * 30)
        print(f'fold {fold} training start!')
        print('*' * 30)
        df_train = df.loc[df['fold']!=fold][:].reset_index(drop=True)
        df_valid = df.loc[df['fold']==fold][:].reset_index(drop=True)
        tab_train = tabd.loc[tabd['fold']!=fold].reset_index(drop=True)
        tab_valid = tabd.loc[tabd['fold']==fold].reset_index(drop=True)
        class_counts = df_train['cancer'].value_counts().to_list() 
        num_samples = sum(class_counts) 
        labels = df_train['cancer'].to_list()
        
        class_weights = [1.0, 16.0] 
        weights = [class_weights[labels[i]] for i in range(int(num_samples))]
        sampler = DistributedProxySampler(
            WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples)),
            2,
            rank = rank)
        
        train_set = rDataset(df = df_train[:],tab = tab_train, image_size=768, transform=train_transform)
        train_loader = DataLoader(train_set, sampler=sampler, batch_size=batch_size, num_workers=4, pin_memory=False, drop_last=True)
        test_set = rDataset(df = df_valid[:], tab = tab_valid, image_size=768, transform=test_transform)
        valid_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=False)
        best_score = -1
        
        scaler = torch.cuda.amp.GradScaler()
        model = timm.create_model('convnext_large', num_classes=1, pretrained=True, drop_rate=0.4).to(device)
        model = ddp(model, device_ids=[device])

        gpus=2
        
        criterion = nn.BCEWithLogitsLoss().to(device)
        num_train_steps = int(len(df_train)/(batch_size*gpus))
        print(num_train_steps)

        for ep in range(num_epoch):
            optimizer = torch.optim.RAdam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_train_steps )
            model.train()
            losses = AverageMeter()
            for j, (images, labels, _, tab) in tqdm(enumerate(train_loader), total = len(train_loader) ):
                optimizer.zero_grad()  
                with torch.cuda.amp.autocast(enabled=True):
                    y_pred = model(images.float().to(device))
                    loss = criterion(y_pred, labels.to(device))


                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.update(loss)
                scheduler.step()

            dist.barrier()
            
            model.eval()
            val_loss = []
            LOGITS = []
            PREDS = []
            TARGETS = []
            IDS = []
            with torch.no_grad():
                for data, target, ids, tab in tqdm(valid_loader, total=len(valid_loader)):
                    with torch.cuda.amp.autocast(enabled=False):
                        logits = model(data.float().to(device))

                        loss = criterion(logits, target.to(device))
                    PREDS.append(torch.sigmoid(logits).squeeze())
                    TARGETS.append(target.squeeze().cpu().numpy())
                    IDS.append(np.asarray(ids))

                    val_loss.append(loss.detach().cpu().numpy())
                val_loss = np.mean(val_loss)
            if(rank==0):
                LOGITS = []    
                PREDS = torch.concat(PREDS).cpu().numpy()
                TARGETS = np.concatenate(TARGETS)

                IDS = list(itertools.chain(*IDS))

                val_df = pd.DataFrame({'pred_id':IDS, 'preds':PREDS.tolist(), 'target' :TARGETS.tolist()})
                val_df_mean = val_df.groupby('pred_id').mean().reset_index()
                val_df_max = val_df.groupby('pred_id').max().reset_index()
                val_df.to_csv('asdf.csv',index=False)
                roc_mean = roc_auc_score(val_df_mean['target'].values.tolist(), val_df_mean['preds'].values.tolist())
                f1_mean = f1_score(val_df_mean['target'].values.tolist(), val_df_mean['preds'].round().values.tolist())
                pf1_mean, thr_mean= optimal_f1(val_df_mean['target'].values.tolist(), val_df_mean['preds'].values.tolist())
                
                roc_max = roc_auc_score(val_df_max['target'].values.tolist(), val_df_max['preds'].values.tolist())
                f1_max = f1_score(val_df_max['target'].values.tolist(), val_df_max['preds'].round().values.tolist())
                pf1_max, thr_max= optimal_f1(val_df_max['target'].values.tolist(), val_df_max['preds'].values.tolist())

                #print(pf1)
                
                ret={'loss' : val_loss,
                    'roc_mean' : roc_mean,
                    'roc_max' : roc_max,
                    'f1_mean' : f1_mean,
                    'f1_max' : f1_max,
                    'pf1_mean' : pf1_mean,
                    'pf1_max' : pf1_max,
                    'th_mean' : thr_mean,
                    'th_max' : thr_max,}
                    
                print('ep {} train loss : {:.4f} valid loss : {:.4f} f1_mean : {:.4f} roc_mean : {:.4f}'.format(ep, losses.avg, ret['loss'], ret['f1_mean'], ret['roc_mean']), flush=True)
                print('ep {} train loss : {:.4f} valid loss : {:.4f} f1_max : {:.4f} roc_max : {:.4f}'.format(ep, losses.avg, ret['loss'], ret['f1_max'], ret['roc_max']), flush=True)
                
                print('mean', ret['pf1_mean'], ret['th_mean'])
                print('max', ret['pf1_max'], ret['th_max'])
                target_pf = max(ret['pf1_mean'], ret['pf1_max'])
            
                torch.save(model.state_dict(), f'output/tf_fold{fold}_{ep}')
                if(target_pf> best_score):
                    print('Best pf1 Score Updated')
                    best_score = target_pf
            dist.barrier()
        if rank == 0 :
            print(f'fold {fold} best_score : {best_score}')
        
if __name__ == "__main__":
    main()
