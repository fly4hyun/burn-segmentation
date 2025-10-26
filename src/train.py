
###################################################################################################

import warnings
warnings.filterwarnings(action='ignore')

import numpy as np
from tqdm import tqdm
import cv2
import json
import os

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.functional import threshold, normalize
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import dataloader

###################################################################################################

model_size = 'vit_b'
model_name = ''
model_folder_path = './models'
epoch_add = 0

train_num = 0

cuda_num = 0

batch_size = 5
warmup_steps = 100
global_step = 0
epochs = 2000

lr = 1e-4
beta = [0.9, 0.999]
weight_decay = 0.2

milestone = [60000, 86666]
# gamma = 0.1
gamma = 0.95

###################################################################################################

if train_num > 0:
    model_folder_path = model_folder_path + '_' + str(train_num)

###################################################################################################

device = 'cuda:' + str(cuda_num) if torch.cuda.is_available() else 'cpu'
torch.manual_seed(234)
if device == 'cuda:' + str(cuda_num):
    torch.cuda.manual_seed_all(234)
print(device + ' is avaulable')

###################################################################################################

train_loader = DataLoader(dataloader.trainset, batch_size = batch_size, shuffle = True, num_workers = 3, pin_memory = True)
valid_loader = DataLoader(dataloader.validset, batch_size = batch_size, shuffle = False, num_workers = 3, pin_memory = True)
test_loader = DataLoader(dataloader.testset, batch_size = batch_size, shuffle = False, num_workers = 3, pin_memory = True)

###################################################################################################

print("Loading model...")

sam = sam_model_registry[model_size](checkpoint='./checkpoint/sam_vit_b_01ec64.pth')

if len(model_name.split("_")) == 2:
    sam.load_state_dict(torch.load(os.path.join(model_folder_path, model_name + '.pth')))
    
elif len(model_name.split("_")) == 3:
    lr = float(model_name.split("_")[2])
    epoch_add = int(model_name.split("_")[1])
    sam.load_state_dict(torch.load(os.path.join(model_folder_path, model_name + '.pth')))
    
sam = sam.to(device=device)

###################################################################################################

def focal_loss(pred, target, gamma=2.0, alpha=0.25, reduction='mean'):
    
    #pred = F.sigmoid(pred)
    pt = torch.where(target == 1, pred, 1-pred)
    ce_loss = F.binary_cross_entropy(pred, target, reduction="none")
    focal_term = (1 - pt).pow(gamma)
    loss = alpha * focal_term * ce_loss
    
    return loss.mean()


def dice_loss(pred, target, smooth=1.0):
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    
    intersection = (pred_flat * target_flat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (pred_flat.sum() + target_flat.sum() + smooth))

def compute_loss(pred_mask, true_mask, pred_iou, true_iou):
    
    pred_mask = F.sigmoid(pred_mask).squeeze(1).to(dtype = torch.float64)
    
    fl = focal_loss(pred_mask, true_mask)
    dl = dice_loss(pred_mask, true_mask)
    
    mask_loss = 20 * fl + dl
    iou_loss = F.mse_loss(pred_iou, true_iou)
    
    total_loss = mask_loss + iou_loss

    return total_loss


def mean_iou(preds, labels, eps=1e-6):
    
    preds = normalize(threshold(preds, 0.0, 0)).squeeze(1)

    pred_cls = (preds == 1).float()
    label_cls = (labels == 1).float()

    intersection = (pred_cls * label_cls).sum(1).sum(1)

    union = (1 - (1 - pred_cls) * (1 - label_cls)).sum(1).sum(1)

    intersection = intersection + (union == 0)
    union = union + (union == 0)

    ious = intersection / union
    
    return ious

###################################################################################################

optimizer = optim.AdamW([{'params': sam.image_encoder.parameters()}, 
                        #{'params': sam.prompt_encoder.parameters()}, 
                        {'params': sam.mask_decoder.parameters()}], 
                       lr = lr, betas = beta, weight_decay = weight_decay)

criterion = nn.CrossEntropyLoss()
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=gamma)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

###################################################################################################

print('Training Start')

best_loss = 999999999
best_mIOU = 0.0

for epoch in range(epochs):
    
    train_loss_list = []
    train_miou_list = []
    
    sam.train()
    
    iterations = tqdm(train_loader)
    for train_data in iterations:
        
        train_input = train_data['image'].to(device)
        train_target_mask = train_data['mask'].to(device, dtype = torch.float64)
        train_original_size = train_data['original_size']
        train_transform_size = train_data['transform_size']

        optimizer.zero_grad()
        train_encode_feature = sam.image_encoder(train_input)
        with torch.no_grad():
            train_sparse_embeddings, train_dense_embeddings = sam.prompt_encoder(points = None, boxes = None, masks = None)
            
        train_mask, train_IOU = sam.mask_decoder(image_embeddings = train_encode_feature, 
                                                 image_pe = sam.prompt_encoder.get_dense_pe(), 
                                                 sparse_prompt_embeddings = train_sparse_embeddings, 
                                                 dense_prompt_embeddings = train_dense_embeddings, 
                                                 multimask_output = False)
        
        train_true_iou = mean_iou(train_mask, train_target_mask, eps=1e-6)
        train_miou_list = train_miou_list + train_true_iou.tolist()
        
        train_loss_one = compute_loss(train_mask, train_target_mask, train_IOU, train_true_iou)
        train_loss_one.backward()
        optimizer.step()
        
        train_loss_list.append(train_loss_one.item())
        
        if epoch_add == 0:
            if global_step < warmup_steps:
                
                lr_scale = global_step / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 8e-4 * lr_scale
                    
            global_step += 1
        
        pbar_desc = "Model train loss --- "
        pbar_desc += f"Total loss: {np.mean(train_loss_list):.5f}"
        pbar_desc += f", total mIOU: {np.mean(train_miou_list):.5f}"
        iterations.set_description(pbar_desc)
        
    train_loss = np.mean(train_loss_list)
    train_miou = np.mean(train_miou_list)

    torch.cuda.empty_cache()
    
    sam.eval()
    scheduler.step()
    
    with torch.no_grad():
        
        valid_loss_list = []
        valid_miou_list = []
        valid_true_iou = 0
        valid_loss = 0
        valid_miou = 0
        
        iterations = tqdm(valid_loader)
        for valid_data in iterations:
            
            valid_input = valid_data['image'].to(device)
            valid_target_mask = valid_data['mask'].to(device, dtype = torch.float64)
            valid_original_size = valid_data['original_size']
            valid_transform_size = valid_data['transform_size']
        
            valid_encode_feature = sam.image_encoder(valid_input)
            valid_sparse_embeddings, valid_dense_embeddings = sam.prompt_encoder(points = None, boxes = None, masks = None)
        
            valid_mask, valid_IOU = sam.mask_decoder(image_embeddings = valid_encode_feature, 
                                                    image_pe = sam.prompt_encoder.get_dense_pe(), 
                                                    sparse_prompt_embeddings = valid_sparse_embeddings, 
                                                    dense_prompt_embeddings = valid_dense_embeddings, 
                                                    multimask_output = False)
            
            valid_true_iou = mean_iou(valid_mask, valid_target_mask, eps=1e-6)
            valid_miou_list = valid_miou_list + valid_true_iou.tolist()
        
            valid_loss_one = compute_loss(valid_mask, valid_target_mask, valid_IOU, valid_true_iou)
            
            valid_loss_list.append(valid_loss_one.item())
            
            
            pbar_desc = "Model valid loss --- "
            pbar_desc += f"Total loss: {np.mean(valid_loss_list):.5f}"
            pbar_desc += f", total mIOU: {np.mean(valid_miou_list):.5f}"
            iterations.set_description(pbar_desc)
            
        valid_loss = np.mean(valid_loss_list)
        valid_miou = np.mean(valid_miou_list)
        
    model_path = os.path.join(model_folder_path, 'sam.pth')
    sam = sam.cpu()
    torch.save(sam.state_dict(), model_path)
    sam = sam.to(device)
        
    if best_mIOU < valid_miou:
        best_loss = valid_loss
        best_mIOU = valid_miou
        model_path = os.path.join(model_folder_path, 'sam_best.pth')
        sam = sam.cpu()
        torch.save(sam.state_dict(), model_path)
        sam = sam.to(device)

    print("epoch : {:3d}, train loss : {:3.4f}, valid loss : {:3.4f}, valid mIOU : {:3.4f}\
        ( best vaild loss : {:3.4f}, best valid mIOU : {:3.4f} )".format(epoch + 1 + epoch_add, 
                                                                         train_loss, 
                                                                         valid_loss, 
                                                                         valid_miou, 
                                                                         best_loss, 
                                                                         best_mIOU
                                                                         ))
    
    lr = optimizer.param_groups[0]["lr"]
    
    if (epoch + 1) % 5 == 0 or (epoch + 1) in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        model_path = os.path.join(model_folder_path, 'sam_' + str(epoch + 1 + epoch_add) + '_' + str(round(lr, 10)) + '.pth')
        sam = sam.cpu()
        torch.save(sam.state_dict(), model_path)
        sam = sam.to(device)

###################################################################################################