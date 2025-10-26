










###################################################################################################

import warnings
warnings.filterwarnings(action='ignore')

import numpy as np
from tqdm import tqdm
import cv2
import json
import os
import time
from PIL import Image

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.functional import threshold, normalize
import torchvision
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import dataloader

###################################################################################################

batch_size = 1

###################################################################################################


device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(234)
if device == 'cuda:2':
    torch.cuda.manual_seed_all(234)
print(device + ' is avaulable')

###################################################################################################

test_loader = DataLoader(dataloader.testset, batch_size = batch_size, shuffle = False, num_workers = 3, pin_memory = True)

###################################################################################################

print("Loading model...")

model_name = 'sam_best'

sam = sam_model_registry['vit_b'](checkpoint='./checkpoint/sam_vit_b_01ec64.pth')
sam.load_state_dict(torch.load('./models/' + model_name + '.pth'))

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
    
    pred_mask = F.sigmoid(pred_mask).squeeze(1)
    
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

criterion = nn.CrossEntropyLoss()
resize = transforms.Resize((256, 256))
pixel_std = sam.pixel_std
pixel_mean = sam.pixel_mean
postprocess_masks = sam.postprocess_masks
img_size = sam.image_encoder.img_size

###################################################################################################

print('Test Start')

sam.eval()

miou_list = []
images = []

i = 0

with torch.no_grad():
    iterations = tqdm(test_loader)
    for test_data in iterations:
        
        test_input_ori = test_data['image'].to(device)#, dtype = torch.float64)
        test_target_mask = test_data['mask'].to(device, dtype = torch.float64)
        test_original_size = test_data['original_size']
        test_transform_size = test_data['transform_size']
        
        # print(test_target_mask)
        # print(test_target_mask.shape)
        # save_image(test_target_mask, 'output/image_name.png')
        
        # continue
        
        test_encode_feature = sam.image_encoder(test_input_ori)
        test_sparse_embeddings, test_dense_embeddings = sam.prompt_encoder(points = None, boxes = None, masks = None)
            
        test_mask, test_IOU = sam.mask_decoder(image_embeddings = test_encode_feature, 
                                                    image_pe = sam.prompt_encoder.get_dense_pe(), 
                                                    sparse_prompt_embeddings = test_sparse_embeddings, 
                                                    dense_prompt_embeddings = test_dense_embeddings, 
                                                    multimask_output = False)
        
        #test_mask = sam.postprocess_masks(test_mask, test_transform_size, test_original_size)
        
        test_true_iou = mean_iou(test_mask, test_target_mask, eps=1e-6)
        
        miou_list = miou_list + test_true_iou.tolist()

        #test_mask = postprocess_masks(test_mask, test_transform_size, test_original_size).cpu()
        test_mask = normalize(threshold(test_mask, 0.0, 0))#.cpu()
        # print(test_mask.shape)
        # h, w = test_mask.shape[-2:]
        
        # padh = img_size - h
        # padw = img_size - w
        
        # test_mask = F.pad(test_mask, (0, padw, 0, padh))
        # test_mask = resize(test_mask).squeeze(0)
        # test_mask = (test_mask != 0) * 1
        
        # test_mask = test_mask.unsqueeze(0)
        
        #test_input_ori = (test_input_ori / 255).float()
        test_input_ori = ((test_input_ori * pixel_std) + pixel_mean) / 255
        test_input_ori = transforms.Resize((1024, 1024))(test_input_ori)
        test_mask = transforms.Resize((1024, 1024))(test_mask)
        test_target_mask = transforms.Resize((1024, 1024))(test_target_mask)
        test_mask = torch.cat([test_mask, test_mask, test_mask], dim = 1)
        test_target_mask = test_target_mask.unsqueeze(1)#.cpu()
        box = torch.cat([test_target_mask, test_target_mask, test_target_mask], dim = 1)
        
        masked_image = test_input_ori + test_mask * 0.2
        masked_image_m = test_input_ori + test_mask * 0.2 - box * 0.1

        x, y = test_original_size
        x_n = x
        y_n = y

        x_n = 800
        y_n = int(x_n * y / x)
        

        
        check_image_n = torch.cat([test_input_ori, test_input_ori + box * 0.2, box], dim = 0)
        check_image_m = torch.cat([masked_image, masked_image_m, test_mask], dim = 0)
        check_image = torch.cat([check_image_n, check_image_m], dim = 2)
        


        image_path = 'output/image' + str(i) + '.png'
        save_image(check_image, image_path)
        save_image(check_image, 'output/image.png')
        
        #image = tf_toPILImage(result_daat[0])
        images.append(Image.open(image_path))

        
        
        pbar_desc = "Model test --- "
        pbar_desc += f"Total mIOU: {np.mean(miou_list):.5f}"
        iterations.set_description(pbar_desc)
        
        i = i + 1
        
    mean_IOU = np.mean(miou_list)
    print(np.mean(miou_list))
    
    images[0].save('results/' + str(mean_IOU)[:6] + '_' + model_name + '.gif', save_all=True, append_images=images[1:],loop=0xff, duration=3000)


###################################################################################################













