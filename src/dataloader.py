




###################################################################################################

import numpy as np
import os
import cv2
import csv
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

###################################################################################################

sam = sam_model_registry['vit_b'](checkpoint='./checkpoint/sam_vit_b_01ec64.pth')

###################################################################################################

# image_infos = {}
# with open('./burn/burn_label.csv', 'r') as f:
#     rdr = csv.reader(f)
    
#     image_name = 'ori_name'
    
#     for line in rdr:
        
#         if line[0] == 'ori_name':
#             continue
        
#         if line[0] != image_name:
#             if image_name != 'ori_name':
#                 image_infos[image_name] = bbox
#             image_name = line[0]
#             bbox = int(line[6])
#         else:
#             if bbox < int(line[6]):
#                 bbox = int(line[6])

###################################################################################################

names = os.listdir('./burn/mask')
names_image = os.listdir('./burn/image')

for name in names:
    if name not in names_image:
        names.remove(name)

# for name in names:
#     if name not in image_infos.keys():
#         names.remove(name)

random.seed(234)
random.shuffle(names)
names_len = len(names)

train_names = names[:names_len//10*8]
valid_names = names[names_len//10*8:names_len//10*9]
test_names = names[names_len//10*9:]

###################################################################################################

class CustomDataset(Dataset):
    def __init__(self, data_type = "train"):
        
        if data_type == "train":
            self.names = train_names
        if data_type == "valid":
            self.names = valid_names
        if data_type == "test":
            self.names = test_names
            
        self.data_type = data_type
            
        self.transform = ResizeLongestSide(1024)
        self.preprocess = sam.preprocess
        self.img_size = sam.image_encoder.img_size
        self.resize = transforms.Resize((256, 256))
        
        self.trans = transforms.RandomApply([
            #transforms.RandomErasing(p=0.2, scale=(0.01, 0.1), ratio=(0.3, 3.3), value=0, inplace=False),
            transforms.RandomHorizontalFlip(), 
            transforms.RandomVerticalFlip(), 
            transforms.RandomAffine(180, shear = 10)
        ], p = 0.3)

    def __len__(self):
        
        return len(self.names)
    
    def __getitem__(self, idx):
        
        #####################################

        name = self.names[idx]
        image_path = os.path.join('./burn', 'image', name)
        mask_path = os.path.join('./burn', 'mask', name)

        #####################################
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        original_size = img.shape[:2]
        
        img = self.transform.apply_image(img)
        
        img = torch.as_tensor(img)
        img = img.permute(2, 0, 1).contiguous()[None, :, :, :].squeeze(0)
        transform_size = tuple(img.shape[-2:])
        
        #####################################
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)#[:, :, np.newaxis]
        mask = self.transform.apply_image(mask)
        mask = torch.as_tensor(mask)
        
        #####################################

        if self.data_type == 'train':
            mask = mask.unsqueeze(0)
            mask = torch.cat([mask, mask, mask], dim = 0)
            
            mask_img = torch.cat([img, mask], dim = 0)
            
            mask_img = transforms.RandomResizedCrop(transform_size, scale=(0.7, 1.0), ratio=(0.7, 1.3))(mask_img)
            
            mask_img = self.trans(mask_img)

            img = mask_img[:3]
            mask = mask_img[3]

            img = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)(img)

        #####################################
        
        img = self.preprocess(img)#.squeeze(0)
        
        #####################################
        
        mask = mask.unsqueeze(0)
        
        h, w = mask.shape[-2:]
        
        padh = self.img_size - h
        padw = self.img_size - w
        
        mask = F.pad(mask, (0, padw, 0, padh))
        #mask = self.preprocess(mask)#.squeeze(0)
        mask = self.resize(mask).squeeze(0)
        mask = (mask != 0) * 1
        
        #####################################

        data = {
            'image': img, 
            'mask': mask, 
            'original_size': original_size, 
            'transform_size': transform_size,
            #'label': image_infos[name]
        }
        
        return data


###################################################################################################

trainset = CustomDataset('train')
validset = CustomDataset('valid')
testset = CustomDataset('test')

###################################################################################################

if __name__ == '__main__':
    
    data_loader = iter(trainset)
    train_data = next(data_loader)

    train_input = train_data['image']
    train_target_mask = train_data['mask']
    train_original_size = train_data['original_size']
    train_transform_size = train_data['transform_size']
    #print(train_target_mask)
    #print((train_target_mask==255).sum())
    
    print(train_input.shape)
    print(train_target_mask.shape)
    
    #print(train_original_size)
    #print(train_transform_size)
    
    
    
    
    asfdasdf
    
    
    print('next')
    
    train_data = next(data_loader)

    train_input = train_data['image']
    train_target_mask = train_data['mask']
    train_original_size = train_data['original_size']
    train_transform_size = train_data['transform_size']
    
    print(train_input.shape)
    print(train_target_mask.shape)
    print(train_original_size)
    print(train_transform_size)
    
    
    

###################################################################################################