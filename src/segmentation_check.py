





###################################################################################################

import warnings
warnings.filterwarnings(action='ignore')

import os
from tqdm import tqdm
import cv2
import random

import torch
from torch.nn import functional as F
from torch.nn.functional import threshold, normalize
import torchvision
from torchvision.utils import save_image

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

###################################################################################################

model_size = 'vit_b'
model_path = './models'
original_image_path = './processing_data/original_image'
original_mask_path = './burn/mask'
processed_image_path = './processing_data/processed_image'
processed_mask_path = './processing_data/processed_mask'
mask_check_path = './check_image'

cuda_num = 0
train_num = 3

###################################################################################################

if train_num > 0:
    model_path = model_path + '_' + str(train_num)

###################################################################################################

device = 'cuda:' + str(cuda_num) if torch.cuda.is_available() else 'cpu'
torch.manual_seed(234)
if device == 'cuda:' + str(cuda_num):
    torch.cuda.manual_seed_all(234)
print(device + ' is avaulable')

###################################################################################################

model_list = os.listdir(model_path)
model_list.remove('sam.pth')
model_list.remove('sam_best.pth')

model_list = sorted(model_list, key = lambda model_list : int(model_list.split("_")[1]))

sam = sam_model_registry[model_size](checkpoint='./checkpoint/sam_vit_b_01ec64.pth')

###################################################################################################

transform = ResizeLongestSide(1024)
preprocess = sam.preprocess
postprocess_masks = sam.postprocess_masks

###################################################################################################

for model_name in model_list:
    
    ##################################################################
    
    Next_model_botton = input('Model Epoch is {:3d} ( {:3d} / {:3d} ), Go to Next Model, Put the n : '.format(int(model_name.split("_")[1]), model_list.index(model_name) + 1, len(model_list)))
    if Next_model_botton == 'n':
        continue
    
    ##################################################################
    
    print('##################################################################')
    print('now ( {} ) model loading ... ( {:3d} / {:3d} ) '.format(model_name, \
        model_list.index(model_name) + 1, \
            len(model_list)))
    
    sam.load_state_dict(torch.load(os.path.join(model_path, model_name)))
    sam = sam.to(device=device)
    
    print('Model Load Done ...')
    
    ##################################################################
    
    original_image_list = os.listdir(original_image_path)
    random.shuffle(original_image_list)
    
    if len(original_image_list) == 0:
        print(' ~ Fin ~.')
        break

    ##################################################################
    
    start_image_count = len(original_image_list)
    
    iterations = tqdm(original_image_list)
    for image_name in iterations:
        
        ##################################################################
        
        try:
            img = cv2.imread(os.path.join(original_image_path, image_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        except:
            print('Image : {} ... error'.format(image_name[:-4]))
            continue
        
        original_size = img.shape[:2]
        
        img_original = torch.tensor(img)
        img_original = img_original.permute(2, 0, 1).contiguous()[None, :, :, :]
        
        img_processed = transform.apply_image(img)
        img_processed = torch.as_tensor(img_processed).to(device)
        img_processed = img_processed.permute(2, 0, 1).contiguous()[None, :, :, :].squeeze(0)
        
        img_processed_size = img_processed.shape[-2:]

        img_processed = preprocess(img_processed).unsqueeze(0)

        ##################################################################
        
        try:
            box = cv2.imread(os.path.join(original_mask_path, image_name), cv2.IMREAD_GRAYSCALE)
            box = torch.tensor(box).unsqueeze(0).unsqueeze(0) / 255
        except:
            box = torch.zeros(1, 1, original_size[0], original_size[1])
        
        box = torch.cat([box, box, box], dim = 1)

        ##################################################################
        
        with torch.no_grad():
            
            img_feature = sam.image_encoder(img_processed)
            img_sparse_embeddings, img_dense_embeddings = sam.prompt_encoder(points = None, boxes = None, masks = None)
            mask, iou = sam.mask_decoder(image_embeddings = img_feature, \
                image_pe = sam.prompt_encoder.get_dense_pe(), \
                    sparse_prompt_embeddings = img_sparse_embeddings, \
                        dense_prompt_embeddings = img_dense_embeddings, \
                            multimask_output = False)

            mask = postprocess_masks(mask, img_processed_size, original_size).cpu()
            mask = normalize(threshold(mask, 0.0, 0)).cpu()
        
        ##################################################################
        
        print('Image : {} ... Predict IOU : {:.4f} ( n : {:3d} )'.format(image_name[:-4], iou.item(), int(model_name.split("_")[1])))
        
        img_original = (img_original / 255).float()
        mask = torch.cat([mask, mask, mask], dim = 1)
        masked_image = img_original + mask * 0.3
        masked_image_m = img_original + mask * 0.3 - box * 0.2
        cap = mask * box
        u_or = mask + box - mask * box
        
        x, y = original_size
        x_n = x
        y_n = y

        x_n = 800
        y_n = int(x_n * y / x)
        
        img_original_n = torchvision.transforms.Resize((x_n, y_n))(img_original)
        img_original_mask_n = torchvision.transforms.Resize((x_n, y_n))(img_original + box * 0.2)
        masked_image_n = torchvision.transforms.Resize((x_n, y_n))(masked_image)
        masked_image_m_n = torchvision.transforms.Resize((x_n, y_n))(masked_image_m)
        mask_n = torchvision.transforms.Resize((x_n, y_n))(mask)
        box_n = torchvision.transforms.Resize((x_n, y_n))(box)

        check_image_n = torch.cat([img_original_n, img_original_mask_n, box_n], dim = 0)
        check_image_m = torch.cat([masked_image_n, masked_image_m_n, mask_n], dim = 0)
        check_image = torch.cat([check_image_n, check_image_m], dim = 2)
        save_image(check_image, os.path.join(mask_check_path, 'check_image.png'))
        
        ##################################################################
        
        check_num = input('If the mask is satisfactory, enter the number 1. If not, enter 2 (Next Model - put n, delet : d, cap : c, or : u, only box : b) : ')
        if check_num == 'n':
            break
        
        if check_num == 'd':
            os.remove(os.path.join(original_image_path, image_name))
            continue
        
        ##################################################################

        if check_num == 'c':
            
            print('Image : {} Saving ... '.format(image_name[:-4]))
            save_image(cap, os.path.join(processed_mask_path, image_name))
            save_image(img_original, os.path.join(processed_image_path, image_name))
            os.remove(os.path.join(original_image_path, image_name))
        
        ##################################################################

        if check_num == 'u':
            
            print('Image : {} Saving ... '.format(image_name[:-4]))
            save_image(u_or, os.path.join(processed_mask_path, image_name))
            save_image(img_original, os.path.join(processed_image_path, image_name))
            os.remove(os.path.join(original_image_path, image_name))
        
        ##################################################################
            
        if check_num == 'b':
            
            print('Image : {} Saving ... '.format(image_name[:-4]))
            save_image(box, os.path.join(processed_mask_path, image_name))
            save_image(img_original, os.path.join(processed_image_path, image_name))
            os.remove(os.path.join(original_image_path, image_name))
        
        ##################################################################
        
        if check_num == '1':
            
            print('Image : {} Saving ... '.format(image_name[:-4]))
            save_image(mask, os.path.join(processed_mask_path, image_name))
            save_image(img_original, os.path.join(processed_image_path, image_name))
            os.remove(os.path.join(original_image_path, image_name))
        
    ##################################################################
    
    original_image_list = os.listdir(original_image_path)
    end_image_count = len(original_image_list)
    
    print('Processed Image : {:4d} ( / {:4d})'.format(start_image_count - end_image_count, start_image_count))
    print('Next Model ... ')
    
    ##################################################################
    
    



########BBOX 도 출력!!!!!!!!!!!!!!!!

















