








import cv2
import numpy as np
from scipy import ndimage
import os
from tqdm import tqdm
import csv




original_mask_path = './mask'
original_image_path = './image'

original_mask_list = os.listdir(original_mask_path)
iterations = tqdm(original_mask_list)

threshold_value = 100
min_area = 100


ii = 0

for mask_name in iterations:

    
    # 마스크 경로
    mask_path = os.path.join(original_mask_path, mask_name)
    image_path = os.path.join(original_image_path, mask_name)

    # 마스크 읽기
    mask = cv2.imread(mask_path)#, cv2.IMREAD_GRAYSCALE)
    #image = cv2.imread(image_path)#, cv2.IMREAD_GRAYSCALE)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    min_area = int((mask.shape[0] * mask.shape[1]) / 200) * 255
    
    if mask is None:
        raise FileNotFoundError(f"{mask_path} not found")
    
    # binary mask로 변환
    _, binary_mask = cv2.threshold(mask, threshold_value, 255, cv2.THRESH_BINARY)
    #_, binary_mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    # 라벨링
    labeled, num_features = ndimage.label(binary_mask)

    # 이미지 읽어오기
    image = cv2.imread(image_path)
    
    # 결과 마스크를 저장할 디렉토리 생성(없으면)
    masked_dir = 'masked_each'

    # 라벨링된 각 영역에 대하여
    for i in range(1, num_features + 1):
        component = (labeled == i).astype('uint8') * 255  # 원본 마스크 이미지와 동일한 크기의 마스크 이미지 생성

        if np.sum(component) < min_area:
            continue
        
        mini_dict = {}
        
        color_mask = cv2.cvtColor(component, cv2.COLOR_GRAY2BGR)
        masked_result = image - (255 - color_mask) * 0.3

        masked_result = np.concatenate((image, masked_result, color_mask), axis = 1)
        
        
        height, width = masked_result.shape[:2]
        
        height_n = 800
        width_n = int(height_n * width / height)
        
        # 이미지 크기 변경
        masked_result = cv2.resize(masked_result, (width_n, height_n))
        
        # 결과 마스크 파일 이름 생성
        base_name = os.path.basename(mask_path).split('.')[0]  # 파일 확장자 제거
        extension_name = os.path.basename(mask_path).split('.')[1]

        result_masked_path = os.path.join(masked_dir, f"{base_name}_M{str(i).zfill(3)}.{extension_name}")
        
        # 결과 마스크 저장
        cv2.imwrite(result_masked_path, masked_result)


        

        
        







