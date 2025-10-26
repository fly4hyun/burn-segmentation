








import cv2
import numpy as np
from scipy import ndimage
import os
from tqdm import tqdm
import csv



filename = 'burn_info.csv'

# csv 읽기
data_dict = {}
new_data_info = []
header_name = ['name', 'mask name', 'min x', 'min y', 'max x', 'max y', 'location', 'team', 'info', 'date', 'gender', 'age', 'parts', 'source', 'days', 'YN', 'Num']

with open(filename, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # 'Name' 필드를 분리하여 키와 값으로 사용
        name = row['name']
        rest_of_values = [row['location'], row['team'], row['info']]
        
        # 최종 딕셔너리에 추가
        data_dict[name] = rest_of_values



original_mask_path = './mask'
original_image_path = './image'

original_mask_list = os.listdir(original_mask_path)
iterations = tqdm(original_mask_list)

threshold_value = 100
min_area = 100


ii = 0

for mask_name in iterations:

    
    if mask_name not in data_dict.keys():
        continue
    
    rest_of_values = data_dict[mask_name]
    
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
    result_dir = 'mask_each'
    masked_dir = 'masked_each'
    os.makedirs(result_dir, exist_ok=True)

    # 라벨링된 각 영역에 대하여
    for i in range(1, num_features + 1):
        component = (labeled == i).astype('uint8') * 255  # 원본 마스크 이미지와 동일한 크기의 마스크 이미지 생성

        if np.sum(component) < min_area:
            continue
        
        mini_dict = {}
        
        color_mask = cv2.cvtColor(component, cv2.COLOR_GRAY2BGR)
        masked_result = image - (255 - color_mask) * 0.3

        masked_result = np.concatenate((image, masked_result, color_mask), axis = 1)
        
        # 결과 마스크 파일 이름 생성
        base_name = os.path.basename(mask_path).split('.')[0]  # 파일 확장자 제거
        extension_name = os.path.basename(mask_path).split('.')[1]
        
        result_mask_path = os.path.join(result_dir, f"{base_name}_M{str(i).zfill(3)}.{extension_name}")
        result_masked_path = os.path.join(masked_dir, f"{base_name}_M{str(i).zfill(3)}.{extension_name}")
        
        # 결과 마스크 저장
        cv2.imwrite(result_mask_path, component)
        cv2.imwrite(result_masked_path, masked_result)
        
        if ii < 1000:
            cv2.imwrite(os.path.join(result_dir + '_1000', f"{base_name}_M{str(i).zfill(3)}.{extension_name}"), component)
            cv2.imwrite(os.path.join(masked_dir + '_1000', f"{base_name}_M{str(i).zfill(3)}.{extension_name}"), masked_result)
            cv2.imwrite(os.path.join('image' + '_1000', f"{base_name}.{extension_name}"), image)
        
        box = component == 255
        
        coords = np.column_stack(np.where(box))
        min_yx = coords.min(axis=0)
        max_yx = coords.max(axis=0)
        
        
        mini_dict[header_name[0]] = mask_name
        mini_dict[header_name[1]] = f"{base_name}_M{str(i).zfill(3)}.{extension_name}"
        
        mini_dict[header_name[2]] = min_yx[1]
        mini_dict[header_name[3]] = min_yx[0]
        mini_dict[header_name[4]] = max_yx[1]
        mini_dict[header_name[5]] = max_yx[0]
        
        mini_dict[header_name[6]] = rest_of_values[0]
        mini_dict[header_name[7]] = rest_of_values[1]
        mini_dict[header_name[8]] = rest_of_values[2]
        mini_dict[header_name[9]] = rest_of_values[3]
        mini_dict[header_name[10]] = rest_of_values[4]
        mini_dict[header_name[11]] = rest_of_values[5]
        mini_dict[header_name[12]] = rest_of_values[6]
        mini_dict[header_name[13]] = rest_of_values[7]
        mini_dict[header_name[14]] = rest_of_values[8]
        mini_dict[header_name[15]] = rest_of_values[9]
        mini_dict[header_name[16]] = rest_of_values[10]

        new_data_info.append(mini_dict)
        
        if ii < 1000:
            filename = 'burn_box_info_1000.csv'

            # CSV 파일로 저장
            with open(filename, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=header_name)
                writer.writeheader()  # 헤더 쓰기
                writer.writerows(new_data_info)  # 딕셔너리들을 한번에 쓰기
        
        
    ii += 1

    

header_name = ['name', 'mask name', 'min x', 'min y', 'max x', 'max y', 'location', 'team', 'info', 'date', 'gender', 'age', 'parts', 'source', 'days', 'YN', 'Num']

import csv

filename = 'burn_box_info.csv'

# CSV 파일로 저장
with open(filename, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header_name)
    writer.writeheader()  # 헤더 쓰기
    writer.writerows(new_data_info)  # 딕셔너리들을 한번에 쓰기













