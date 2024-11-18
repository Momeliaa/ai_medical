import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 이미지 폴더 경로
img_path = 'C:/Users/drago/Desktop/의료 인공지능/train-image/image/'

csv_file = pd.read_csv('C:\\Users\\drago\\Desktop\\의료 인공지능\\train-metadata.csv')


# 악성(malignant)과 양성(benign) 이미지 목록 추출
ben = csv_file[csv_file['target'] == 0]
mal = csv_file[csv_file['target'] == 1]

# 400개 이미지를 무작위로 샘플링
ben = ben.sample(n=400, random_state=42)
mal = mal.sample(n=400, random_state=42)

# 특징 추출 함수
def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    # 색상 히스토그램 (RGB 채널)
    hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])  # blue channel
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])  # green channel
    hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])  # red channel

    # LBP 텍스처 특징 (그레이스케일로 변환 후)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59), range=(0, 58))

    # 색상 히스토그램과 LBP 특징을 결합
    features = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten(), lbp_hist.flatten()])
    return features

# 악성(malignant)과 양성(benign) 이미지에 대해 특징 추출
def extract_features_for_class(image_list):
    features_list = []
    for img_name in image_list:
        img_path_full = os.path.join(img_path, img_name + '.jpg')
        features = extract_features(img_path_full)
        if features is not None:
            features_list.append(features)
    return np.array(features_list)

# Benign과 Malignant 이미지에서 특징 추출
benign_features = extract_features_for_class(ben['isic_id'].tolist())
malignant_features = extract_features_for_class(mal['isic_id'].tolist())

# 평균과 표준편차 계산
benign_mean = np.mean(benign_features, axis=0)
benign_std = np.std(benign_features, axis=0)

malignant_mean = np.mean(malignant_features, axis=0)
malignant_std = np.std(malignant_features, axis=0)

# 특징의 평균 비교
plt.figure(figsize=(12, 6))
plt.plot(benign_mean, label='Benign Mean', color='blue', alpha=0.7)
plt.plot(malignant_mean, label='Malignant Mean', color='red', alpha=0.7)
plt.title('Mean Features of Benign and Malignant Images')
plt.xlabel('Feature Index')
plt.ylabel('Mean Value')
plt.legend()
plt.grid(True)
plt.show()

# 특징의 표준편차 비교
plt.figure(figsize=(12, 6))
plt.plot(benign_std, label='Benign Std Dev', color='blue', alpha=0.7)
plt.plot(malignant_std, label='Malignant Std Dev', color='red', alpha=0.7)
plt.title('Standard Deviation of Features for Benign and Malignant Images')
plt.xlabel('Feature Index')
plt.ylabel('Standard Deviation')
plt.legend()
plt.grid(True)
plt.show()