import pandas as pd
import shutil
import os

img_path = 'C:\\Users\\drago\\Desktop\\의료 인공지능\\train-image\\image'

csv_file = pd.read_csv('C:\\Users\\drago\\Desktop\\의료 인공지능\\train-metadata.csv')

# ISIC 이미지 이름
img_name = csv_file['isic_id']

# 악성과 양성
ben_or_mal = csv_file['target']
ben = csv_file[ben_or_mal == 0]
mal = csv_file[ben_or_mal == 1]

print("Count of ben: ", len(ben))
print("Count of mal: ", len(mal))
print(img_name)

# 악성, 양성에 따라 ISIC 이미지를 복사해 폴더에 저장하는 함수
# tumor: ben, mal   /   dir_loc: Benign, Malignant   /   category: train, validation, test   /   복사할 이미지 dataset 개수
def copy_img(tumor, category, dir_loc, num):

    dataset_dir = os.path.join('Dataset')
    os.makedirs(dataset_dir, exist_ok=True)

    full_dir = os.path.join(dataset_dir, category, dir_loc)
    os.makedirs(full_dir, exist_ok=True)

    sampled_tumor = tumor.sample(n=num, random_state=42)
    img_names = sampled_tumor['isic_id'] + '.jpg'

    for ISIC_id in img_names:
        src_img_path = os.path.join(img_path, ISIC_id)
        dest_img_path = os.path.join(full_dir, ISIC_id)

        if not os.path.exists(dest_img_path):
            shutil.copy(src_img_path, dest_img_path)

    # 샘플링된 데이터 반환
    return sampled_tumor


train_ben = copy_img(ben, 'train', 'benign', 300)
train_mal = copy_img(mal, 'train', 'malignant', 300)

# validation dataset(train에서 일부를 분리)
val_ben = copy_img(train_ben, 'validation', 'benign', 92)
val_mal = copy_img(train_mal, 'validation', 'malignant', 92)

# test dataset(train dataset과 중복되지 않게 설정)
test_ben = copy_img(ben.drop(train_ben.index), 'test', 'benign', 300)
test_mal = copy_img(mal.drop(train_mal.index), 'test', 'malignant', 92)