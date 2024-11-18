import timm
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

# 이미지 경로 및 모델 초기화
IMAGE_PATH = 'C:\\Users\\drago\\PycharmProjects\\ai_medical\\Dataset\\train\\malignant\\ISIC_0082829.jpg'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

# 모델 불러오기
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)
model.to(device)

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # EfficientNet 모델에 맞는 입력 크기로 조정
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet의 평균과 표준편차 사용
])

# 이미지 로드 및 전처리
img = Image.open(IMAGE_PATH).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)  # 배치 차원 추가 후 GPU로 이동

# 레이블 생성 (이 예시에서는 'malignant'로 1, 'benign'으로 0)
label = torch.tensor([1]).to(device)

# GradCAM을 위한 타겟 레이어 설정
target_layers = [model.conv_head]

# GradCAM 생성
cam = GradCAM(model=model, target_layers=target_layers)
targets = [BinaryClassifierOutputTarget(label)]  # 바이너리 분류의 타겟 설정
grayscale_cam = cam(input_tensor=img_tensor, targets=targets)

# 이미지를 float32로 변환하고 [0, 1] 범위로 정규화
img = np.array(img).astype(np.float32) / 255.0

# heatmap 크기를 원본 이미지 크기(224x224)로 맞추기
heatmap_resized = cv2.resize(grayscale_cam[0, :], (img.shape[1], img.shape[0]))

# GradCAM 결과 시각화
visualization = show_cam_on_image(img, heatmap_resized, use_rgb=True)

# 결과 출력
plt.imshow(visualization)
plt.axis('off')
plt.show()
