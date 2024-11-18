import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# 이미지 경로 및 하이퍼파라미터 설정
base_dir = 'C:\\Users\\drago\\PycharmProjects\\ai_medical\\Dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
img_size = (224, 224)  # VGG16 모델의 입력 크기

# 전체 데이터셋에 대한 데이터 증강을 위한 ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# 데이터 로더 (train, validation, test)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='binary',
    subset='training',  # 'train' subset for training
)

val_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='binary'
)

# VGG16 모델 생성
def create_vgg16_model(input_shape=(224, 224, 3)):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # VGG16의 가중치는 고정하여 학습하지 않음

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 모델 생성
model = create_vgg16_model()

# EarlyStopping 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# 모델 학습
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    batch_size=32,
    callbacks=[early_stopping]
)

model.save('vgg16_custom_trained.h5')

# 모델 평가
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')
print(f'Test loss: {test_loss}')

# 예측 결과 생성 (classification_report)
y_true = test_generator.classes
y_pred = (model.predict(test_generator) > 0.5).astype('int32')

# F1-score, Precision, Recall 등 출력
print(classification_report(y_true, y_pred))

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, model.predict(test_generator))
pr_auc = auc(recall, precision)

# PR 곡선 시각화
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.show()

# 모델의 loss 그래프 시각화
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
