import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

from tensorflow.keras.applications.vgg16 import VGG16 as Model
from tensorflow.keras.applications.vgg16 import VGG16 as preprocess_input
from tensorflow.keras.preprocessing.image import load_img

model = load_model('vgg16_custom_trained.h5')
print(model.summary())

image_titles = ['Malignant', 'Benign']

mal = '..\\Dataset\\train\\malignant\\ISIC_0082829.jpg'
ben = '..\\Dataset\\train\\benign\\ISIC_0114227.jpg'

img1 = load_img('base_dir1', target_size =(224, 224))
img2 = load_img('base_dir2', target_size =(224, 224))

print(type(img1))

img1_array = cv2.cvtColor(np.array(np.array(img1), cv2.COLOR_RGB2BGR))
img2_array = cv2.cvtColor(np.array(np.array(img2), cv2.COLOR_RGB2BGR))

cv2.imshow("Original mal", img1_array)
cv2.imshow("Original ben", img2_array)


images = np.asarray([np.array(img1), np.array(img2)])
X = preprocess_input(images)

def loss(output):
    return(output[0][283], output[1][150])


def model_modifier(md1):
    md1.layers[-1].activation = tf.keras.activations.linear


from tf_keras_vis.utils import normalize
from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam

gradcam = Gradcam(model,
                  model_modifier=model_modifier,
                  clone=False)

cam = gradcam(loss, X, penultimate_layer=-1)

cam = normalize(cam)


heatmapImg1  = np.uint8(cm.jet(cam[0])[..., :3] * 255)
heatmapImg1 = cv2.applyColorMap(heatmapImg1, cv2.COLORMAP_JET)

alpha = 0.5

overlay = heatmapImg1.copy()
result1 = cv2.addWeighted(img1_array, alpha, heatmapImg1, 1-alpha, 0)

scale_percent = 200
w = int(heatmapImg1.shape[1]  * scale_percent / 100)
h = int(heatmapImg1.shape[0]  * scale_percent / 100)
dim = (w, h)

result1 = cv2.resize(result1, dim, interpolation=cv2.INTER_AREA)
img1_array =  cv2.resize(img1_array, dim, interpolation=cv2.INTER_AREA)

cv2.imshow("GradCam - mal", result1)
cv2.imshow('Original - mal', img1_array)
cv2.waitKey(0)



heatmapImg2  = np.uint8(cm.jet(cam[1])[..., :3] * 255)
heatmapImg2 = cv2.applyColorMap(heatmapImg2, cv2.COLORMAP_JET)

overlay = heatmapImg2.copy()
result2 = cv2.addWeighted(img2_array, alpha, heatmapImg2, 1-alpha, 0)


w = int(heatmapImg1.shape[1]  * scale_percent / 100)
h = int(heatmapImg1.shape[0]  * scale_percent / 100)
dim = (w, h)

result2 = cv2.resize(result2, dim, interpolation=cv2.INTER_AREA)
img2_array =  cv2.resize(img2_array, dim, interpolation=cv2.INTER_AREA)

cv2.imshow("GradCam - ben", result2)
cv2.imshow('Original - ben', img2_array)
cv2.waitKey(0)
