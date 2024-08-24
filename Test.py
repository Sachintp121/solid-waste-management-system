import warnings
warnings.filterwarnings('ignore')
import visualkeras
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.graph_objects as go
from PIL import Image
import cv2
import os

import tensorflow as tf
from keras import layers
from keras.layers import Input, Lambda, Dense, Flatten, Dropout, BatchNormalization
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

directory_path = "archive/Garbage classification/Garbage classification/trash"
image_files = sorted([file for file in os.listdir(directory_path) if file.lower().endswith(('.jpg', '.jpeg', '.png'))])[:20]
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
for i, image_file in enumerate(image_files):
    img = Image.open(os.path.join(directory_path, image_file))
    ax = axes[i // 5, i % 5]
    ax.imshow(img)
    ax.axis('off')
plt.show()

directory_path = "archive/Garbage classification/Garbage classification/glass"
image_files = sorted([file for file in os.listdir(directory_path) if file.lower().endswith(('.jpg', '.jpeg', '.png'))])[:20]
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
for i, image_file in enumerate(image_files):
    img = Image.open(os.path.join(directory_path, image_file))
    ax = axes[i // 5, i % 5]
    ax.imshow(img)
    ax.axis('off')
plt.show()
data={}

import os
import shutil

original_path = 'archive/Garbage classification/Garbage classification/'
bonus_data_path = 'archive/garbage classification/garbage classification/'
destination_path = 'modified_dataset/'

if os.path.exists(destination_path) and os.path.isdir(destination_path):
    shutil.rmtree(destination_path)

os.makedirs(destination_path, exist_ok=True)
classes_to_remove = ['cardboard', 'trash', 'glass']  # in original data
new_classes = ['battery', 'biological', 'white-glass']  # in bonus data
target_image_count = 775

for class_name in os.listdir(original_path):
    class_path = os.path.join(original_path, class_name)

    if class_name in classes_to_remove:
        continue

    new_class_name = class_name
    if class_name in new_classes:
        new_class_name = new_classes[new_classes.index(class_name)]

    new_class_path = os.path.join(destination_path, new_class_name)
    os.makedirs(new_class_path, exist_ok=True)
    files_to_copy = os.listdir(class_path)[:target_image_count]
    for file_name in files_to_copy:
        file_path = os.path.join(class_path, file_name)
        shutil.copy(file_path, new_class_path)

    # If the class has fewer than target_image_count images, fill up from bonus data
    if len(os.listdir(new_class_path)) < target_image_count:
        remaining_images = target_image_count - len(os.listdir(new_class_path))
        print(
            f"{class_name} has {len(os.listdir(new_class_path))} images and missing {remaining_images} to fill {target_image_count}.")
        if class_name == "glass":
            bonus_class_path = os.path.join(bonus_data_path, "white-glass")
        else:
            bonus_class_path = os.path.join(bonus_data_path, class_name)
        if bonus_class_path:
            bonus_files = os.listdir(bonus_class_path)
            copied_names = set(os.listdir(new_class_path))
            for file_name in bonus_files:
                if remaining_images == 0:
                    break
                new_file_name = file_name
                counter = 1
                while new_file_name in copied_names:
                    base_name, extension = os.path.splitext(file_name)
                    new_file_name = f"{class_name}_{counter}{extension}"
                    counter += 1
                file_path = os.path.join(bonus_class_path, file_name)
                new_file_path = os.path.join(new_class_path, new_file_name)
                shutil.copy(file_path, new_file_path)
                copied_names.add(new_file_name)
                remaining_images -= 1

# Process the bonus dataset to copy battery and organic data
for class_name in os.listdir(bonus_data_path):
    class_path = os.path.join(bonus_data_path, class_name)
    if class_name in new_classes:
        new_class_name = class_name
        if class_name in new_classes:
            new_class_name = new_classes[new_classes.index(class_name)]
        if new_class_name == "biological":
            new_class_path = os.path.join(destination_path, "organic")
        elif new_class_name == "white-glass":
            new_class_path = os.path.join(destination_path, "glass")
        else:
            new_class_path = os.path.join(destination_path, new_class_name)
        os.makedirs(new_class_path, exist_ok=True)
        files_to_copy = os.listdir(class_path)[:target_image_count]
        for file_name in files_to_copy:
            file_path = os.path.join(class_path, file_name)
            shutil.copy(file_path, new_class_path)

print("\nFINISH: Dataset modification complete.")
for class_ in os.listdir(destination_path):
    count_class = len(os.listdir(os.path.join(destination_path, class_)))
    print(f"{class_} has {count_class} images.")
