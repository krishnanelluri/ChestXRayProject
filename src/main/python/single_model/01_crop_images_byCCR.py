#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import h5py
import numpy as np
import pandas as pd

root_path = "../../dataset_small/"

# 0 for negative, -1 for uncertain, and 1 for positive. unmentioned considered as negative
np.random.seed(123)
train_df = pd.read_csv(root_path + "CheXpert-v1.0-small/train.csv")
train_df = train_df.fillna(0)
train_df = train_df[train_df["Frontal/Lateral"] == "Frontal"]
train_df = train_df.sample(frac=1).reset_index(drop=True)

valid_df = pd.read_csv(root_path + "CheXpert-v1.0-small/valid.csv")
valid_df = valid_df.fillna(0)
valid_df = valid_df[valid_df["Frontal/Lateral"] == "Frontal"]
train_df.head(5)

# ### Read template
tmp = cv2.imread("./template.jpg", cv2.IMREAD_GRAYSCALE)
h, w = tmp.shape
print(h, w)

# ### Crop images by cross correlation
# - the height and width is identical to the size of the template
# - save cropped images + labels to h5 file (separated by train and valid)

pathology_names = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity","Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax","Pleural Effusion","Pleural Other", "Fracture", "Support Devices"]


def crop_images(df):
    images = []
    labels = []

    for idx, row in df.iterrows():
        if (idx % 10000 == 0):
            print(idx)

        # uncomment the following line to create smaller h5 with 5000 images
        # if (len(images) >= 5000):
        #     break

        img = cv2.imread(root_path + row.Path, cv2.IMREAD_GRAYSCALE)
        # Apply template Matching        
        res = cv2.matchTemplate(img, tmp, method=cv2.TM_CCOEFF)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # crop image
        # reshape to (height, width, 1) because 2D convolutional layers need shape (height, width, channels)        
        cropped = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].reshape(h, w, 1)

        label = []
        for p in pathology_names:
            label.append(int(row[p]))

        images.append(cropped)
        labels.append(label)

    return np.array(images), np.array(labels)


hf = h5py.File(root_path + 'cropped_frontal.h5', 'w')

train_images, train_labels = crop_images(train_df)
g1 = hf.create_group('train')
g1.create_dataset('images', data=train_images)
for i, name in enumerate(pathology_names):
    g1.create_dataset(name.replace(" ", "_"), data=train_labels[:, i])

valid_images, valid_labels = crop_images(valid_df)
g1 = hf.create_group('valid')
g1.create_dataset('images', data=valid_images)
for i, name in enumerate(pathology_names):
    g1.create_dataset(name.replace(" ", "_"), data=valid_labels[:, i])

hf.close()
