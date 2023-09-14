import cv2
import pandas as pd
import numpy as np
import h5py

root_path = "../../dataset_small/"
view = "Frontal"
pathology = "Cardiomegaly"
size = 320

# 0 for negative, -1 for uncertain, and 1 for positive
train_df = pd.read_csv(root_path + "CheXpert-v1.0-small/train.csv")
train_df = train_df[train_df["Frontal/Lateral"] == view]
train_df = train_df.fillna(0.0)
train_df = train_df[(train_df[pathology] == 0.0) | (train_df[pathology] == 1.0)]
train_df = train_df.sample(n=1000, random_state=123)

valid_df = pd.read_csv(root_path + "CheXpert-v1.0-small/valid.csv")
valid_df = valid_df[valid_df["Frontal/Lateral"] == view]

print("\n---- %s ----" % pathology)
print("\n---- Train ----")
print(train_df[pathology].value_counts())
print("\n---- Valid ----")
print(valid_df[pathology].value_counts())

# crop images to 320x320 from center of image
def crop_images(df):
    images = []
    labels = []

    for idx, row in df.iterrows():
        if(idx % 10000 == 0):
            print(idx)
        img = cv2.imread(root_path + row.Path, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape

        # exclude big images that won't crop well
        if height < 400 and width < 400:
            height_margin = 0
            width_margin = 0

            if height > size:
                height_margin = int((height - size) / 2)
            if width > size:
                width_margin = int((width - size) / 2)

            # reshape to (height, width, 1) because 2D convolutional layers need shape (height, width, channels)
            cropped = img[height_margin : height_margin+size, width_margin : width_margin+size].reshape(size,size,1)

            label = [int(row[pathology])]

            images.append(cropped)
            labels.append(label)

    return images, labels

# save cropped images + labels to h5 file (separated by train and valid)
hf = h5py.File('{}{}.h5'.format(root_path, pathology.lower()), 'w')

train_images, train_labels = crop_images(train_df)
g1 = hf.create_group('train')
g1.create_dataset('images',data=np.array(train_images))#, compression="gzip", compression_opts=9)
g1.create_dataset('labels',data=np.array(train_labels))#, compression="gzip", compression_opts=9)

valid_images, valid_labels = crop_images(valid_df)
g1 = hf.create_group('valid')
g1.create_dataset('images',data=np.array(valid_images))#, compression="gzip", compression_opts=9)
g1.create_dataset('labels',data=np.array(valid_labels))#, compression="gzip", compression_opts=9)

hf.close()
