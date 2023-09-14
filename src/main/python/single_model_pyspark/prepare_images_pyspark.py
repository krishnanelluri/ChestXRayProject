#!/usr/bin/env python
# coding: utf-8

from pyspark import SparkContext
from pyspark.sql.types import StringType
from pyspark import SQLContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import *

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py

root_path = "../../dataset_small/"
pathology = "Edema"

spark = SparkSession.builder
spark = spark.config("spark.master", "local[*]")
spark = spark.config("spark.driver.memory", "30G")
spark = spark.config("spark.executor.cores", 6)
spark = spark.config("spark.executor.memory", "20G")
spark = spark.config("spark.driver.maxResultSize", "30G")
spark = spark.appName("ChestXray").getOrCreate()

train_df=spark.read.csv(root_path+'CheXpert-v1.0-small/train.csv', mode="DROPMALFORMED",inferSchema=True, header = True)
train_df = train_df.fillna(0)
train_df = train_df[train_df["Frontal/Lateral"]=="Frontal"]
train_df = train_df[(train_df[pathology] == 0.0) | (train_df[pathology] == 1.0)]

valid_df=spark.read.csv(root_path+'CheXpert-v1.0-small/valid.csv', mode="DROPMALFORMED",inferSchema=True, header = True)
valid_df = valid_df[valid_df["Frontal/Lateral"]=="Frontal"]

#train_df = train_df.limit(20000)

def crop_images(img,height,width):
    height_margin = 0 
    width_margin = 0
    if height > 320:
        height_margin = int((height - 320) / 2)
    if width > 320:
        width_margin = int((width - 320) / 2)
    cropped = img[height_margin : height_margin+320, width_margin : width_margin+320].reshape(320,320,1)
    return cropped

def read_valid_images(x):
    img = cv2.imread(root_path + x.Path, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    cropped = None    
    label = None
    if height < 400 and width < 400:
        cropped = crop_images(img,height,width)
        label = [int(x[pathology])]
    return cropped, label

validimageRDD = valid_df.rdd.map(lambda l: read_valid_images(l))
valid_images = validimageRDD.map(lambda x: x[0]).collect()
valid_labels = validimageRDD.map(lambda x: x[1]).collect()

valid_images = [x for x in valid_images if x is not None]
valid_labels = [x for x in valid_labels if x is not None]

sc = spark.sparkContext
valid_images = np.array(valid_images)
rdd = sc.parallelize(valid_images)
mean = rdd.mean()
stdev = rdd.stdev()
valid_images = (valid_images - mean)/stdev

def read_train_images(x,mn,sd):
    img = cv2.imread(root_path + x.Path, cv2.IMREAD_GRAYSCALE)    
    height, width = img.shape
    cropped = None
    label = None
    if height < 400 and width < 400:        
        cropped = crop_images(img,height,width)
        cropped = (cropped - mn)/sd
        label = [int(x[pathology])]
    return cropped, label

train_df = train_df.repartition(5000)
trainimageRDD = train_df.rdd.map(lambda l: read_train_images(l,mean,stdev))
train_images = trainimageRDD.map(lambda x: x[0]).collect()
train_labels = trainimageRDD.map(lambda x: x[1]).collect()

train_images = [x for x in train_images if x is not None]
train_labels = [x for x in train_labels if x is not None]

hf = h5py.File(root_path + '{}.h5'.format(pathology.lower()), 'w')

g1 = hf.create_group('train')
g1.create_dataset('images',data=np.array(train_images))
g1.create_dataset('labels',data=np.array(train_labels))

g1 = hf.create_group('valid')
g1.create_dataset('images',data=np.array(valid_images))
g1.create_dataset('labels',data=np.array(valid_labels))

hf.close()

