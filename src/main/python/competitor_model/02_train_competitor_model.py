#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import numpy as np
from keras.applications.densenet import DenseNet121
from keras import backend as K
import tensorflow as tf
with K.tf.device('/gpu:0'):
    config = tf.ConfigProto(intra_op_parallelism_threads=4,\
           inter_op_parallelism_threads=4, allow_soft_placement=True,\
           device_count = {'CPU' : 1, 'GPU' : 1})
    session = tf.Session(config=config)
    K.set_session(session)
from sklearn.preprocessing import binarize
from keras.models import Model
from keras.layers import (Dense, Input)
from keras.callbacks import Callback
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from keras.callbacks import CSVLogger
import pandas as pd
from keras.utils import np_utils


def load_data(root_path, branch_names, subset=0):
    
    hf = h5py.File(root_path + 'cropped_frontal.h5', 'r')
    
    num_images = hf['train']['images'].shape[0]

    if(subset>0):
        np.random.seed(123)
        train_idx = sorted(np.random.choice(num_images, subset, replace=False).tolist())
    else:
        train_idx = np.arange(num_images).tolist()

    train_x = np.array(hf['train']['images'][train_idx]).astype('float32') / 255.0
    valid_x = np.array(hf['valid']['images']).astype('float32') / 255.0


    train_y = hf['train']['labels'][train_idx]
    valid_y = np.array(hf['valid']['labels'])

    valid_y = np_utils.to_categorical(valid_y, num_classes=3)
    valid_y = valid_y.reshape((-1, 42))
    train_y=np_utils.to_categorical(train_y,num_classes=3)
    train_y = train_y.reshape((-1, 42))
    print("Shape for Train DataSet")
    print(train_y.shape)
    print("Shape for Validation DataSet")
    print(valid_y.shape)

    hf.close()
    return train_x, train_y,valid_x, valid_y



root_path = "../../dataset_small/"
# select your heads
branch_names = [
    "No_Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

for i in range(len(branch_names)):
    branch_names[i] = branch_names[i].replace(" ", "_")
    
train_x, train_y, valid_x, valid_y= load_data(root_path, branch_names, 5) #upto 50000


# ### Modeling: DenseNet121

# custom metrics object to print out area under ROC for positive class
class Metrics(Callback):
    def __init__(self, write_model): 
        super(Metrics, self).__init__()
        self.write_model = write_model
    
    def on_train_begin(self, logs={}):
        self.train_aucs = []
        self.valid_aucs = []

    def on_epoch_end(self, epoch, logs={}):
        prob_train = self.model.predict(train_x)#list of probablity arrays
        # prob_train = np.where(prob_train > 0.5, 1, 0)
        print("Shape of Predicted Values")
        print(prob_train.shape)
        val_auc_array = np.zeros((1, 14))
        train_auc_array = np.zeros((1, 14))
        t = 1
        for i in range(14):

            if (len(np.unique(train_y[:, t])) > 1 and len(np.unique(prob_train[:, t]) > 1)):
                _train_auc = roc_auc_score(train_y[:, t], prob_train[:, t])
                train_auc_array[0][i] = _train_auc

            t += 3

        df_train = pd.DataFrame(train_auc_array, columns=branch_names)
        with open('./model_metrics/train_auc.csv', 'a') as f:
            df_train.to_csv(f, mode='a', header=f.tell() == 0)

        
        prob_valid = self.model.predict(valid_x)

        k = 1
        for i in range(14):

            if (len(np.unique(valid_y[:, k])) > 1 and len(np.unique(prob_valid[:, k]) > 1)):
                val_man = np.where(prob_valid[:, k] > prob_valid[:, k + 1], prob_valid[:, k],
                                   prob_valid[:, k + 1]).reshape(prob_valid.shape[0], 1)
                _valid_auc = roc_auc_score(valid_y[:, k], val_man)
                val_auc_array[0][i] = _valid_auc
            k += 3
        df_valid = pd.DataFrame(val_auc_array, columns=branch_names)
        with open('./model_metrics/val_auc.csv', 'a') as f:
            df_valid.to_csv(f, mode='a', header=f.tell() == 0)


        if self.write_model:
            self.model.save("./model/all_pathologies_model_val_auc_{:0.4f}.h5".format(_valid_auc))

        return

# model training
input_tensor = Input(shape=(320, 320, 1))

base_model = DenseNet121(input_tensor=input_tensor, weights=None, include_top=False, pooling='avg')

x = base_model.output
x = Dense(128, activation='relu')(x)
output = Dense(42, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999), loss='binary_crossentropy', metrics=['accuracy'])
csv_logger = CSVLogger('./model_metrics/model_metrics.csv', append=True, separator=',')
# TODO: 10 epochs when using full dataset
history = model.fit(x=train_x, y=train_y, batch_size=16, epochs=1,
          validation_data=[valid_x, valid_y], shuffle=True, callbacks=[Metrics(write_model=True), csv_logger])






