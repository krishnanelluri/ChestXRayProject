#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras
import h5py
import numpy as np

from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.models import Model
from keras.layers import (Dense, Input)
from keras.callbacks import Callback
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

pathology = "Edema"
root_path = "../../dataset_small/"

print("Reading data...")
hf = h5py.File(root_path+'{}.h5'.format(pathology.lower()), 'r')

train_x = np.array(hf['train']['images'])
train_y = np.array(hf['train']['labels'])

valid_x = np.array(hf['valid']['images'])
valid_y = np.array(hf['valid']['labels'])

hf.close()

unique, counts = np.unique(train_y, return_counts=True)
print(np.asarray((unique, counts)).T)

print("One hot encoding...")
encoder = LabelEncoder()
encoder.fit(train_y)

train_y = encoder.transform(train_y)
valid_y = encoder.transform(valid_y)

# one hot encoding
train_y = np_utils.to_categorical(train_y)
valid_y = np_utils.to_categorical(valid_y)

# custom metrics object to print out area under ROC for positive class
class Metrics(Callback):

    def __init__(self, write_model):
        super(Metrics,self).__init__()
        self.write_model = write_model

    def on_train_begin(self, logs={}):
        self.train_aucs = []
        self.valid_aucs = []

    def on_epoch_end(self, epoch, logs={}):

        _train_auc = roc_auc_score(train_y, self.model.predict(train_x))
        self.train_aucs.append(_train_auc)
        print("Train AUC: {:.4f}".format(_train_auc))

        _valid_auc = roc_auc_score(valid_y, self.model.predict(valid_x))

        if self.write_model:
            if len(self.valid_aucs) > 0:
                if _valid_auc > max(self.valid_aucs):
                    print("\nValid AUC {:0.4f} > {:0.4f}, so saving model".format(_valid_auc, max(self.valid_aucs)))
                    self.model.save("../resources/models/classification-best-model.h5")
            else:
                 self.model.save("../resources/models/classification-best-model.h5")

        self.valid_aucs.append(_valid_auc)
        print("Valid AUC: {:0.4f}".format(_valid_auc))
        return

print("Training model...")
input_tensor = Input(shape=(320, 320, 1))

base_model = DenseNet121(input_tensor=input_tensor, weights=None, include_top=False, pooling='avg')

x = base_model.output
x = Dense(128, activation='relu')(x)
output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_x, y=train_y, batch_size=16, epochs=1,
          validation_data=[valid_x, valid_y], shuffle=True, callbacks=[Metrics(write_model=False)]
          ,class_weight = {0: 0.25, 1: 0.75})

