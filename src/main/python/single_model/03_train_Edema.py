#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import numpy as np
from keras.applications.densenet import DenseNet121
from keras.callbacks import Callback
from keras.layers import (Dense, Input)
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

def load_data(root_path, branch_names, subset=0):
    hf = h5py.File(root_path + 'cropped_frontal.h5', 'r')

    num_images = hf['train']['images'].shape[0]

    if (subset > 0):
        train_idx = np.arange(subset).tolist()
    else:
        train_idx = np.arange(num_images).tolist()

    print("{0:d} out of {1:d} images are used.".format(len(train_idx), num_images))

    train_x = np.array(hf['train']['images'][train_idx]).astype('float32') / 255.
    valid_x = np.array(hf['valid']['images']).astype('float32') / 255.

    train_y_branches = {}
    valid_y_branches = {}
    class_weights_branches = {}

    for name in branch_names:
        # encode class values as integers

        train_y = np.array(hf['train'][name][train_idx]).astype('int')
        train_y[train_y == -1] = 1  # U-ones
        valid_y = np.array(hf['valid'][name])

        class_weights = compute_class_weight('balanced', np.unique(train_y), train_y)
        class_weights_dict = dict(zip(np.unique(train_y), class_weights))

        train_y_branches[name] = train_y
        valid_y_branches[name] = valid_y
        class_weights_branches[name] = class_weights_dict

        # display classes for each pathology
        print("\n----[Training] %s ----" % name)
        unique, counts = np.unique(train_y, return_counts=True)
        print(np.asarray((unique, counts)).T)
        print("----[Validation] %s ----" % name)
        unique, counts = np.unique(valid_y, return_counts=True)
        print(np.asarray((unique, counts)).T)
        print("----[Class weight (optional) ] %s -" % name)
        print(class_weights_dict)

    hf.close()

    return train_x, train_y_branches, valid_x, valid_y_branches, class_weights_branches


root_path = "../../dataset_small/"
# select your heads
branch_names = [
    # "No_Finding",
    # "Enlarged Cardiomediastinum",
    # "Cardiomegaly",
    # "Lung Opacity",
    # "Lung Lesion",
    "Edema",
    # "Consolidation",
    # "Pneumonia",
    # "Atelectasis",
    # "Pneumothorax",
    # "Pleural Effusion",
    # "Pleural Other",
    # "Fracture",
    # "Support Devices",
]

for i in range(len(branch_names)):
    branch_names[i] = branch_names[i].replace(" ", "_")

train_x, train_y_branches, valid_x, valid_y_branches, class_weights_branches = load_data(root_path, branch_names)


# ### Modeling: DenseNet121
# custom metrics object to print out area under ROC for positive class
class Metrics(Callback):
    def __init__(self, write_model):
        super().__init__()
        self.write_model = write_model

    def on_train_begin(self, logs={}):
        self.train_aucs = []
        self.valid_aucs = []

    def on_epoch_end(self, epoch, logs={}):
        prob_train = self.model.predict(train_x)  # list of probablity arrays
        if (len(self.model.output_names) == 1):  # single head
            prob_train = [np.array(prob_train)]

        _train_auc = 0
        printline = "- AUCLine "
        for i, train_pred in enumerate(prob_train):
            bname = self.model.output_names[i]

            # To avoid error: Only one class present in y_true. ROC AUC score is not defined in that case.
            if (len(np.unique(train_y_branches[bname])) > 1):
                _train_auc = roc_auc_score(train_y_branches[bname], train_pred)

            printline += "- {:s}_Pos_AUC: {:.4f} ".format(bname, _train_auc)

        prob_valid = self.model.predict(valid_x)
        if (len(self.model.output_names) == 1):  # single head
            prob_valid = [np.array(prob_valid)]

        _valid_auc = 0
        _valid_auc1 = 0

        for i, valid_pred in enumerate(prob_valid):
            bname = self.model.output_names[i]

            if (len(np.unique(valid_y_branches[bname])) > 1):
                _valid_auc1 = roc_auc_score(valid_y_branches[bname], valid_pred)
                _valid_auc += _valid_auc1

            printline += "- val_{:s}_Pos_AUC: {:.4f} ".format(bname, _valid_auc1)

        _valid_auc /= len(train_y_branches)
        print(printline)

        self.valid_aucs.append(_valid_auc)

        if self.write_model:
            self.model.save("./Edema_val_auc_{:0.4f}.h5".format(_valid_auc))

        return


class myDenseNet:
    @staticmethod
    def _build_one_branch(inputs, branch_name, fn_act, final_act):
        x = Dense(128, activation=fn_act)(inputs)
        output = Dense(1, activation=final_act, name=branch_name)(x)
        return output

    @staticmethod
    def build(inputs, list_branch_names, fn_act='relu', final_act='sigmoid', pooling='avg'):
        base_model = DenseNet121(input_tensor=inputs, weights=None, include_top=False, pooling=pooling)
        branches = []
        for name in list_branch_names:
            bn = myDenseNet._build_one_branch(base_model.output, name, fn_act, final_act)
            branches.append(bn)

        model = Model(inputs=inputs, outputs=branches, name="myDenseNet")

        return model

# compile the multi-output model
input_tensor = Input(shape=(260, 300, 1))
model = myDenseNet.build(input_tensor, branch_names)

# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
losses = {}
loss_weights = {}
for name in branch_names:
    losses[name] = "binary_crossentropy"
    loss_weights[name] = 1.0

opt = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=["accuracy"])
model.fit(x=train_x, y=train_y_branches, batch_size=16, epochs=15, verbose=2,
          validation_data=[valid_x, valid_y_branches],
          class_weight=class_weights_branches,
          shuffle=True, callbacks=[Metrics(write_model=True)])

