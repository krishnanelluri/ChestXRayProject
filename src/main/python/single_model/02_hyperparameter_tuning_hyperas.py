import h5py
import numpy as np
from hyperas import optim
from hyperas.distributions import choice
from hyperopt import Trials, STATUS_OK, tpe
from keras.applications.densenet import DenseNet121
from keras.layers import (Dense, Input)
from keras.models import Model
from keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight


# select your heads
def data():
    root_path = "../../dataset_small/"
    branch_names = [
        # "No Finding",
        # "Enlarged Cardiomediastinum",
        # "Cardiomegaly",
        # "Lung Opacity",
        # "Lung Lesion",
        # "Edema",
        # "Consolidation",
        # "Pneumonia",
        # "Atelectasis",
        # "Pneumothorax",
        "Pleural Effusion",
        # "Pleural Other",
        # "Fracture",
        # "Support Devices",
    ]
    for i in range(len(branch_names)):
        branch_names[i] = branch_names[i].replace(" ", "_")

    hf = h5py.File(root_path + 'cropped_frontal.h5', 'r')

    train_x = np.array(hf['train']['images']).astype('float32') / 255.
    valid_x = np.array(hf['valid']['images']).astype('float32') / 255.

    train_y_branches = {}
    valid_y_branches = {}
    class_weights_branches = {}

    for name in branch_names:
        # encode class values as integers
        train_y = np.array(hf['train'][name]).astype('int')
        train_y[train_y == -1] = 1  # U-ones
        valid_y = np.array(hf['valid'][name])

        class_weights = compute_class_weight('balanced', np.unique(train_y), train_y)
        class_weights_dict = dict(zip(np.unique(train_y), class_weights))

        train_y_branches[name] = train_y
        valid_y_branches[name] = valid_y
        class_weights_branches[name] = class_weights_dict

    hf.close()

    return train_x, train_y_branches, valid_x, valid_y_branches, class_weights_branches


def model(train_x, train_y_branches, valid_x, valid_y_branches, class_weights_branches):
    # hard-wired 2-heads
    branch_names = [
        # "No Finding",
        # "Enlarged Cardiomediastinum",
        # "Cardiomegaly",
        # "Lung Opacity",
        # "Lung Lesion",
        # "Edema",
        # "Consolidation",
        # "Pneumonia",
        # "Atelectasis",
        # "Pneumothorax",
        "Pleural Effusion",
        # "Pleural Other",
        # "Fracture",
        # "Support Devices",
    ]
    for i in range(len(branch_names)):
        branch_names[i] = branch_names[i].replace(" ", "_")

    # compile the multi-output model
    input_tensor = Input(shape=(260, 300, 1))

    base_model = DenseNet121(input_tensor=input_tensor, weights=None, include_top=False, pooling='avg')
    branches = []
    for name in branch_names:
        x = Dense({{choice([128, 256, 512])}}, activation='relu')(base_model.output)
        bn = Dense(1, activation='sigmoid', name=name)(x)
        branches.append(bn)

    model = Model(inputs=input_tensor, outputs=branches, name="myDenseNet")

    # define two dictionaries: one that specifies the loss method for
    # each output of the network along with a second dictionary that
    # specifies the weight per loss
    losses = {}
    loss_weights = {}
    for name in branch_names:
        losses[name] = "binary_crossentropy"
        loss_weights[name] = 1.0

    opt = Adam(lr={{choice([1e-3, 1e-4, 1e-5])}}, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=["accuracy"])
    result = model.fit(x=train_x, y=train_y_branches, batch_size={{choice([8, 16, 32])}}, epochs=15, verbose=2,
                       validation_data=[valid_x, valid_y_branches],
                       class_weight=class_weights_branches,
                       shuffle=True)

    # get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


best_run, best_model = optim.minimize(model=model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=2,
                                      trials=Trials())

from sklearn.metrics import roc_auc_score

branch_names = [
    # "No Finding",
    # "Enlarged Cardiomediastinum",
    # "Cardiomegaly",
    # "Lung Opacity",
    # "Lung Lesion",
    # "Edema",
    # "Consolidation",
    # "Pneumonia",
    # "Atelectasis",
    # "Pneumothorax",
    "Pleural Effusion",
    # "Pleural Other",
    # "Fracture",
    # "Support Devices",
]

pathology = branch_names[0]
train_x, train_y_branches, valid_x, valid_y_branches, class_weights_branches = data()
print("Evalutation of best performing model:")
print(best_model.evaluate(valid_x, valid_y_branches))
prob_train = best_model.predict(train_x)
prob_valid = best_model.predict(valid_x)
_train_auc = roc_auc_score(next(iter(train_y_branches.values())), prob_train)
_valid_auc = roc_auc_score(next(iter(valid_y_branches.values())), prob_valid)
print("{:s}_AUC: {:.4f} - val_{:s}_AUC: {:.4f}".format(pathology, _train_auc, pathology, _valid_auc))

print("Best performing model chosen hyper-parameters:")
print(best_run)
