import h5py
import numpy as np
from keras.layers import Input
from keras.models import load_model
from sklearn.metrics import roc_auc_score


def import_models(pathdict):
    models = []
    for key, path2modelx in pathdict.items():
        modelTemp = load_model(path2modelx)  # load model
        modelTemp.name = "Model_{}".format(key)  # change name to be unique
        models.append(modelTemp)

    return models


def load_validation_data(root_path, branch_names):
    hf = h5py.File(root_path + 'cropped_frontal.h5', 'r')

    valid_x = np.array(hf['valid']['images']).astype('float32') / 255.
    valid_y_branches = {}

    for name in branch_names:
        # encode class values as integers
        valid_y = np.array(hf['valid'][name])
        valid_y_branches[name] = valid_y

        # display classes for each pathology
        print("----[Validation] %s ----" % name)
        unique, counts = np.unique(valid_y, return_counts=True)
        print(np.asarray((unique, counts)).T)

    hf.close()

    return valid_x, valid_y_branches


multi_class_model_paths = {
    "run1": "../../models/model_val_auc_0.6693.h5",  # the best run among the 3 runs
}

root_path = "../../dataset_small/"
# select your heads
branch_names = [
    "No Finding",  # 0
    "Enlarged Cardiomediastinum",  # 1
    "Cardiomegaly",  # 2
    "Lung Opacity",  # 3
    "Lung Lesion",  # 4
    "Edema",  # 5
    "Consolidation",  # 6
    "Pneumonia",  # 7
    "Atelectasis",  # 8
    "Pneumothorax",  # 9
    "Pleural Effusion",  # 10
    "Pleural Other",  # 11
    "Fracture",  # 12
    "Support Devices",  # 13
]

for i in range(len(branch_names)):
    branch_names[i] = branch_names[i].replace(" ", "_")

valid_x, valid_y_branches = load_validation_data(root_path, branch_names)

model_input = Input(shape=(260, 300, 1))

# ensemble multi-class model from different runs
multi_class_models = import_models(multi_class_model_paths)

multi_class_prob_valid = []
for model in multi_class_models:
    multi_class_prob_valid.append(model.predict(valid_x))  # 1 x 14 x 202 x 1

multi_class_prob_valid = np.mean(np.array(multi_class_prob_valid), axis=(0,3))  # 14 x 202

five_pathologies = [
    "Atelectasis",
    "Consolidation",
    "Pleural_Effusion",
    "Cardiomegaly",
    "Edema"
]
# keep only 5 pathologies
multi_class_prob_valid = multi_class_prob_valid[[8, 6, 10, 2, 5], :]  # 5 x 202

printline = "- Validation AUC "
for i, k in enumerate(five_pathologies):
    mc_auc = roc_auc_score(valid_y_branches[k], multi_class_prob_valid[i])
    printline += "- {:s}_mc_auc: {:.4f} ".format(k, mc_auc)

print(printline)
