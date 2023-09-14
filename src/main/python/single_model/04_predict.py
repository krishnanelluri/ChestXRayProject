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

single_class_model_paths = {
    "Atelectasis": "../../models/Atelectasis_val_auc_0.8321.h5",
    "Consolidation": "../../models/Consolidation_val_auc_0.8814.h5",
    "Pleural_Effusion": "../../models/Pleural_val_auc_0.9245.h5",
    "Cardiomegaly": "../../models/Cardiomegaly_val_auc_0.8559.h5",
    "Edema": "../../models/Edema_val_auc_0.9094.h5"
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

# merge single-class model for each pathology
single_class_models = import_models(single_class_model_paths)
single_class_prob_valid = []
for model in single_class_models:
    single_class_prob_valid.append(model.predict(valid_x))  # 5 x 202 x 1

single_class_prob_valid = np.array(single_class_prob_valid)[:, :, 0]  # 5 x 202

printline = "- Validation AUC "
for i, k in enumerate(single_class_model_paths):
    sc_auc = roc_auc_score(valid_y_branches[k], single_class_prob_valid[i])
    printline += "- {:s}_sc_auc: {:.4f} ".format(k, sc_auc)

print(printline)
