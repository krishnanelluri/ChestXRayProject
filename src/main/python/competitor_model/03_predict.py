import h5py
import numpy as np
from keras.layers import Input
from keras.models import load_model
from sklearn.metrics import roc_auc_score
from keras.utils import np_utils


def import_models(pathdict):
    models = []
    for key, path2modelx in pathdict.items():
        modelTemp = load_model(path2modelx)  # load model
        modelTemp.name = "Model_{}".format(key)  # change name to be unique
        models.append(modelTemp)

    return models


def load_validation_data(root_path):
    hf = h5py.File(root_path + 'cropped_frontal.h5', 'r')

    valid_x = np.array(hf['valid']['images']).astype('float32') / 255.0
    valid_y = np.array(hf['valid']['labels'])

    valid_y = np_utils.to_categorical(valid_y, num_classes=3)
    valid_y = valid_y.reshape((-1, 42))


    hf.close()

    return valid_x, valid_y


multi_class_model_paths = {
    "run1": "../../models/peev_all_pathologies_model_val_auc_0.89.h5",  # the best run among the 3 runs
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

valid_x, valid_y = load_validation_data(root_path)

model_input = Input(shape=(320, 300, 1))

# ensemble multi-class model from different runs
multi_class_models = import_models(multi_class_model_paths)
multi_class_prob_valid = []
for model in multi_class_models:
    multi_class_prob_valid.append(model.predict(valid_x))  # 3 x 14 x 202 x 1

# multi_class_prob_valid = np.mean(np.array(multi_class_prob_valid), axis=(0, 3))  # 14 x 202

# keep only 5 pathologies
# multi_class_prob_valid = multi_class_prob_valid[[8, 6, 10, 2, 5], :]  # 5 x 202
multi_class_prob_valid = np.array(multi_class_prob_valid)  # 5 x 202
multi_class_prob_valid = np.array(multi_class_prob_valid)[0,:, [7,16, 19, 25, 31]]
print("Probabilities For Validation Images")
indexes_prob =[7,16, 19, 25, 31]
pathologies =[2,5, 6, 8, 10]
printline =""
for i in range(multi_class_prob_valid.shape[1]):
    printline = "Image No" + str(i + 1) + " "
    for j in range(5):
        printline += branch_names[pathologies[j]] + " " + str(multi_class_prob_valid[j, i]) + " "
    print(printline)

