### Team30_ChestXray

#### Build multiple single-head models

To train several single-head models from raw images, follow the steps below:

* Step 0: Skip this step if already done. Otherwise please unzip a subset of X-ray images from dataset.tgz and unzip models.tgz for pre-trained models. 
Make sure `root_path` in 01_crop_images_byCRR.py points to the folder 
where CheXpert images are stored is e.g. `root_path = "../../dataset_small/"`. The full set of X-ray 
images can be downloaded from http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip. 

* Step 1: run `python 01_crop_images_byCRR.py` to create one h5 file. File `cropped_frontal.h5` will 
be created under the same folder as `root_path`. 

* Step 2: run `python 02_hyperparameter_tuning_hyperas.py` to optimized hyper-parameters on 5000 images for single-head model for 
each pathology. Users shall uncomment the pathology in the branch_name accordingly. The optimal hyper-parameters 
including *Dense layer unit*, *learning rate* and *batch size*, will be printed at the end. 
```angular2html
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
        # "Pleural Effusion",
        # "Pleural Other",
        # "Fracture",
        # "Support Devices",
    ]
```

* Step 3: Based on the optimal hyper-parameters from step 3, update the corresponding hyper-parameters in 
```angular2html
03_train_Atelectasis.py, 
03_train_Cardiomegaly.py, 
03_train_Consolidation.py, 
03_train_Edema.py,
03_train_Pleural.py
```
Run those python scripts to train single-head model for each pathology. 

* Step 4: run `python 04_predict.py` to predicate the probability of each pathology using the combined single-head models. 
The results will be printed at the end. **Note** the pre-trained models are used here. 



