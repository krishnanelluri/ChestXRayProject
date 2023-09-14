### Team30_ChestXray

#### Build one baseline multi-head model

To train a multi-head model from raw images, follow the steps below:

* Step 0: Skip this step if already done. Otherwise please unzip a subset of X-ray images from dataset.tgz and models.tgz file. Make sure `root_path` in 01_crop_images_byCRR.py points to the folder 
where CheXpert images are stored is e.g. `root_path = "../../dataset_small/"`. The full set of X-ray 
images can be downloaded from http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip. 

* Step 1: run `python 01_crop_images_byCRR.py` to create one h5 file. `cropped_frontal.h5` will 
be created under the same folder as `root_path`. 

* Step 2: run `python 02_train_baseline_model.py` to train the baseline multi-header model for all pathologies. Trained modeled 
from each epoch will be saved to the file `model_val_auc_{:0.4f}.h5`, where the averaged validation auc will be 
in the file name. The h5 model will be save in the same folder. 

* Step 3: run `python 03_predict.py` to predicate the probability of each pathology using the baseline multi-head model. 
The results will be printed at the end. **Note** the pre-trained model is used here. 


