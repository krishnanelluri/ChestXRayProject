### Team30_ChestXray

To train several single-head models from raw images, follow the steps below:

* Step 0: Skip this step if already done. Otherwise please unzip a subset of X-ray images from dataset.tgz and unzip models.tgz for pre-trained models. 
Make sure `root_path` in prepare_images_pyspark.py points to the folder 
where CheXpert images are stored is e.g. `root_path = "../../dataset_small/"`. The full set of X-ray 
images can be downloaded from http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip. 

* Run these files with python 2.7

* Step 1: Run `python prepare_images_pyspark.py`. This code selects all images and filters the data for the specified `pathology`,`view` and crop the images to 320x320 and calculates the mean and standard deviation for image standardization purposes using dributed processing with pyspark and saves the cropped images to an h5 file. Since mean and standard deviation calculation are memory-intensive operations, it downsamples the number of images. The output file will be created under the same folder as `root_path`.

* Step 2: Run `python hyperparamater_tuning_py`. This code is used to train and tune a single-head model after standardizing images.

* Step 3: Change the pathology name and run step1 and step2 for each pathology.



