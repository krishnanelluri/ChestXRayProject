To run datapreprocessing code:

* Step 0: Skip this step if already done. Otherwise please unzip a subset of X-ray images from dataset.tgz and models.tgz file. Make sure `root_path` in `prepare_images_center_cropping.py`, `calculate_mean_std.py`, and `hyperparameter_tuning.py` points to the folder where CheXpert images are stored (e.g. `root_path = "../../dataset_small/"`). The full set of X-ray images can be downloaded from http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip.

* Step 1: Run `python prepare_images_center_cropping.py`. This code selects a random sample of images for the specified `pathology` and `view`, crops the images to 320x320, and saves the cropped images to an h5 file. The output file will be created under the same folder as `root_path`.

* Step 2: Run `python calculate_mean_std.py`. This code calculates the mean and standard deviation for image standardization purposes for the specified `pathology`. Since mean and standard deviation calculation are memory-intensive operations, it downsamples the number of images. The output files will be created under the same folder as `root_path`.

* Step 3: Run `python hyperparameter_tuning.py`. This code is used to train and tune a single-head model after standardizing images.

* Step 4: Change the `pathology` and run step 1, 2, and 3 for each pathology.
