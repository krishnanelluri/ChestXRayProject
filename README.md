# Pathology Detection in Chest X-Rays Using Deep CNN

Github repo: https://github.gatech.edu/cwhalen6/bd4hspring19project

Chexpert competition Leader board: https://stanfordmlgroup.github.io/competitions/chexpert/ see submissions by Team30. 

The notebook `src/main/jupyter/data_exploration.ipynb` is our initial exploration of the CheXpert dataset.

# Follow README file in each subfolder to build models

1. first of all, please run `./download_data_models.sh` in src/main folder to download a subset of data and pre-trained models. 
2. Enter each folder under src/main/python and follow the README file there to run each model. 

# Steps to run the code in Google Cloud

1. url: cloud.google.com
2. login Credential: bd4hspring19projectteam30/bd4hteam30
3. click on menu icon on top left and scroll all the way down
4. mouse over on AI Platform in Artificial Intelligence Section then click on Notebooks from the list
5. Select NVIDIA TESla P100 GPU - 1
6. select the bd4hprojectinstance and click start
7. upon instance successfully started OPEN JUPYTERLAB link will be enabled
8. click on OPEN JUPYTERLAB a new window will be opened with jupyter notebook.
9. in the folder tab bd4hspring19project folder will be available
10. Click on New Launcher '+' icon and select terminal

* Download Sample dataset
11. change the directory to /home/jupyter/bd4hspring19project/src/main
12. run download_data_models.sh

* To run the multi head model - base line model
13. change the directory to /home/jupyter/bd4hspring19project/src/main/python/baseline_model
14. run 01_crop_images_by_CCR.py
15. run 02_train_baseline_model.py
16. run 03_predict.py

* To run the competitor model
17. change the directory to /home/jupyter/bd4hspring19project/src/main/python/competitor_model
18. run 01_crop_images_by_CCR.py
19. run 02_train_competitor_model.py
20. run 03_predict.py

* TO run the single model - cropping images with template approach
21. change the directory to /home/jupyter/bd4hspring19project/src/main/python/single_model
22. run 01_crop_images_by_CCR.py
23. run 02_hyperparameter_tuning_hyperas.py
24. run 03_train_atelactasis.py
25. run 03_train_Cardiomegaly.py
26. run 03_train_Consolidation.py
27. run 03_train_Edema.py
28. run 03_train_Pleural.py
29. run 04_predict.py

* To run the single model - cropping images from center in batches
30. change the directory to /home/jupyter/bd4hspring19project/src/main/python/single_model_batch
31. run prepare_images_center_cropping
32. run calculate_mean_std.py
33. run hyperparameter_tuning.py

* To run the single model with pyspark
34. change the directory to /home/jupyter/bd4hspring19project/src/main/python/single_model_pyspark
35. run prepare_images_pyspark.py - upon succesfull execution h5 file will be created
36. run hyperparameter_tuning.py
