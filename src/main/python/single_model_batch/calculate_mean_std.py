import h5py
import numpy as np

root_path = "/data/chexpert/"
pathology = "Cardiomegaly"
num_images = 900
sample_size = 900

sample_indices = np.random.choice(num_images, sample_size, replace=False)

hf = h5py.File('{}{}.h5'.format(root_path, pathology.lower()), 'r')

train_x = np.take(np.array(hf['train']['images']), sample_indices, axis=0).flatten()
del sample_indices

hf.close()

mean = np.mean(train_x)
std = (np.mean(abs(train_x - mean)**2))**(1/2)

np.save('{}{}_mean.npy'.format(root_path, pathology.lower()), mean)
np.save('{}{}_std.npy'.format(root_path, pathology.lower()), std)
