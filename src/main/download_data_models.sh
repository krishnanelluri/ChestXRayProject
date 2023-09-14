#!/bin/bash

rm *.tgz
wget https://www.dropbox.com/s/773r68xl285j7e0/models.tgz
wget https://www.dropbox.com/s/tbn0vz9wtqlmkpw/dataset_small.tgz

tar -xvzf models.tgz
tar -xvzf dataset_small.tgz
rm *.tgz