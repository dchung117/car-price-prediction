#!/bin/bash

# Download anaconda installer
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-1-Linux-x86_64.sh

# Install anaconda
shasum -a 256 Anaconda3*.sh
bash Anaconda3*.sh

echo "Installed Anaconda; run 'source ~/.bashrc; conda env create -f environment.yml' to complete setup."
echo "Make sure to update git config --global user.name and user.email."

# Delete install scripts
rm Anaconda3*.sh

# configure cudnn, ld_library path
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh