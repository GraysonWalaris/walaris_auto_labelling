# Walaris Auto Labelling

This repository can be used with mmdetection to automatically label a dataset
using pretrained models included with the mmdetection library.

# Setup

## Ensure that you have CUDA and cuDNN downloaded

Check CUDA version with:

```nvcc --version```

If not installed, install CUDA from the official installation guide:
https://developer.nvidia.com/cuda-toolkit-archive

If you have not yet installed cuDNN, follow the official installation guide:
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

*** You need to make sure that your cuda installation version matches the version that pytorch is installed with. For example, I use CUDA 11.8 and download the version of pytorch compatible with CUDA 11.8. If you do not do this, you will not be able to compile mmdetection in the following steps. ***


## Download mmdetection

This repo is meant to be cloned into the mmdetection repository. As such, make sure you follow **Step 1: Case a** in the setup instructions shown below.

Follow the official mmdetection installation instruction found here:
https://mmdetection.readthedocs.io/en/latest/get_started.html

## Configure mmdetection

Create a checkpoint directory in the base mmdetection directory and clone this repository

```bash
cd /path/to/mmdetection
mkdir checkpoints
git clone https://github.com/GraysonWalaris/walaris_auto_labelling.git
```

## In your mmdetection environment, download and install the Walaris Data_Processing repository

Follow the installation instructions found here:
https://github.com/GraysonWalaris/Data_Processing

# Get started

To get started, download the checkpoints for the desired model and place them in the checkpoints directory organized as shown below:

![Alt text](https://github.com/GraysonWalaris/walaris_auto_labelling/blob/ref/checkpoint_dir_setup.png?raw=true)

See main.py for an example script.



