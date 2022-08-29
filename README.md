# Demo for running segmentation models on iMX8MPlus SoC
# Introduction
A repository demonstrating use of upsampling layer for segmentation models on NXP's iM8M Plus SoC using MNIST dataset. The repo contains a colab notebook for training
a segmentation model with upsampling layer and MNIST datset. In addition, the notebook also demonstrates post-training quantiation of the trained model into INT8 format
for the NPU. A python script which runs on 8MPlus is also provided to run the demo. The script takes a model (tflite) and samples of MNIST dataset (images) as input
and generates output onto the console. The script compares inference results from CPU (using XNNPack) and NPU (using VX delegate). Two sample models are also provided with the repo,
but the modles and dataset could be generated using the colab notebook.

# iMX8MPlus system configurations

NAME="NXP i.MX Release Distro"

VERSION="5.10-hardknott (hardknott)"

Kernel: Linux 5.10.72+g2e6a992bbb32

# Problem with default upsampling layer
The repo highlights the fact that using the default interpolation in upsampling layer i.e., 'nearest' results in a tflite model which runs on CPU/XNNPack, but the model does not run
on NPU/VX delegate. The model generates 'fixed' results when running on NPU. On the other hand, if 'bilinear' interpolation is used in the upsampling layer, the model runs wihtout any
issues both on NPU and CPU. Did not find any clarification about this in NXP's machine learning guide, but a lot of users struggle with segmentaiton based models on NPU and therefore,
this repo might be a help.

# UNET and other segmentation
As a final note, both UNET (with concat and upsampling) and sequential (without concat, as in the notebook) based segmentation models were tested and results were similar, i.e., 
'bilinear' interpolation works fine on NPU and 'nearest' interpolation fails on NPU.

# Files
mnist_segmentation_model_demo_notebook.ipynb  ->  colab notebook  

demo_script.py                                ->  python script for iMX8MPlus

mnist_samples_labels.pkl                      ->  sample images of MNIST

mnist_nearest_demo_ptq.tflite                 ->  sample model

mnist_bilinear_demo_ptq.tflite                ->  sample model

