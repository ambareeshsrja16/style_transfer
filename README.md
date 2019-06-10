Description
===========
Real-Time Style Transfer : Improvement on the Gatys approach, which produces high-quality images, but is slow since inference requires solving an optimization problem. Justin Johnson et al. came up with a variant of style transfer that was much faster and produced similar results to the original implementation. Their approach involves training a CNN in a supervised manner, using perceptual loss function to measure the difference between output and ground-truth images.


https://arxiv.org/abs/1603.08155

Requirements
============
Install the following packages using pip/conda:

1. cv2
2. numpy
3. matplotlib
4. torch

Code organization
=================

get_Image_Transform_Network.py
get_VGG_network.py
get_data_loaders.py
main.py
trainer.py
utils.py

main.ipynb
test.ipynb

Instructions for the Repo:
=================

1. main.ipynb is the notebook for training and test demos. 
2. Delete and make these folders before starting the notebook - debug, artifacts
3. "artifacts" folder will have a .txt file from which you can plot the loss curves, and debug has intermediate stylizations.
4. "artifacts" will also have saved models which can be loaded (the code for this is present in one of the last few cells) for testing purposes


Jupyter notebook file for demonstration:
=======================================

test.ipynb is the notebook for a quick demo of the best model weights (trained on COCOCaptions, style image Starry Night)
A python pickle file containing the saved model is part of the git, and this will be loaded, and can be tested for any specific content image
