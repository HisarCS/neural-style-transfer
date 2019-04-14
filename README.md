# neural-style-transfer

Artistic Neural Style Slideshow with Pytorch (ANSS)

ANSS was designed as an implementation of the Pytorch implementation of MSG-Net and Neural Style. It sets up a realtime presentation where you can cycle between different style images and use the camera to receive input, which will be customized on the background.


**Table of Contents**
* Stylize Single Images with Pre-Trained Model
* Setup presentation-mode
* Train a new model
* [How It Works](#how-it-works)


## **How It Works:**
ANSS uses Hang Zhang and Kristin Dana's implementation of MSG-Net, and its pretrained model. There are several files to analyze.
* main.py

This contains the training and evaluating functions of the neural net. I removed unused code for the project but the majority of this belongs to Zhang.

* net.py

This uses VGG-16, a deep CNN. This is where the magic happens.

* option.py

These contain the command line arguments.
TODO: Add more options for running run.py directly

* run.py

This file is essential for running the application. It consists of three separate processes. (Camera, slideshow and style-transfer processes, all simultaneously running in order to keep cv2 responsive and not halt camera feed.)

* slideshow_final.py

This is the slideshow of images in /style-images.

* utils.py

Contains helper functions.
