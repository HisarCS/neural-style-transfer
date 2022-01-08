# neural-style-transfer

Artistic Neural Style Slideshow with Pytorch (ANSS)

ANSS was designed as an implementation of the Pytorch implementation of MSG-Net and Neural Style. It sets up a realtime presentation where you can cycle between different style images and use the camera to receive input, which will be customized on the background. I'm using a pretrained net for this project, this project uses the model to create an interactive Photo Booth where you can stylize your photos real time.

![alt text](https://analyticsindiamag.com/wp-content/uploads/2020/06/1_kOQOZxBDNw4lI757soTEyQ.png)
Image provided as an example of what Neural Style Transfer does to a photo.


**Table of Contents**
* [Stylize Single Images with Pre-Trained Model](#stylize-single-images-with-pre-trained-model)
* [Setup presentation-mode](#setup-presentation-mode)
* [Train a new model](#train-a-new-model)
* [How It Works](#how-it-works)
* [Acknowledgements](#acknowledgements)


## **Stylize Single Images With Pre Trained Model**
Clone the github repo. The pre-trained model is already downloaded for your use.
Navigate to your project folder, and do the following command:
```bash
	python3 main.py eval --content-image images/content/venice-boat.jpg --style-image images/21styles/mosaic.jpg --model models/21styles.model --content-size 1024
```
The output is saved as ```output.jpg``` on your main folder.

## **Setup Presentation Mode**
Navigate to your project folder, and type out the following command:
```bash
	python3 run.py
```
This will launch the processes necessary for the slideshow, along with periodically calling ```main.py``` for image stylizing.


## **Train A New Model**
In order to train your own model for stylizing, you have got to download a dataset first. Zhang provides one of COCO's datasets, and you can download it through ```bash datasets/download_dataset.sh```

```bash
	python3 main.py train --epochs 4
```
An epoch value of at least 2 is recommended. Training is computationally expensive, don't try with a CPU if you don't have a certain desire of waiting a couple weeks.

* If you would like to customize styles, set `--style-folder path/to/your/styles`. More options:
	* `--style-folder`: path to the folder style images.
	* `--vgg-model-dir`: path to folder where the vgg model will be downloaded.
	* `--save-model-dir`: path to folder where trained model will be saved.
	* `--cuda`: set it to 1 for running on GPU, 0 for CPU.

## **How It Works**
ANSS uses Hang Zhang and Kristin Dana's implementation of MSG-Net, and its pretrained model. It takes in the image from camera every x seconds, processes it in the background, and displays the stylized version alongside with the original artwork. There are three processes running with cv2.

Drawbacks: Since this was run on CPU (Mac Pro Xeon E3), I limited the output size to 600x600, and each image took around 5-6 seconds to process. Therefore, the frame captured by the camera every x seconds was processed for the next slide's style in order to preserve seamlessness.

There are several files to analyze.
* ```main.py```

This contains the training and evaluating functions of the neural net. I removed unused code for the project but the majority of this belongs to Zhang.

* ```net.py```

This uses VGG-16, a deep CNN. This is where the magic happens.

* ```option.py```

These contain the command line arguments.
TODO: Add more options for running run.py directly

* ```run.py```

This file is essential for running the application. It consists of three separate processes. (Camera, slideshow and style-transfer processes, all simultaneously running in order to keep cv2 responsive and not halt camera feed.)

* ```slideshow_final.py```

This is the slideshow of images in /style-images.

* ```utils.py```

Contains helper functions.

## **Acknowledgements**
The code benefits from outstanding prior work and their implementations including:
- [MSG-Net: Multi Style Generative Network](https://arxiv.org/pdf/1703.06953.pdf) by Hang Zhang and Kristin Dana ([code](https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer))
- [Texture Networks: Feed-forward Synthesis of Textures and Stylized Images](https://arxiv.org/pdf/1603.03417.pdf) by Ulyanov *et al. ICML 2016*. ([code](https://github.com/DmitryUlyanov/texture_nets))
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/pdf/1603.08155.pdf) by Johnson *et al. ECCV 2016* ([code](https://github.com/jcjohnson/fast-neural-style)) and its pytorch implementation [code](https://github.com/darkstar112358/fast-neural-style) by Abhishek.
- [Image Style Transfer Using Convolutional Neural Networks](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) by Gatys *et al. CVPR 2016* and its torch implementation [code](https://github.com/jcjohnson/neural-style) by Johnson.
