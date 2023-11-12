# CPSC580-2023-De-weathering-Project
This project uses computer vision to identify and reduce weather-related impacts in images, focusing on rain and haze, enhancing image clarity and aiding in self-driving car safety.

## 1  Project Title

Multi-Weather Classification and Effect Removal on Images


## 2  Names

Yining Wang [yining.wang@yale.edu](yining.wang@yale.edu) 

Andrew Yi [andrew.yi@yale.edu](andrew.yi@yale.edu)

Qiuhao Zhang [qiuhao.zhang@yale.edu](qiuhao.zhang@yale.edu)


## 3  Problem Statement

In daily life, we could encounter many different types of weather conditions. However, some weathers could have unwanted effects such as increased noise level and reduced visibility on the images captured. On a more daily perspective, one would wish the weather to be sunny when taking pictures of people and scenery on a trip – but that’s not always the case. From a more technical aspect, self-driving cars’ detection ability might be hindered in rainy or hazy weather, which could lead to safety concerns. Therefore, in this project, we aim to address the problem of unwanted weather effects on images using computer vision techniques. We plan to first perform classification to determine the weather condition presented in the image, then perform de-weathering according to the weather type detected. We will probably focus on rainy and hazy weather for now.


## 4  General Approach

The general approach to this project can consist of the following steps:
Collect datasets and preprocess data
Develop a machine learning classification model to distinguish different weather conditions
Evaluate the classification model
Develop a traditional denoising / de-weathering algorithm to serve as a performance baseline
Develop a machine learning denoising / de-weathering model
Evaluate and compare the denoising / de-weathering algorithm and models
Draw Conclusions

The challenges we may face:
Exploring machine learning / deep learning based approaches
Unbalanced datasets for the classification model


## 5  Data

We tentatively plan to leverage two datasets for classifying weather conditions and denoising images. The first dataset is the GT-RAIN dataset, which provides pairs of real rainy images and corresponding ground truth images captured shortly after the rain had stopped. It covers a wide range of scenarios, including different types of rain conditions, geographic locations, degrees of illumination, and camera parameters. This large-scale dataset consists of 31,524 pairs of rain and clean frames taken from 101 videos, with 26,124 training pairs, 3,300 validation pairs, and 2,100 testing pairs. 

The second dataset is the A2I2-Haze Dataset, the first real-world dataset for paired haze and haze-free aerial and ground images. A2I2-Haze consists of two subsets, A2I2-UAV and A2I2-UGV. The training set of A2I2-UAV has 224 pairs of hazy and haze-free images, along with an extra 240 haze-free images. The corresponding test set has 119 hazy images. The training set of A2I2-UGV includes 50 pairs of hazy and haze-free images, and an additional 200 haze-free images. The corresponding test set of A2I2-UGV has 200 hazy images.

## 6  Measure of Success

For the classification part, the level of success can be measured using traditional metrics such as accuracy, precision, recall, and F1-score. For the de-weathering part, the level of success can be measured using suitable metrics to evaluate the quality of the algorithms’ output images by comparing them with ground truth images (i.e. images of the same scene but in clear weather). Some possible evaluation metrics for the denoised / de-weathered images include the Peak Signal-to-Noise Ratio (PSNR) and the Structural Similarity Index (SSIM). (A higher PSNR value indicates a better quality of the denoised image. A high SSIM value suggests that the denoised image is similar to the ground truth image.)


## 7  Milestones and Deliverables

**Tentative milestones:**
Data collection: collect and organize the datasets
Data preprocessing:  filter and process raw image data into the desired input format
Development of weather classification algorithm using machine learning approaches
Development of de-weathering algorithms for different weather types
(focusing on rainy and hazy weathers)
  a: Develop the algorithm using traditional computer vision approaches
  b: Develop the algorithm using machine learning approaches
Compare the performance of different approaches
Integrate the algorithms into a pipeline
Analyze the result and finalize the project report

**Possible deliverables:**
Weather classification algorithm / model
De-weathering algorithms / model
Pipeline that integrate the above two steps
Final project report
