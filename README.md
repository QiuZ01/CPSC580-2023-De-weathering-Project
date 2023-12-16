# CPSC580-2023-De-weathering-Project
This project uses computer vision to identify and reduce weather-related impacts in images, focusing on rain and haze, enhancing image clarity and aiding in self-driving car safety.

## Instruction for Running the Experiments
Before running the experiments, please first download the related datasets from:
1. [GT-RAIN](https://drive.google.com/drive/folders/1NSRl954QPcGIgoyJa_VjQwh_gEaHWPb8)
2. [A2I2-Haze](https://a2i2-archangel.vision/haze)  (Registration required)
3. [RESIDE-IN](https://drive.google.com/drive/folders/1ggEslNXkWT1GukuQZn0B-cKsBBOz5cfc)
4. [RESIDE-OUT](https://drive.google.com/drive/folders/1rUnTeACiM0rztxV6BViDAV7l2zdC-q1L)

All code for running the main experiments are located within the ./src folder. Before running the code, please upload the code and datasets to Google colab, following the below directory structure:

To generate the results for Gaussian / Sharpening filters:
```
1. Go to Google Colab
2. Please make sure that the test data of GT-Rain (GT-Rain_test), which includes seven subfolders, respectively named
   "Gurutto_0-0", "M1135_0-0", "Oinari_0-0", "Oinari_1-1", "Table_Rock_0-0", "Winter_Garden_0-1", "Winter_Garden_0-4"
   is located at the path "/content/drive/MyDrive/Colab Notebooks/CPSC 480-580: Computer Vision/Final/src/data/GT-RAIN_test/"
3. Then, copy the first two images in the Winter_Garden_0-4 subfolder, which are
   "Winter_Garden_0-1-Webcam-C-000.png" and "src/data/Winter_Garden_0-1-Webcam-R-000.png",
   into "/content/drive/MyDrive/Colab Notebooks/CPSC 480-580: Computer Vision/Final/src/data"
4. Please run the notebook ./src/traditional_algorithm/traditional_algorithm.ipynb
```

To test the results for GT-rain:
```
please run the notebook ./src/dl_model/GT-RAIN/testHaze.ipynb. 
And make sure that the 'load_checkpoint' and data path is correct.
Put the model weight in 'load_checkpoint'
Put the data to be tested in 'input_path', and the ground truth in 'gt_path'
```

To test the results for MixDehazeNet:
```
please run the notebook ./src/dl_model/Dehaze/testRESIDE.ipynb. 
And make sure that the 'save_dir' and 'data_dir' path is correct.
Put the model weight in 'save_dir' + 'dataset';
Put the data in 'data_dir' + 'dataset', and the test subfiles are 'test/GT' and 'test/hazy'.
(the data to be tested in 'test/GT', and the ground truth in 'gt_path')
```

To test the results for Classifier:
```
please run the notebook ./src/dl_model/classifier/predict.py.
And make sure that the 'test_image_path' is correct.
Put the model weight './best.pth' in the same level directory.
```

Additionally, the code for generating the plots is located in ./EXPresult/classify/Plotting/classifier_plot.ipynb, which can be run in Google Colab regardless of directory structure.
Other results can also be found within the ./EXPresult folder.

If you encounter any problems when running the code, please do not hesitate to contact us. Thank you!

## (Below was the Project Proposal)
## 1  Project Title

**Multi-Weather Classification and Effect Removal on Images**


## 2  Names

- Yining Wang [yining.wang@yale.edu](yining.wang@yale.edu) 
- Andrew Yi [andrew.yi@yale.edu](andrew.yi@yale.edu)
- Qiuhao Zhang [qiuhao.zhang@yale.edu](qiuhao.zhang@yale.edu)


## 3  Problem Statement

In daily life, we could encounter many different types of weather conditions. However, some weathers could have unwanted effects such as increased noise level and reduced visibility on the images captured. On a more daily perspective, one would wish the weather to be sunny when taking pictures of people and scenery on a trip – but that’s not always the case. From a more technical aspect, self-driving cars’ detection ability might be hindered in rainy or hazy weather, which could lead to safety concerns. Therefore, in this project, we aim to address the problem of unwanted weather effects on images using computer vision techniques. We plan to first perform classification to determine the weather condition presented in the image, then perform de-weathering according to the weather type detected. We will probably focus on rainy and hazy weather for now.


## 4  General Approach

**The general approach to this project can consist of the following steps:**
1. Collect datasets and preprocess data
2. Develop a machine learning classification model to distinguish different weather conditions
3. Evaluate the classification model
4. Develop a traditional denoising / de-weathering algorithm to serve as a performance baseline
5. Develop a machine learning denoising / de-weathering model
6. Evaluate and compare the denoising / de-weathering algorithm and models
7. Draw Conclusions

**The challenges we may face:**
1. Exploring machine learning / deep learning based approaches
2. Unbalanced datasets for the classification model


## 5  Data

We tentatively plan to leverage two datasets for classifying weather conditions and denoising images. The first dataset is the [GT-RAIN](https://visual.ee.ucla.edu/gt_rain.htm/) dataset, which provides pairs of real rainy images and corresponding ground truth images captured shortly after the rain had stopped. It covers a wide range of scenarios, including different types of rain conditions, geographic locations, degrees of illumination, and camera parameters. This large-scale dataset consists of 31,524 pairs of rain and clean frames taken from 101 videos, with 26,124 training pairs, 3,300 validation pairs, and 2,100 testing pairs. 

The second dataset is the [A2I2-Haze](https://arxiv.org/abs/2206.06427) Dataset, the first real-world dataset for paired haze and haze-free aerial and ground images. A2I2-Haze consists of two subsets, A2I2-UAV and A2I2-UGV. The training set of A2I2-UAV has 224 pairs of hazy and haze-free images, along with an extra 240 haze-free images. The corresponding test set has 119 hazy images. The training set of A2I2-UGV includes 50 pairs of hazy and haze-free images, and an additional 200 haze-free images. The corresponding test set of A2I2-UGV has 200 hazy images.

## 6  Measure of Success

For the classification part, the level of success can be measured using traditional metrics such as accuracy, precision, recall, and F1-score. For the de-weathering part, the level of success can be measured using suitable metrics to evaluate the quality of the algorithms’ output images by comparing them with ground truth images (i.e. images of the same scene but in clear weather). Some possible evaluation metrics for the denoised / de-weathered images include the Peak Signal-to-Noise Ratio (PSNR) and the Structural Similarity Index (SSIM). (A higher PSNR value indicates a better quality of the denoised image. A high SSIM value suggests that the denoised image is similar to the ground truth image.)


## 7  Milestones and Deliverables

### Tentative Milestones

1. **Data Collection:** Collect and organize the datasets.
2. **Data Preprocessing:** Filter and process raw image data into the desired input format.
3. **Development of Weather Classification Algorithm:** Utilize machine learning approaches for weather classification.
4. **Development of De-weathering Algorithms for Different Weather Types** (focusing on rainy and hazy weathers):
   - Develop the algorithm using traditional computer vision approaches.
   - Develop the algorithm using machine learning approaches.
5. **Compare the Performance of Different Approaches:** Analyze and compare the effectiveness of each method.
6. **Integration:** Integrate the algorithms into a pipeline.
7. **Analysis and Finalization:** Analyze the result and finalize the project report.

### Possible Deliverables

1. **Weather Classification Algorithm / Model**
2. **De-weathering Algorithms / Model**
3. **Pipeline that integrate the above two steps**
4. **Final Project Report**
