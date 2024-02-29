# Object Detection Model for Crop and Weed Detection
 ## Introduction
 This project focuses on the development of an object detection model tailored specifically for 'Crop and Weed Detection' in sugarcane, grapes, and banana fields. The model utilizes YOLO (v8) to train a custom model on a custom dataset, enabling precise identification of crop stems and weeds with distinct separation.

 ## Installation
 To run this project, you need to install the following dependencies:
```bash
pip install ultralytics
pip install wandb
```
## Wandb Login
refer to following site for wandb login
```bash
https://www.kaggle.com/code/samuelcortinhas/weights-biases-tutorial-beginner
```

# Dataset Structure

This dataset contains images and corresponding labels for various purposes. The images are organized in the following structure:
```bash
dataset/
│
├── images/
│ ├── train/
│ │ ├── image_1.jpg
│ │ ├── image_2.jpg
│ │ └── ...
│ │
│ ├── val/
│ │ ├── image_1.jpg
│ │ ├── image_2.jpg
│ │ └── ...
│ │
│ └── ...
│
├── labels/
│ ├── train/
│ │ ├── image_1.txt
│ │ ├── image_2.txt
│ │ └── ...
│ │
│ ├── val/
│ │ ├── image_1.txt
│ │ ├── image_2.txt
│ │ └── ...
│ │
│ └── 
PathToDataset.yaml
   |_train:pathtotrainimages
   |_val:pathtoValimages
```



