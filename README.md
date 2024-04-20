## The repo is used to save the group project of COMP4211 in HKUST. Created by Minghao Liu and Ruiming Min.

- The path to the project model is   `./Project`
- The classify model is in `./Project/classify`
- The CycleGAN model is in `./Project/cyclegan`
- The data is in `./Project/dataset`, all input should be saved as `*.jpg` or `*.png`
- The data loader and other related loading code are in `./Project/data`
- The main file to train is `./Project/main.py`
- All setting should in `./Project/config.py`
- Dataset:
(1) Lung and Colon Cancer Histopathological Image Dataset
https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images?resource=download
The dataset contains color 25000 images with 5 classes of 5000 images each. All images are 768 × 768 pixels in size and are in jpeg. Colon image sets contain two classes: benign and adenocarcinomas; Lung image sets contain three classes: adenocarcinomas,
squamous, and benign.
(2) Alzheimer_MRI Disease Classification Dataset
dataset = load_dataset("Falah/Alzheimer_MRI")
This dataset contains MRI scans of the brains of patients with Alzheimer’s disease. This dataset contains 5120 images with shapes of 128×128 and their corresponding labels representing severity. Specifically, the sizes of each class are: Mild Demented (724), Moderate Demented (49), Non-Demented(2566), Very Mild Demented (1781) .
