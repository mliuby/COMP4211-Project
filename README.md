## The repo is used to save the group project of COMP4211 in HKUST. Created by Minghao Liu and [Ruiming Min](https://raymonmin.github.io).


### Project structure
```
Project/
│
├── classifier/
│   ├── __init__.py
│   ├── classifier.py
|   ├── saliency.py
│   └── train_classifier.py
│
├── cyclegan/
│   ├── __init__.py
│   ├── cycle_gan_model.py
│   ├── cyclegan.py
│   ├── discriminators.py
│   ├── generators.py
│   ├── train_cyclegan.py
│   └── Unet.py
│
├── data/
│   ├── __init__.py
│   └── dataloader.py
│
├── dataset/
│
├── config.py
├── main.py
└── train.py
```

### Project Explanation

- The path to the project model is   `./Project`
- The classify model is in `./Project/classify`
- The CycleGAN model is in `./Project/cyclegan`
- The data is in `./Project/dataset`, all input should be saved as `*.jpg` or `*.png`
- The data loader and other related loading code are in `./Project/data`
- The file to train is `./Project/train.py`
- The file to inference is `./Project/main.py`
- All setting should in `./Project/config.py`
- Dataset:
Lung and Colon Cancer Histopathological Image Dataset
https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images?resource=download
The dataset contains color 25000 images with 5 classes of 5000 images each. All images are 768 × 768 pixels in size and are in jpeg. Colon image sets contain two classes: benign and adenocarcinomas; Lung image sets contain three classes: adenocarcinomas,
squamous, and benign. 

### Package used

| Name          | Version              | Build                   | Channel |
|---------------|----------------------|-------------------------|---------|
| pytorch       | 2.2.2                | py3.10_cuda11.8_cudnn8.7.0_0 | pytorch |
| torchvision   | 0.17.2               | py310_cu118             | pytorch |
| torchaudio    | 2.2.2                | py310_cu118             | pytorch |
| cuda-cudart   | 11.8.89              |                         | nvidia  |
