import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os

import config
opt = config.opt()
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, num_classes=None):
        super().__init__(root, transform=transform)
        self.num_classes = num_classes

    def __getitem__(self, index):
        img, label = super().__getitem__(index)

        one_hot = torch.zeros(self.num_classes)
        one_hot[label] = 1

        return img, one_hot
    
class DataLoader_Classifier:
    def __init__(self, data_dir = opt.data_path, data_fold = opt.data_fold_c, batch_size = opt.batch_size_c, img_size = opt.img_size, n_classes = opt.n_classes, shuffle = True):
        self.data_dir = data_dir
        self.data_fold = os.path.join(self.data_dir, data_fold)
        self.batch_size = batch_size
        self.img_size = img_size
        
        self.dataset = CustomImageFolder(
            root=self.data_fold,
            transform=transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ]),
            num_classes=n_classes
        )
        
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=opt.n_cpu)
        
    def get_dataloader(self):
        return self.dataloader
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_img_size(self):
        return self.img_size
    
    def get_data_fold(self):
        return self.data_fold

class DataLoader_CycleGAN:
    def __init__(self, data_dir = opt.data_path, data_fold_A = opt.data_fold_A, data_fold_B = opt.data_fold_B,  batch_size = opt.batch_size_g, img_size = opt.img_size, shuffle = True):
        self.data_dir = data_dir
        self.data_fold_A = os.path.join(self.data_dir, data_fold_A)
        self.data_fold_B = os.path.join(self.data_dir, data_fold_B)
        self.batch_size = batch_size
        self.img_size = img_size
        
        
        self.dataset_A = ImageFolder(root=self.data_fold_A, transform=transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
        ]))
        
        self.dataset_B = ImageFolder(root=self.data_fold_B, transform=transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
        ]))
        
        self.dataloader_A = DataLoader(self.dataset_A, batch_size=self.batch_size, shuffle=shuffle, num_workers=opt.n_cpu)
        self.dataloader_B = DataLoader(self.dataset_B, batch_size=self.batch_size, shuffle=shuffle, num_workers=opt.n_cpu)
        
    def get_dataloader(self):
        return zip(self.dataloader_A, self.dataloader_B)
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_img_size(self):
        return self.img_size
    
    def get_data_fold_A(self):
        return self.data_fold_A
    
    def get_data_fold_B(self):
        return self.data_fold_B