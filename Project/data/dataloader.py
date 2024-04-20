import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from config import opt
opt = opt()

class DataLoader:
    def __init__(self, data_dir = opt.data_path, batch_size = opt.batch_size_c, img_size = opt.img_size, shuffle = True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        
        self.dataset = ImageFolder(root=self.data_dir, transform=transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
        ]))
        
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=opt.n_cpu)
        
    def get_dataloader(self):
        return self.dataloader

    def get_data(self):
        return next(iter(self.dataloader))

    def get_img_size(self):
        return self.img_size

    def get_batch_size(self):
        return self.batch_size

    def get_data_dir(self):
        return self.data_dir
    
    def get_dataset(self):
        return self.dataset
    
    def get_data(self):
        return next(iter(self.dataloader))
    def __len__(self):
        return len(self.dataset)