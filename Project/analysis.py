import torch
from classifier.classifier import Classifier
from cyclegan.cyclegan import cycleG
from data.dataloader import DataLoader_CycleGAN, DataLoader_Classifier
from config import opt
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

opt = opt()

name_type = opt.name_T
name_A = 'lung_aca'
name_B = 'lung_n'
name_C = 'lung_scc'

name_R = name_C

name_r = name_R.replace("_", "")

img_path = f'./dataset/lung/{name_R}'
save_path = f'./output/analysis/{name_type}'


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
    
    
def analysis(check_point, netC, idx_to_class,device, name_A, name_B, save_path= save_path):

    acc_A = 0
    acc_B = 0
    total_A = 0
    total_B = 0
    data_fold_A = os.path.join(opt.data_path, name_A)
    data_fold_B = os.path.join(opt.data_path, name_B)
    # Test the model
    
    dataloader = DataLoader_CycleGAN(data_fold_A=data_fold_A, data_fold_B=data_fold_B, img_size=64,shuffle=False)

    netG = cycleG()
    netG.load_state_dict(torch.load(os.path.join(g_save_path, f"generator/{opt.name_T}/{name_A}_and_{name_B}/netG_{check_point}.pth")))
    netG.eval()
    netG.to(device)
    netC.eval()

    for data in dataloader.get_dataloader():
        real_A, real_B = data

        real_A = real_A.to(device)
        real_B = real_B.to(device)

        fake_A, fake_B = netG(real_A, real_B)

        # resize the images to 32x32
        fake_A = F.interpolate(fake_A, size=(64, 64))
        fake_B = F.interpolate(fake_B, size=(64, 64))


        outputs_A = netC(fake_A)

        _, predicted_A = torch.max(outputs_A.data, 1)


        outputs_B = netC(fake_B)

        _, predicted_B = torch.max(outputs_B.data, 1)
        

        for label_A, label_B in zip(predicted_A, predicted_B):
            if idx_to_class[label_A.item()] == name_A:
                acc_A += 1
            if idx_to_class[label_B.item()] == name_B:
                acc_B += 1

        total_A += len(predicted_A)
        total_B += len(predicted_B)
        

    acc_A = acc_A / total_A
    acc_B = acc_B / total_B

    # save the result to output folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_file_path = os.path.join(save_path, f"result_{name_A}_and_{name_B}.txt")
    with open(save_file_path, 'w') as file:
        file.write(f"Accuracy of the network on the fake images from {name_B} to {name_A}: {100 * acc_A}%")
        
    with open(save_file_path, 'a') as file:
        file.write(f"Accuracy of the network on the fake images from {name_A} to {name_B}: {100 * acc_B }%")
        
    print(f"Accuracy of the network on the fake images from {name_B} to {name_A}: {100 * acc_A}%")
    print(f"Accuracy of the network on the fake images from {name_A} to {name_B}: {100 * acc_B}%")
    
if __name__ == '__main__':
    index_loader = DataLoader_Classifier(img_size=64)
    labels_name = index_loader.get_labels_names()
    idx_to_class = {v: k for k, v in labels_name.items()}
    
    netC = Classifier(input_size=64)
    
    # Load the model
    check_point_aca_n = 25
    check_point_scc_n = 25
    check_point_aca_scc = 35
    check_point = 50
    
    g_save_path = opt.gan_save_path
    c_save_path = opt.classifier_save_path


    netC.load_state_dict(torch.load(os.path.join(c_save_path, f"{name_type}/netC_{check_point}.pth")))
    netC.eval()
    netC.to(device)

    analysis(check_point_aca_n, netC, idx_to_class, device, name_A, name_B)
        
    analysis(check_point_scc_n, netC, idx_to_class, device, name_C, name_B)
    
    analysis(check_point_aca_scc, netC, idx_to_class, device, name_A, name_C)