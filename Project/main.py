import torch
from classifier.classifier import Classifier
from cyclegan.cyclegan import cycleG
from data.dataloader import DataLoader_Classifier
from config import opt
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

opt = opt()

name_type = opt.name_T
name_A = 'lung_aca'
name_B = 'lung_n'
name_C = 'lung_scc'

name_R = name_B

name_r = name_R.replace("_", "")

img_path = f'./dataset/lung/{name_R}'
save_path = f'./output/{name_R}'


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def transform (size):
  return transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
])

def inference(image_path, save_path,netC, netG_aca_n, netG_aca_scc, netG_scc_n, idx_to_class, idx=0):
    raw_image = Image.open(image_path).convert('RGB')
    raw_image.show()
    
    image = transform(64)(raw_image).unsqueeze(0)
    if torch.cuda.is_available():
        image = image.to(device)

    output = netC(image)
    _, predicted = torch.max(output.data, 1)
    predicted_class = idx_to_class[predicted.item()]
    
    print(f"Predicted class: {predicted_class}")

    image = transform(96)(raw_image).unsqueeze(0)
    image = image.to(device)
    
    blank = torch.zeros(1, 3, 96, 96)

    blank = blank.to(device)

    match predicted_class:
        case 'lung_aca':
            image_aca = image
            _, image_n = netG_aca_n(image, blank)
            _, image_scc = netG_aca_scc(image, blank)
        case 'lung_scc':
            image_scc = image
            _, image_n = netG_scc_n(image, blank)
            image_aca, _ = netG_aca_scc(blank, image)
        case 'lung_n':
            image_n = image
            image_aca, _ = netG_aca_n(blank, image)
            image_scc, _ = netG_scc_n(blank, image)

    image_aca = image_aca.cpu().detach().numpy()
    image_n = image_n.cpu().detach().numpy()
    image_scc = image_scc.cpu().detach().numpy()

    image_aca = np.transpose(image_aca, (0, 2, 3, 1))
    image_n = np.transpose(image_n, (0, 2, 3, 1))
    image_scc = np.transpose(image_scc, (0, 2, 3, 1))

    plt.figure(figsize=(10, 10))

    plt.subplot(3, 1, 1)
    plt.imshow(image_n[0])
    plt.title('n (real)' if predicted_class == 'lung_n' else 'n (fake)')

    plt.subplot(3, 1, 2)
    plt.imshow(image_aca[0])
    plt.title('aca (real)' if predicted_class == 'lung_aca' else 'aca (fake)')

    plt.subplot(3, 1, 3)
    plt.imshow(image_scc[0])
    plt.title('scc (real)' if predicted_class == 'lung_scc' else 'scc (fake)')

    plt.savefig(os.path.join(save_path, f'output{idx}.png'))


if __name__ == '__main__':
    dataloader = DataLoader_Classifier(img_size=64)
    labels_name = dataloader.get_labels_names()
    idx_to_class = {v: k for k, v in labels_name.items()}
    
    netC = Classifier(input_size=64)
    netG_aca_n = cycleG()
    netG_scc_n = cycleG()
    netG_aca_scc = cycleG()
    
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
    
    netG_aca_n.load_state_dict(torch.load(os.path.join(g_save_path, f"generator/{name_type}/{name_A}_and_{name_B}/netG_{check_point_aca_n}.pth")))
    netG_scc_n.load_state_dict(torch.load(os.path.join(g_save_path, f"generator/{name_type}/{name_C}_and_{name_B}/netG_{check_point_scc_n}.pth")))
    netG_aca_scc.load_state_dict(torch.load(os.path.join(g_save_path, f"generator/{name_type}/{name_A}_and_{name_C}/netG_{check_point_aca_scc}.pth")))

    netG_aca_n.eval().to(device)
    netG_scc_n.eval().to(device)
    netG_aca_scc.eval().to(device)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(10):
        img_path_i = os.path.join(img_path, f"{name_r}{i+1}.jpeg")
        inference(img_path_i, save_path,netC, netG_aca_n, netG_aca_scc, netG_scc_n, idx_to_class, i)