import torch
import torch.nn as nn
import tqdm
from data.dataloader import DataLoader_Classifier
from classifier.classifier import Classifier
import numpy as np
import matplotlib.pyplot as plt
import os
from config import opt
def compute_saliency_maps(X, y, model):
    """
    Compute the saliency maps for a batch of input images.

    Inputs:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels corresponding to X; LongTensor of shape (N,)
    - model: Pre-trained neural network model used to compute X.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input images.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    X.requires_grad_(True)
    model.zero_grad()
    scores = model(X)
    scores = criterion(scores, labels)
    scores.backward(torch.ones_like(scores))
    saliency = X.grad
    saliency = saliency.abs()
    saliency, _ = torch.max(saliency, dim=1)

    return saliency

if __name__ == '__main__':
    opt = opt()
    dataloader = DataLoader_Classifier(img_size=64)
    labels_name = dataloader.get_labels_names()
    idx_to_class = {v: k for k, v in labels_name.items()}
    training_data, _ = dataloader.get_dataloader()
    pbar_batch = tqdm.tqdm(training_data)
    save_path = f'./output/saliency'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    netC = Classifier(input_size=64)
    check_point = 50
    c_save_path = opt.classifier_save_path
    name_type = opt.name_T
    netC.load_state_dict(torch.load(os.path.join(c_save_path, f"{name_type}/netC_{check_point}.pth")))
    netC.eval()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    netC.to(device)
    for i, data in enumerate(pbar_batch):
        if i>3:
            break
        images, labels = data
        images = images.to(device)
        N = images.shape[0]
        labels = labels.to(device)
        saliency = compute_saliency_maps(images, labels, netC)
        saliency = saliency.cpu().detach().numpy()
        images=images.cpu().detach().numpy()
        images = np.transpose(images, (0, 2, 3, 1))
        _, label = torch.max(labels.data, 1)
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        for k in range(N):
            plt.subplot(6, N, k + 1)
            plt.imshow(images[k])
            plt.axis('off')
            plt.title(idx_to_class[label[k].item()])
            plt.subplot(6, N, N + k + 1)
            plt.imshow(saliency[k], cmap=plt.cm.hot)
            plt.axis('off')
    plt.gcf().set_size_inches(12, 15)
    plt.savefig(os.path.join(save_path, f'saliency{i}.png'), bbox_inches='tight')