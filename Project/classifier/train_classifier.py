import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import os

from data.dataloader import DataLoader_Classifier
import config
import classifier.classifier as classifier

opt = config.opt()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_cyclegan(netC = classifier(), n_epochs = opt.n_epoch_c, lr = opt.lr_c, momentum = opt.momentum, save_path = opt.classifier_save_path, n_save = opt.n_save, name_C = 'C', check_point = opt.check_point_c):
    
    dataloader = DataLoader_Classifier()

    criterion = nn.CrossEntropyLoss()
    
    netC.to(device)
    
    optimizer_C = optim.SGD(netC.parameters(), lr=lr, momentum=momentum)

    pbar_epoch = tqdm.tqdm(range(n_epochs))
    
    # load the model if it exists
    if os.path.exists(os.path.join(save_path, f"classifer/{name_C}/.pth")):
        netC.load_state_dict(torch.load(os.path.join(save_path, f"classifer/{name_C}/netC_{check_point}.pth")))

    for epoch in pbar_epoch:
        
        pbar_batch = tqdm.tqdm(dataloader.get_dataloader())
        
        for i, data in enumerate(pbar_batch):

            images, labels = data
            
            images = images.to(device)
            
            optimizer_C.zero_grad()
            
            outputs = netC(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_C.step()
            
            
            # Update progress bar
            pbar_batch.set_description(f"Epoch {epoch+1}/{n_epochs}, Batch {i+1}/{dataloader.get_batch_size()}, Loss: {loss.item()}")
            
        # Save models
        if (epoch+1) % n_save == 0:
            torch.save(netC.state_dict(), os.path.join(save_path, f"classifer/{name_C}/.pth"))




            