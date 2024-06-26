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

def train_classifier(netC = classifier.Classifier(), n_epochs = opt.n_epoch_c, lr = opt.lr_c, momentum = opt.momentum, save_path = opt.classifier_save_path, n_save = opt.n_save, check_point = opt.check_point_c, name_type=opt.name_T):
    
    dataloader = DataLoader_Classifier()

    criterion = nn.CrossEntropyLoss()
    
    netC.to(device)
    
    optimizer_C = optim.SGD(netC.parameters(), lr=lr, momentum=momentum)
    
    # load the model if it exists
    if check_point is not None and os.path.exists(os.path.join(save_path, f"{name_type}")):
        netC.load_state_dict(torch.load(os.path.join(save_path, f"{name_type}/netC_{check_point}.pth")))
        print(f"Model loaded from epoch {check_point}")
    else:
        print("Model is trained from scratch")
        
    print(f"Training Classifier in {device}")

    for epoch in range(n_epochs):
        
        total_loss = 0
        
        training_data, _ = dataloader.get_dataloader()
        
        pbar_batch = tqdm.tqdm(training_data)
        
        for i, data in enumerate(pbar_batch):

            images, labels = data
            
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer_C.zero_grad()
            
            outputs = netC(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_C.step()
            
            total_loss += loss.item()
            
            avg_loss = total_loss / (i+1)
            
            # Update progress bar
            pbar_batch.set_description(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss}")
            
        # Save models
        if (epoch+1) % n_save == 0:
            if not os.path.exists(os.path.join(save_path, f"{name_type}")):
                os.makedirs(os.path.join(save_path, f"{name_type}"))
            if check_point is not None:
                torch.save(netC.state_dict(), os.path.join(save_path, f"{name_type}/netC_{check_point+epoch+1}.pth"))
            else:
                torch.save(netC.state_dict(), os.path.join(save_path, f"{name_type}/netC_{epoch+1}.pth"))