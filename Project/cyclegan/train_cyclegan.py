import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import os

from data.dataloader import DataLoader_CycleGAN
import config
import cyclegan.cyclegan as cyclegan

opt = config.opt()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cycle_loss(real, fake):
    return torch.mean(torch.abs(real-fake)) / torch.mean(real)

def train_cyclegan(netG = cyclegan.cycleD(), netD = cyclegan.cycleG(), n_epochs = opt.n_epoch, lr = opt.lr, beta1 = opt.beta1, beta2 = opt.beta2, gen_loss_weight = opt.gen_loss_weight, cycle_loss_weight = opt.cycle_loss_weight, save_path = opt.gan_save_path, n_save = opt.n_save, name_A = 'A', name_B = 'B', check_point = opt.check_point):
    
    dataloader = DataLoader_CycleGAN()
    
    netG.to(device)
    netD.to(device)
    
    optimizer_G = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    
    pbar_epoch = tqdm.tqdm(range(n_epochs))
    
    # load the model if it exists
    if os.path.exists(os.path.join(save_path, f"generator/{name_A}_and_{name_B}/netG_{check_point}.pth")) and os.path.exists(os.path.join(save_path, f"discriminator/{name_A}_and_{name_B}/netD_{check_point}.pth")):
        netG.load_state_dict(torch.load(os.path.join(save_path, f"generator/{name_A}_and_{name_B}/netG_{check_point}.pth")))
        netD.load_state_dict(torch.load(os.path.join(save_path, f"discriminator/{name_A}_and_{name_B}/netD_{check_point}.pth")))
    
    for epoch in pbar_epoch:
        
        pbar_batch = tqdm.tqdm(dataloader.get_dataloader())
        
        for i, (real_A, real_B) in enumerate(pbar_batch):
            
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            # Train Discriminators
            optimizer_D.zero_grad()
            
            fake_A, fake_B = netG(real_A, real_B)
            
            validity_A_real, validity_B_real = netD(real_A, real_B)
            validity_A_fake, validity_B_fake = netD(fake_A, fake_B)
            
            d_loss_real = -torch.log(validity_A_real+1e-20) - torch.log(validity_B_real+1e-20)
            d_loss_fake = -torch.log(1-validity_A_fake+1e-20) - torch.log(1-validity_B_fake+1e-20)
            d_loss = d_loss_fake + d_loss_real
            
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generators
            optimizer_G.zero_grad()
            
            fake_A, fake_B = netG(real_A, real_B)
            cycle_A, cycle_B = netG(fake_A, fake_B)
            
            validity_A_fake, validity_B_fake = netD(fake_A, fake_B)
            
            g_loss_fake = -torch.log(validity_A_fake+1e-20) - torch.log(validity_B_fake+1e-20)
            
            g_loss_cycle = cycle_loss(real_A, cycle_A) + cycle_loss(real_B, cycle_B)

            g_loss = gen_loss_weight * g_loss_fake + cycle_loss_weight * g_loss_cycle
            
            g_loss.backward()
            optimizer_G.step()
            
            # Update progress bar
            pbar_batch.set_description(f"Epoch {epoch+1}/{n_epochs}, Batch {i+1}/{dataloader.get_batch_size()}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
            
        # Save models
        if (epoch+1) % n_save == 0:
            torch.save(netG.state_dict(), os.path.join(save_path, f"generator/{name_A}_and_{name_B}/netG_{epoch+1}.pth"))
            torch.save(netD.state_dict(), os.path.join(save_path, f"discriminator/{name_A}_and_{name_B}/netD_{epoch+1}.pth"))