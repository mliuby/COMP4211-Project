import torch

from . import Unet
import config

opt = config.opt()

class Generator(torch.nn.Module):
    def __init__(self, input_channels = opt.channels, ngf = opt.ngf, ch_mults = opt.ch_mults, is_attn = opt.is_attn, n_blocks = opt.n_blocks):
        super(Generator, self).__init__()
        
        if len(ch_mults) == len(is_attn) == len(n_blocks):
            self.model = Unet.Unet(image_channels=input_channels,n_channels=ngf,ch_mults=ch_mults,is_attn=is_attn,n_blocks=n_blocks)
        
        else:
            raise ValueError("The length of ch_mults, is_attn, and n_blocks should be the same.")
        
    def forward(self, x):
        return self.model(x)
    