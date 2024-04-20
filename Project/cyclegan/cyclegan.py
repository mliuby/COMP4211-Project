import torch
import torch.nn as nn

from . import discriminators
from . import generators


class cycleG(nn.Module):
    def __init__(self):
        super(cycleG, self).__init__()
        self.generator_A_to_B = generators.Generator()
        self.generator_B_to_A = generators.Generator()
        
    def forward(self, real_A, real_B):
        fake_B = self.generator_A_to_B(real_A)
        fake_A = self.generator_B_to_A(real_B)
        return fake_A, fake_B
    
class cycleD(nn.Module):
    def __init__(self):
        super(cycleD, self).__init__()
        self.discriminator_A = discriminators.Discriminator()
        self.discriminator_B = discriminators.Discriminator()
        
    def forward(self, real_A, real_B):
        validity_A = self.discriminator_A(real_A)
        validity_B = self.discriminator_B(real_B)
        return validity_A, validity_B

