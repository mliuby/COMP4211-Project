import torch
import torch.nn as nn
import numpy as np

import config

opt = config.opt()


class Classifier(nn.Module):
    def __init__(self, input_channels = opt.channels, input_size = opt.img_size, n_classes = opt.n_classes, ncf = opt.ncf):
        super(Classifier, self).__init__()
        self.input_size = input_size
        self.channels = input_channels
        self.n_classes = n_classes
        
        self.ncf = ncf
        self.input_size = input_size
        self.channels = input_channels

        size = self.input_size
        self.num_downsamples = int(np.log2(size) - np.log2(2)) 
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        for i in range(self.num_downsamples):
            out_channels = max(int(self.ncf * 2 ** (i - self.num_downsamples + 1)), 4)
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels) if i > 0 else nn.Identity(),  # Skip batchnorm for the first layer
                    nn.LeakyReLU(0.2, inplace=True),
                )
                
            )
            in_channels = out_channels
        
        final_conv_dim = 4
        self.final_dim = final_conv_dim * self.ncf
        
        self.linear = nn.Linear(self.final_dim, self.n_classes, bias=True)
        nn.init.xavier_uniform_(self.linear.weight.data, 1.)
        
    def forward(self, x):
        size = self.input_size
        out = x.view(-1, self.channels,size,size)
        
        for conv_layer in self.conv_layers:
            out = conv_layer(out)
        
        out = out.view(-1, self.final_dim)
        
        output = self.linear(out)
        
        output = torch.softmax(output, dim=1)
        
        return output