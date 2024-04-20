import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List

import config

opt = config.opt()

class Swish(nn.Module):
    def forward(self,x):
        return x*torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,n_groups=32,dropout=opt.dropout):
        super().__init__()
        self.norm1=nn.GroupNorm(n_groups,in_channels)
        self.act1=Swish()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=(3,3),padding=(1,1))
        self.norm2=nn.GroupNorm(n_groups,out_channels)
        self.act2=Swish()
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=(3,3),padding=(1,1))
        self.shortcut=nn.Conv2d(in_channels,out_channels,kernel_size=(1,1)) if in_channels!=out_channels else nn.Identity()
        self.dropout=nn.Dropout(dropout) if dropout is not None else None
    def forward(self,x):
        h=self.conv1(self.act1(self.norm1(x)))
        if self.dropout is None:
            h=self.conv2(self.act2(self.norm2(h)))
        else:
            h=self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h+self.shortcut(x)/torch.sqrt(torch.tensor(2.0))

class AttentionBlock(nn.Module):
    def __init__(self,n_channels,n_heads=8,d_k=None,n_groups=32):
        super().__init__()
        if d_k is None:
            d_k=n_channels
        self.n_heads=n_heads
        self.d_k=d_k
        self.norm=nn.GroupNorm(n_groups,n_channels)
        self.projection=nn.Linear(n_channels,n_heads*d_k*3)
        self.output=nn.Linear(n_heads*d_k,n_channels)
        self.scale=d_k**-0.5
    def forward(self,x):
        batch_size,n_channels,height,width=x.shape
        x=x.view(batch_size,n_channels,-1).permute(0,2,1)
        qkv=self.projection(x).view(batch_size,-1,self.n_heads,3*self.d_k)
        q,k,v=torch.chunk(qkv,3,dim=-1)
        attn=torch.einsum('bihd,bjhd->bijh',q,k)*self.scale
        attn=attn.softmax(dim=2)
        res=torch.einsum('bijh,bjhd->bihd',attn,v)
        res=res.reshape(batch_size,-1,self.n_heads*self.d_k)
        res=self.output(res)
        res+=x
        res=res.permute(0,2,1).view(batch_size,n_channels,height,width)
        return res




class DownBlock(nn.Module):
    def __init__(self,in_channels,out_channels,has_attn):
        super().__init__()
        self.res=ResidualBlock(in_channels, out_channels)
        '''
        if has_attn:
            self.attn=AttentionBlock(out_channels)
        else:
          self.attn=nn.Identity()
        '''  
    def forward(self,x):
        x=self.res(x)
        #x=self.attn(x)
        return x

class UpBlock(nn.Module):
    def __init__(self,in_channels,out_channels,has_attn):
        super().__init__()
        self.res=ResidualBlock(in_channels+out_channels, out_channels)
        '''
        if has_attn:
            self.attn=AttentionBlock(out_channels)
        else:
          self.attn=nn.Identity()
        '''  
    def forward(self,x):
        x=self.res(x)
        #x=self.attn(x)
        return x
class MiddleBlock(nn.Module):
    def __init__(self,n_channels):
        super().__init__()
        self.res1=ResidualBlock(n_channels, n_channels)
        #self.attn=AttentionBlock(n_channels)
        self.res2=ResidualBlock(n_channels, n_channels)
    def forward(self,x,t):
        x=self.res1(x,t)
        #x=self.attn(x)
        x=self.res2(x,t)
        return x

class Upsample(nn.Module):
    def __init__(self,n_channels):
        super().__init__()
        self.conv=nn.ConvTranspose2d(n_channels,n_channels,(4,4),(2,2),(1,1))
    def forward(self,x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self,n_channels):
        super().__init__()
        self.conv=nn.Conv2d(n_channels,n_channels,(3,3),(2,2),(1,1))
    def forward(self,x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self,image_channels=4,n_channels=64,ch_mults=(1,2,2,2),is_attn=(False,False,False,False),n_blocks=(2,2,2,8)):
        super().__init__()
        n_resolutions=len(ch_mults)
        self.image_proj=nn.Conv2d(image_channels,n_channels,kernel_size=(3,3),padding=(1,1))

        down=[]
        out_channels=in_channels=n_channels
        down.append(Downsample(in_channels))
        for i in range(n_resolutions):
            out_channels=in_channels*ch_mults[i]
            for _ in range(n_blocks[i]):
                down.append(DownBlock(in_channels, out_channels, is_attn[i]))
                in_channels=out_channels
            if i<n_resolutions-1:
                down.append(Downsample(in_channels))
        self.down=nn.ModuleList(down)

        self.middle=MiddleBlock(out_channels)

        up=[]
        in_channels=out_channels
        for i in reversed(range(n_resolutions)):
            out_channels=in_channels
            for _ in range(n_blocks[i]):
                up.append(UpBlock(in_channels, out_channels, is_attn[i]))
            out_channels=in_channels//ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, is_attn[i]))
            in_channels=out_channels
            up.append(Upsample(in_channels))
        self.up=nn.ModuleList(up)

        self.norm=nn.GroupNorm(8,n_channels)
        self.act=Swish()
        self.final=nn.Conv2d(in_channels,1,kernel_size=(3,3),padding=(1,1))

    def forward(self,x):
        x=self.image_proj(x)
        h=[x]
        for m in self.down:
            x=m(x)
            h.append(x)
        for m in self.up:
            if isinstance(m,Upsample):
                x=m(x)
            else:
                s=h.pop()
                x=torch.cat((x,s),dim=1)
                x=m(x)

        return self.final(self.act(self.norm(x)))