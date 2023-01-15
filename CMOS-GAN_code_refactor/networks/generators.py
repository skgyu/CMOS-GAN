# -*- coding: utf-8 -*-
from .init import *
from .blocks import *


def get_encoder(input_dim,params):

    return Encoder(input_dim=input_dim,n_downsample=params['n_downsample'], n_res=params['n_res'], dim=params['dim'], norm=params['en_norm'], activ=params['activ'], pad_type=params['pad_type'])

def get_decoder(input_dim,params,output_dim):  #  decoder.output_dim   ,  input_dim ,    

    return Decoder(n_upsample=params['n_downsample'], n_res=params['n_res'],  dim=input_dim, output_dim=output_dim, norm=params['de_norm'], activ=params['activ'], pad_type=params['pad_type'])


class VAE(nn.Module):

    def __init__(self, enc, dec ):
        super(VAE, self).__init__()
        self.enc= enc
        self.dec= dec
        

    def forward(self, images):

        hidden = self.enc(images)

        out = self.dec(x=hidden)

        return out, hidden



class Encoder(nn.Module):      

    def __init__(self, input_dim=3,n_downsample=2, n_res=4, dim=64, norm='in', activ='relu', pad_type='reflect' ):
        super(Encoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]

        self.model = nn.Sequential(*self.model)
        self.output_dim = dim


    def forward(self, x):

        x=self.model(x)

        return x




class Decoder(nn.Module):  

    def __init__(self,n_upsample=2, n_res=4, dim=256, output_dim=3, norm='bn', activ='relu', pad_type='reflect'):
        assert(n_res >= 0)
        super(Decoder, self).__init__()
        self.model = []
        self.model += [ResBlocks(n_res, dim, norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):  
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)]
            dim //= 2
        
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm=norm, activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)


    def forward(self, x):

        x= self.model(x)
        return x




