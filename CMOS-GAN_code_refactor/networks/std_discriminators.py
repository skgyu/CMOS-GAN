# -*- coding: utf-8 -*-
from .blocks import *
from torch.nn import DataParallel as DPL
import torch

def get_discriminator(input_nc,params,gpu_ids):

    return Dis_std( input_dim=input_nc,  params=params, gpu_ids=gpu_ids)

class Dis_std(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, params , gpu_ids):

        super(Dis_std, self).__init__()

        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        self.input_dim = input_dim
        self.gpu_ids =  gpu_ids

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.cnns = nn.ModuleList()  

        #ModuleList can be indexed like a regular Python list, 
        #but modules it contains are properly registered, and will be visible by all Module methods. 
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

        if self.gan_type=='vanilla':
            self.register_buffer('real_label', torch.tensor(1.0))
            self.register_buffer('fake_label', torch.tensor(0.0))
            self.loss_func= nn.BCEWithLogitsLoss()


    def _make_net(self):

        dim = self.dim
        cnn_x = []
        
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):

        outputs = []
        for model in self.cnns:
            outputs.append( DPL(model,device_ids=self.gpu_ids)(x) )
            x = self.downsample(x)


        return outputs




    def calc_dis_loss(self, input_fake, input_real  ):
        # calculate the loss to train D


        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)

        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):

            if self.gan_type=='lsgan':
                loss +=  torch.mean((out0 - 0)**2)
                loss +=  torch.mean((out1 - 1)**2)

            elif self.gan_type=='vanilla':
                loss+=  (self.loss_func(  out0, self.fake_label.expand_as(out0))
                + self.loss_func(  out1, self.real_label.expand_as(out1)) ) *0.5


        return loss



    def calc_gen_loss(self, input_fake):

        outs0 = self.forward(input_fake)
 
        batchsize=input_fake.size()[0]

        loss = 0
        for it, out0 in enumerate(outs0):

            if self.gan_type=='lsgan':
                loss += torch.mean((out0 - 1)**2) 
            
            elif self.gan_type=='vanilla':
                loss+= self.loss_func(  out0, self.real_label.expand_as(out0))

        return loss