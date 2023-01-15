#-*- coding:utf-8 -*-
import torch
from collections import OrderedDict
import os
from torch.nn import DataParallel as DPL
from stools import sutil


import networks
import glob,torchvision


def testC(self,epoch):

    if self.sizes_C.sum()!=0:
       
        for x in range(self.sizes_C.shape[0]):
            if self.sizes_C[x]==0:
                self.sizes_C[x]=1

        sutil.log('epoch:='+ str(epoch),'cls.log'  )
        sutil.log('data_Y_source, data_Y_target, fake_Y_source, fake_Y_target','cls.log'  )
        sutil.log('class_accuracy:='+ str(1.0*self.accuracy/self.sizes_C),'cls.log'  )

    if self.sizes_C_forG.sum()!=0:
       
        for x in range(self.sizes_C_forG.shape[0]):
            if self.sizes_C_forG[x]==0:
                self.sizes_C_forG[x]=1

        sutil.log('epoch:='+ str(epoch),'cls.log'  )
        sutil.log('fake_Y_source, fake_Y_target','cls.log'  )
        sutil.log('class_accuracy_forG:='+ str(1.0*self.accuracy_forG/self.sizes_C_forG),'cls.log'  )





def test(self,epoch,phase,imgy_repeat3=False):


    n_row =  self.n_row
    nrow =  2*self.nrow


    nimg=len( glob.glob( os.path.join(self.opt.debug_image_dir,'sample_{}*'.format(phase) )  ))
    typename=    'epoch{}_{}'.format(epoch,nimg) 
    sutil.log('type='+typename,'sample')

    images=[]

    images.append( self.data_X_source[0:self.n_row])
    with torch.no_grad():
        fake_Y_source,_    = DPL(self.G_X2Y_source,device_ids=self.gpu_ids)(self.data_X_source[0:self.n_row]) 
        if imgy_repeat3:
            images.append(fake_Y_source.repeat(1,3,1,1))
        else:
            images.append(fake_Y_source)

    if imgy_repeat3:
        images.append( self.data_Y_source[0:self.n_row].repeat(1,3,1,1))
    else:
        images.append( self.data_Y_source[0:self.n_row])


    images.append( self.data_X_target[0:self.n_row])
    with torch.no_grad():
        fake_Y_target,_    = DPL(self.G_X2Y_source,device_ids=self.gpu_ids)(self.data_X_target[0:self.n_row]) 
        if imgy_repeat3:
            images.append(fake_Y_target.repeat(1,3,1,1))
        else:
            images.append(fake_Y_target)


    if hasattr(self,'data_Y_target'):
        if imgy_repeat3:
            images.append( self.data_Y_target[0:self.n_row].repeat(1,3,1,1))
        else:
            images.append( self.data_Y_target[0:self.n_row])


    tensor_image=torch.cat(images,dim=0)
    image_grid=torchvision.utils.make_grid(tensor_image, nrow=n_row, padding=0, normalize=False, range=None, scale_each=False, pad_value=0)
    image_numpy=sutil.tensor2im(image_grid)
    sutil.save_image( image_numpy,  os.path.join(   self.opt.debug_image_dir  ,'sample_{}_{}.jpg'.format(phase,typename)   ) )




def test_input_output_network(self,input_name,output_name,network_name):
    
    with torch.no_grad():

        output,_ = DPL(getattr(self,network_name),device_ids=self.gpu_ids)(getattr(self,input_name) )

    output_numpy = sutil.tensor2images(output.data)
    
    return OrderedDict( [  (output_name, output_numpy )] )


