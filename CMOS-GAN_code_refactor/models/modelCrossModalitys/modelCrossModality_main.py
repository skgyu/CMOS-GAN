#-*- coding:utf-8 -*-
import torch
from util.image_pool import ImagePool
from ..base_model import BaseModel
import networks
import numpy as np
from torch.nn import DataParallel as DPL
from sync_batchnorm import  convert_model
from stools import sutil



def L2dis(self,a,b):

    c=b.T

    return (a**2).sum(1,keepdims=True)+ (c**2).sum(0,keepdims=True)- 2 * np.matmul(a,c) 


def corresponding_image(self, idx,bs):

    img  =  self.data_X_source[idx:idx+1] if idx<bs else self.data_X_target[idx-bs:idx-bs+1]

    return img


def initialize_main(self,opt):

    if 'use_GAN' not in opt:
        opt['use_GAN'] = False    

    if 'use_FFL' not in opt:
        opt['use_FFL'] = False


    sutil.log('modelCrossModality_S2P','main')
    print('modelCrossModality_S2P')


    ##################################################################################
    self.generate_names=['encoder_X_source','decoder_Y'  ]
    self.auxiliary_names=['feature_extraction_model']
    self.optimizer_names=['optimizer_G','optimizer_C']



    if opt.use_GAN:
        self.auxiliary_names=['feature_extraction_model','dis_Y']
        self.optimizer_names=['optimizer_G','optimizer_C','optimizer_D']
    ##################################################################################

    print('initialize_S2P')



    self.encoder_X_source=  networks.generators.get_encoder(input_dim=opt.dim_X,params=opt['gen'])
    self.decoder_Y=  networks.generators.get_decoder(input_dim=self.encoder_X_source.output_dim ,params=opt['gen'],output_dim=opt.dim_Y)
    self.decoder_Y= convert_model(self.decoder_Y)
    self.G_X2Y_source=  networks.generators.VAE(self.encoder_X_source,self.decoder_Y)


    self.load_feature_extraction_model(opt=opt)


    if self.isTrain and opt.use_GAN:
        self.dis_Y =  networks.std_discriminators.get_discriminator(input_nc= 3 ,params=opt['dis'],gpu_ids=opt.gpu_ids)  # discriminator for domain a

    if not self.isTrain or opt.continue_train:  #test or continue train
        which_epoch = opt.which_epoch 
        self.load_network(self.encoder_X_source, 'encoder_X_source', which_epoch)
        self.load_network(self.decoder_Y, 'decoder_Y', which_epoch)
        self.load_network(self.feature_extraction_model, 'feature_extraction_model', which_epoch)

        if self.isTrain:
            if opt.use_GAN:
                self.load_network(self.dis_Y, 'dis_Y', which_epoch)

    else:  #####   train from scratch  

        print('#########train from scratch#############')
        for name in self.generate_names:
            getattr(self,name).apply(networks.weights_init(opt['gen']['init'])  )

        if opt.use_GAN:
            self.dis_Y.apply(networks.weights_init('gaussian'))

    ###### GPU ###############
    for name in self.generate_names:
        getattr(self,name).to(self.opt.device)

    if self.isTrain and opt.use_GAN:
        self.dis_Y.to(self.opt.device)

    self.feature_extraction_model.to(self.opt.device)





    ####loss################
    if self.isTrain:
        self.init_loss()
        self.init_optG()
        self.init_optC()
        if opt.use_GAN:
            self.init_optD()

        if opt.continue_train:  
            self.load_optimizer(self.optimizer_G,'optimizer_G',opt.which_epoch )
            self.load_optimizer(self.optimizer_C,'optimizer_C',opt.which_epoch )
            #self.optimizer_C.param_groups[0]['lr']=opt.lr_group1
            #self.optimizer_C.param_groups[1]['lr']=opt.lr
            if opt.use_GAN:
                self.load_optimizer(self.optimizer_D, 'optimizer_D', opt.which_epoch)

        self.init_scheduler()

        self.fake_Y_target_pool = ImagePool(opt.pool_size)


    self.n_row = min(opt.batchSize, 8)
    self.nrow =  3


def train_G(self):


    opt=self.opt

    ###################GAN LOSS#############################################################
    fake_Y_source, C_X_source   =   DPL(self.G_X2Y_source,device_ids=self.gpu_ids)(self.data_X_source) 
    fake_Y_target, C_X_target   =   DPL(self.G_X2Y_source,device_ids=self.gpu_ids)(self.data_X_target) 




    loss_G_X2Y_target=0

    if opt.use_GAN:
        loss_G_X2Y_target = self.dis_Y.calc_gen_loss(input_fake=fake_Y_target )*self.opt.lambda_GAN
        self.update_loss(  'loss_G_X2Y_target', loss_G_X2Y_target.item() )
        print('loss_G_X2Y_target {}'.format(loss_G_X2Y_target.item() ) ) 


    ################################L1 loss###########################################################

    
    loss_L1_X2Y_source=0

    if opt.lambda_L1>0:

        loss_L1_X2Y_source =  self.criterion_L1(fake_Y_source,self.data_Y_source)*opt.lambda_L1
        self.update_loss(  'loss_L1_X2Y_source', loss_L1_X2Y_source.item() )
        print('loss_L1_X2Y_source {}'.format(loss_L1_X2Y_source.item() ) ) 



    fake_Y_source_caffe=self.feature_extraction_model.rcg_func(fake_Y_source)
    fake_Y_target_caffe=self.feature_extraction_model.rcg_func(fake_Y_target)


    face_feature_fake_Y_source, probe_fake_Y_source = DPL(self.feature_extraction_model,device_ids=self.gpu_ids)(fake_Y_source_caffe )




    loss_ffl=0

    if opt.use_FFL:

        with torch.no_grad():
            data_Y_source_caffe=self.feature_extraction_model.rcg_func(self.data_Y_source)
            face_feature_data_Y_source, probe_data_Y_source= DPL(self.feature_extraction_model,device_ids=self.gpu_ids)(data_Y_source_caffe)

        _,prin_out  =face_feature_data_Y_source

        _,ffe_output=face_feature_fake_Y_source

        nfeature=len(ffe_output)

        for i in range(len(prin_out)):
        
            prin_out[i]=prin_out[i].detach()


        for ffl_idx in range(nfeature):
            loss_ffl+=self.criterion_MSE(ffe_output[ffl_idx],prin_out[ffl_idx])*self.opt.lambda_FFL  #conv4_1

        self.update_loss(  'loss_ffl', loss_ffl.item() )
        print('loss_ffl {}'.format(loss_ffl.item() ) ) 

    
    #########################msssim ############################



    loss_cls_trainG=0

    if  not ('ablation' in opt and 'no_clsG' in opt['ablation']):



        loss_cls_trainG_fake_Y_source=self.classify_loss_forG(predict=probe_fake_Y_source, target=self.source_imageX_id, idx=0, lambda_cls=self.opt.lambda_cls)
        self.update_loss(  'loss_cls_trainG_fake_Y_source', loss_cls_trainG_fake_Y_source.item() )
        print('loss_cls_trainG_fake_Y_source {}'.format(loss_cls_trainG_fake_Y_source.item() ) ) 





        face_feature_fake_Y_target, probe_fake_Y_target = DPL(self.feature_extraction_model,device_ids=self.gpu_ids)(fake_Y_target_caffe )
        loss_cls_trainG_fake_Y_target=self.classify_loss_forG(predict=probe_fake_Y_target, target=self.target_imageX_id, idx=1, lambda_cls=self.opt.lambda_cls)
        self.update_loss(  'loss_cls_trainG_fake_Y_target', loss_cls_trainG_fake_Y_target.item() )
        print('loss_cls_trainG_fake_Y_target {}'.format(loss_cls_trainG_fake_Y_target.item() ) ) 

        loss_cls_trainG= loss_cls_trainG_fake_Y_source+ loss_cls_trainG_fake_Y_target
    
    
    
    loss_G =  loss_L1_X2Y_source  +  loss_cls_trainG + loss_G_X2Y_target + loss_ffl   
    loss_G.backward()





    


    self.fake_Y_source_caffe=fake_Y_source_caffe.detach()
    self.fake_Y_target_caffe=fake_Y_target_caffe.detach()

    self.fake_Y_target=fake_Y_target.detach() ####??????????


def train_C(self):



    opt=self.opt

    fake_Y_source_caffe   = self.fake_Y_source_caffe
    fake_Y_target_caffe   = self.fake_Y_target_caffe


    

    if opt.train_cls_data_source: 
        data_Y_source_caffe=self.feature_extraction_model.rcg_func(self.data_Y_source)
        face_feature_data_Y_source, probe_data_Y_source= DPL(self.feature_extraction_model,device_ids=self.gpu_ids)(data_Y_source_caffe)


    face_feature_fake_Y_source, probe_fake_Y_source= DPL(self.feature_extraction_model,device_ids=self.gpu_ids)(fake_Y_source_caffe)
    face_feature_fake_Y_target, probe_fake_Y_target= DPL(self.feature_extraction_model,device_ids=self.gpu_ids)(fake_Y_target_caffe)
    
    # print(face_feature_data_X_source[0].size())
    # print(face_feature_data_X_target[0].size())
    # print(face_feature_fake_Y_source[0].size())

    loss_cls_fake_Y_target=0
    loss_cls_fake_Y_source=0
    loss_cls_data_Y_source=0

    if 'lambda_temperature' in opt and  (opt.lambda_temperature !=1 or opt.lambda_temperature_decay!=1):
        temperature = opt.lambda_temperature *  (opt.lambda_temperature_decay**  min(self.epoch,9) ) if 'lambda_temperature' in opt else 1

        if opt.train_cls_data_source:        
            loss_cls_data_Y_source=self.classify_loss_forC(predict=probe_data_Y_source/temperature, target=self.source_imageY_id, idx=0, lambda_cls=self.opt.lambda_cls)

        loss_cls_fake_Y_source=self.classify_loss_forC(predict=probe_fake_Y_source/temperature, target=self.source_imageX_id, idx=2, lambda_cls=self.opt.lambda_cls)
        loss_cls_fake_Y_target=self.classify_loss_forC(predict=probe_fake_Y_target/temperature, target=self.target_imageX_id, idx=3, lambda_cls=self.opt.lambda_cls)

    else:

        if opt.train_cls_data_source:        
            loss_cls_data_Y_source=self.classify_loss_forC(predict=probe_data_Y_source, target=self.source_imageY_id, idx=0, lambda_cls=self.opt.lambda_cls)

        loss_cls_fake_Y_source=self.classify_loss_forC(predict=probe_fake_Y_source, target=self.source_imageX_id, idx=2, lambda_cls=self.opt.lambda_cls)
        loss_cls_fake_Y_target=self.classify_loss_forC(predict=probe_fake_Y_target, target=self.target_imageX_id, idx=3, lambda_cls=self.opt.lambda_cls)



    self.update_loss('loss_cls_fake_Y_source', loss_cls_fake_Y_source.item() )
    print('loss_cls_fake_Y_source={}'.format(loss_cls_fake_Y_source.item() ) )

    if opt.train_cls_data_source:  
        self.update_loss('loss_cls_data_Y_source', loss_cls_data_Y_source.item() )
        print('loss_cls_data_Y_source={}'.format(loss_cls_data_Y_source.item() ) )

    self.update_loss('loss_cls_fake_Y_target', loss_cls_fake_Y_target.item() )
    print('loss_cls_fake_Y_target={}'.format(loss_cls_fake_Y_target.item() ) )

    loss= loss_cls_fake_Y_source + loss_cls_fake_Y_target + loss_cls_data_Y_source
    loss.backward()


def optimize_step(self):

    self.feature_extraction_model.eval()
    self.optimizer_G.zero_grad()
    self.train_G()
    self.optimizer_G.step()

    if self.opt.use_GAN:
        self.optimizer_D.zero_grad()
        self.backward_D_target(fake_Y_target=self.fake_Y_target_pool.query(self.fake_Y_target)  )
        self.optimizer_D.step()
    
    self.feature_extraction_model.train()
    self.optimizer_C.zero_grad()
    self.train_C()
    self.optimizer_C.step()




def trainstate(self):

    for name in self.generate_names+self.auxiliary_names:
        getattr(self,name).train()


def evalstate(self):

    for name in self.generate_names+self.auxiliary_names:
        if hasattr(self,name):
            getattr(self,name).eval()


def saveall(self, epoch):

    for name in self.generate_names+self.auxiliary_names:
        ret=getattr(self, name)
        self.save_network(ret,name, epoch , self.gpu_ids )


    self.save_network(self.feature_extraction_model.backbone,'feature_extraction_model.backbone',epoch,self.gpu_ids)


    for name in self.optimizer_names:
        self.save_optimizer( getattr(self,name), name, epoch)



        


