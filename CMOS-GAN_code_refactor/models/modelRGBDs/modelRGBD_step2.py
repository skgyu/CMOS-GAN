#-*- coding:utf-8 -*-
import torch
import networks
import random,os
import numpy as np
from torch.nn import DataParallel as DPL
from sync_batchnorm import  convert_model
from stools import sutil
from util.image_pool import ImagePool
from tqdm import tqdm


def initialize_main(self,opt):

    opt['use_GAN'] = False
    opt['use_FFL'] = False
    opt['use_normalization']=False
    opt['hard_mining']=True
    


    sutil.log('RGBandDepth_main_hard_negative','main')
    print('RGBandDepth_main_hard_negative')


    ##################################################################################
    self.generate_names=['encoder_X_source','decoder_Y'  ]
    self.auxiliary_names=['feature_extraction_model']
    self.optimizer_names=['optimizer_G']

    if opt.use_GAN:
        self.auxiliary_names=['feature_extraction_model','dis_Y']
        self.optimizer_names=['optimizer_G','optimizer_D']

    ##################################################################################

    print('initialize_RGBandDepth_main')


    self.encoder_X_source=  networks.generators.get_encoder(input_dim=opt.dim_X,params=opt['gen'])
    self.decoder_Y=  networks.generators.get_decoder(input_dim=self.encoder_X_source.output_dim ,params=opt['gen'],output_dim=opt.dim_Y)
    self.decoder_Y= convert_model(self.decoder_Y)
    self.G_X2Y_source=  networks.generators.VAE(self.encoder_X_source,self.decoder_Y)


    self.load_feature_extraction_model(opt=opt)

    if self.isTrain and opt.use_GAN:
        self.dis_Y =  networks.std_discriminators.get_discriminator(input_nc= 3 ,params=opt['dis'],gpu_ids=opt.gpu_ids)  # discriminator for domain a


    if not self.isTrain or opt.continue_train:   #test or continue train
        which_epoch = opt.which_epoch 
        self.load_network(self.encoder_X_source, 'encoder_X_source', which_epoch)
        self.load_network(self.decoder_Y, 'decoder_Y', which_epoch)
        self.load_network(self.feature_extraction_model, 'feature_extraction_model', which_epoch)

        if self.isTrain:
            if opt.use_GAN:
                self.load_network(self.dis_Y, 'dis_Y', which_epoch)


    else:  ############load from previous step#############  

        print('#########load from previous step#############')
        self.load_network_frompath(self.encoder_X_source, opt['gen']['en_load_path'])
        self.load_network_frompath(self.decoder_Y, opt['gen']['de_load_path'])

        if opt.use_GAN:
            self.load_network_frompath(self.dis_Y, opt['dis']['load_path'])

    ###### GPU ###############
    for name in self.generate_names:
        getattr(self,name).to(self.opt.device)

    if self.isTrain and opt.use_GAN:
        self.dis_Y.to(self.opt.device)

    self.feature_extraction_model.to(self.opt.device)




    ####loss################
    if self.isTrain:
        self.init_loss()
        if 'hard_mining' in opt and opt.hard_mining:
            self.criterion_triplet_loss = torch.nn.TripletMarginLoss(margin= opt.triplet_margin, p= opt.triplet_norm)
        self.init_optG()
        if opt.use_GAN:
            self.init_optD()

        if opt.continue_train:  
            self.load_optimizer(self.optimizer_G,'optimizer_G',opt.which_epoch )
            #self.optimizer_C.param_groups[0]['lr']=opt.lr_group1
            #self.optimizer_C.param_groups[1]['lr']=opt.lr
            if opt.use_GAN:
                self.load_optimizer(self.optimizer_D, 'optimizer_D', which_epoch)

        self.init_scheduler()
        self.fake_Y_target_pool = ImagePool(opt.pool_size)



    if 'hard_mining' in opt and opt.hard_mining:

        self.face_dataset=opt.face_dataset
        face_dataset=self.face_dataset

        self.name2pos, self.pos2class = face_dataset.get_name2pos_pos2class()



    self.n_row = min(opt.batchSize, 8)
    self.nrow =  3


    sutil.get_logger(path=os.path.join(opt.expr_dir ,'semi_idx_in_batch.log' ),name='semi_idx_in_batch' ) ###





def semi_hard_negative_mining(self,numpy_feature_fake_Y_source, numpy_feature_fake_Y_target,log=False):

    print('...hard_negative_mining...')


    bs=   self.data_X_source.size(0)
    face_dataset=self.face_dataset

    opt=self.opt

    source_id2images=face_dataset.source_id2images
    target_id2images=face_dataset.target_id2images


    print('...get poses and classes...')
    classes=[]
    for i in  tqdm(range(bs)):
        pos = self.name2pos[ sutil.get_file_name(self.source_imageX_paths[i])    ]
        classes.append(  self.pos2class[pos]  )

    for i in  tqdm(range(bs)):
        pos = self.name2pos[ sutil.get_file_name(self.target_imageX_paths[i])    ]
        classes.append(  self.pos2class[pos]  )

    classes=np.array(classes)

    data_matrix= np.concatenate([numpy_feature_fake_Y_source, numpy_feature_fake_Y_target], axis=0)



    dis_matrix= self.L2dis(data_matrix,data_matrix)


    
    print('...argsort dis_matrix...')

    sorted_indexs=np.argsort(dis_matrix,axis=1)  # distantce from small to big


    pos2class_ret= []
    for i in range(bs):
        pos2class_ret.append(   face_dataset.get_id_from_loc(  self.source_imageX_paths[i] ) )
    for i in range(bs):
        pos2class_ret.append(   face_dataset.get_id_from_loc(  self.target_imageX_paths[i] ) )
    pos2class_ret = np.array(pos2class_ret)


    sorted_classes=pos2class_ret[  sorted_indexs ]  #class id distantce from small to big

   



    reverse_sorted_classes= sorted_classes[:,::-1]    #class id distantce from big to small


    negative_poses= np.argmax( np.logical_not( sorted_classes==  np.expand_dims(classes,axis=1) ) , axis=1) #return the first 
    positive_poses= np.argmax(  reverse_sorted_classes==  np.expand_dims(classes,axis=1)   , axis=1) 
    


    source_positive_samples = []
    source_negative_samples = []
    target_positive_samples = []
    target_negative_samples = []


    source_positive_records = []
    source_negative_records = []
    target_positive_records = []
    target_negative_records = []

    # negative_poses_supp=[]
    # positive_poses_supp=[]


    print('...get triplet samples...')
    for i,negative_pos in tqdm( enumerate(negative_poses) ):
        positive_pos=  2*bs-1-positive_poses[i]

        if i< bs:

            if sorted_classes[i][negative_pos]==classes[i]:
                #negative_poses_supp.append(i)
                #source_negative_samples.append( np.zeros(1,3,224,224) )

                real_class=  self.source_imageX_id[i].item()
                neg_class_id = np.random.randint( opt.num_identities ) 
                if neg_class_id==real_class:
                    neg_class_id= (neg_class_id+1)%opt.num_identities

                id2images=   source_id2images if neg_class_id in  source_id2images else  target_id2images
                att_name=    'source_imageXs'  if neg_class_id in  source_id2images else  'target_imageXs'

                tlen=len(id2images[neg_class_id])
                random_image_num=np.random.randint(0,tlen)
                negative_sample=  getattr(face_dataset,att_name)[ id2images[neg_class_id][random_image_num]   ]  
                source_negative_samples.append(face_dataset.get_samples_from_loc(negative_sample).cuda() )


                if log:
                    source_negative_records.append(  str(i)+':-1' )


            else:
                source_negative_sample=self.corresponding_image( sorted_indexs[i][negative_pos],bs )
                source_negative_samples.append(source_negative_sample)

                if log:
                    source_negative_records.append(  str(i)+':'+str(sorted_indexs[i][negative_pos] ) )


            if sorted_classes[i][positive_pos]!=classes[i] or sorted_indexs[i][positive_pos]==i:
                #positive_poses_supp.append(i)
                #source_positive_samples.append( np.zeros(1,3,224,224) )

                real_class=  self.source_imageX_id[i].item()
                tlen=len(source_id2images[real_class])
                random_image_num=np.random.randint(0,tlen)
                positive_sample=  face_dataset.source_imageXs[ source_id2images[real_class][random_image_num]   ]  
                if positive_sample  == self.source_imageX_paths[i]:
                    random_image_num = (random_image_num+1)%tlen
                    positive_sample  = face_dataset.source_imageXs[ source_id2images[real_class][random_image_num]   ]  

                source_positive_samples.append(face_dataset.get_samples_from_loc(positive_sample).cuda() )


                if log:
                    source_positive_records.append(  str(i)+':-1' )

            else:

                source_positive_sample=self.corresponding_image( sorted_indexs[i][positive_pos],bs )
                source_positive_samples.append(source_positive_sample)


                if log:
                    source_positive_records.append(  str(i)+':'+str(sorted_indexs[i][positive_pos] ))


        else:

            if sorted_classes[i][negative_pos]==classes[i]:
                # negative_poses_supp.append(i)
                # target_negative_samples.append( np.zeros(1,3,224,224) )

                real_class=  self.target_imageX_id[i-bs].item()  
                neg_class_id = np.random.randint( opt.num_identities ) 
                if neg_class_id==real_class:
                    neg_class_id= (neg_class_id+1)%opt.num_identities

                id2images=   source_id2images if neg_class_id in  source_id2images else  target_id2images
                att_name=    'source_imageXs'  if neg_class_id in  source_id2images else  'target_imageXs'
                tlen=len(id2images[neg_class_id])
                random_image_num=np.random.randint(0,tlen)
                negative_sample=  getattr(face_dataset,att_name)[ id2images[neg_class_id][random_image_num]   ]  
                target_negative_samples.append(face_dataset.get_samples_from_loc(negative_sample).cuda() )


                if log:
                    target_negative_records.append(  str(i)+':-1' )


            else:
                target_negative_sample=self.corresponding_image( sorted_indexs[i][negative_pos], bs )
                target_negative_samples.append(target_negative_sample)

                if log:
                    target_negative_records.append(  str(i)+':'+str(sorted_indexs[i][negative_pos]) )



            if sorted_classes[i][positive_pos]!=classes[i] or sorted_indexs[i][positive_pos]==i: 
                # positive_poses_supp.append(i)
                # target_positive_samples.append( np.zeros(1,3,224,224) )
                real_class=  self.target_imageX_id[i-bs].item()
                tlen=len(target_id2images[real_class])
                random_image_num=np.random.randint(0,tlen)
                positive_sample=  face_dataset.target_imageXs[ target_id2images[real_class][random_image_num]   ]  
                if positive_sample  == self.target_imageX_paths[i-bs]:
                    random_image_num = (random_image_num+1)%tlen
                    positive_sample  = face_dataset.target_imageXs[ target_id2images[real_class][random_image_num]   ]  

                target_positive_samples.append(face_dataset.get_samples_from_loc(positive_sample).cuda() )

                if log:
                    target_positive_records.append(  str(i)+':-1' )


            else:
                
                target_positive_sample=self.corresponding_image( sorted_indexs[i][positive_pos], bs )
                target_positive_samples.append(target_positive_sample)

                if log:
                    target_positive_records.append( str(i)+':'+str(sorted_indexs[i][positive_pos]) )


    self.source_positive_samples = torch.cat(source_positive_samples,dim=0).cuda()
    self.source_negative_samples = torch.cat(source_negative_samples,dim=0).cuda()
    self.target_positive_samples = torch.cat(target_positive_samples,dim=0).cuda()
    self.target_negative_samples = torch.cat(target_negative_samples,dim=0).cuda()

    if log:
        sutil.log( 'source_positive_records' ,'semi_idx_in_batch' )
        sutil.log( source_positive_records ,'semi_idx_in_batch' )
        sutil.log( 'source_negative_records' ,'semi_idx_in_batch' )
        sutil.log( source_negative_records ,'semi_idx_in_batch' )
        sutil.log( 'target_positive_records' ,'semi_idx_in_batch' )
        sutil.log( target_positive_records ,'semi_idx_in_batch' )
        sutil.log( 'target_negative_records' ,'semi_idx_in_batch' )
        sutil.log( target_negative_records ,'semi_idx_in_batch' )

    #self.target_positive_samples
    #self.target_negative_samples

def train_G(self):
    ## 12_21 V1


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
    face_feature_fake_Y_target, probe_fake_Y_target = DPL(self.feature_extraction_model,device_ids=self.gpu_ids)(fake_Y_target_caffe )

    

 

    if 'hard_mining' in opt and opt.hard_mining and opt.lambda_triplet>0:

        # numpy_feature_fake_Y_source  =  face_feature_fake_Y_source[0].detach().cpu().float().numpy()
        # numpy_feature_fake_Y_target  =  face_feature_fake_Y_target[0].detach().cpu().float().numpy()

        pytorch_feature_fake_Y_source  =  face_feature_fake_Y_source[0].detach()
        pytorch_feature_fake_Y_target  =  face_feature_fake_Y_target[0].detach()

        if opt.use_normalization:
            pytorch_feature_fake_Y_source=torch.nn.functional.normalize(pytorch_feature_fake_Y_source, p=2, dim=1, eps=1e-12)
            pytorch_feature_fake_Y_target=torch.nn.functional.normalize(pytorch_feature_fake_Y_target, p=2, dim=1, eps=1e-12)

        numpy_feature_fake_Y_source  =  pytorch_feature_fake_Y_source.cpu().float().numpy()
        numpy_feature_fake_Y_target  =  pytorch_feature_fake_Y_target.cpu().float().numpy()




        self.semi_hard_negative_mining(numpy_feature_fake_Y_source, numpy_feature_fake_Y_target)




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

        loss_cls_trainG_fake_Y_source=self.classify_loss_forG(predict=probe_fake_Y_source, target=self.source_imageY_id, idx=0, lambda_cls=self.opt.lambda_cls)
        self.update_loss(  'loss_cls_trainG_fake_Y_source', loss_cls_trainG_fake_Y_source.item() )
        print('loss_cls_trainG_fake_Y_source {}'.format(loss_cls_trainG_fake_Y_source.item() ) ) 


        loss_cls_trainG_fake_Y_target=self.classify_loss_forG(predict=probe_fake_Y_target, target=self.target_imageX_id, idx=1, lambda_cls=self.opt.lambda_cls)
        self.update_loss(  'loss_cls_trainG_fake_Y_target', loss_cls_trainG_fake_Y_target.item() )
        print('loss_cls_trainG_fake_Y_target {}'.format(loss_cls_trainG_fake_Y_target.item() ) ) 

        loss_cls_trainG= loss_cls_trainG_fake_Y_source+ loss_cls_trainG_fake_Y_target
    
    
    
    loss_G =   loss_L1_X2Y_source  +  loss_cls_trainG  + loss_G_X2Y_target + loss_ffl


    loss_G.backward()




    if 'hard_mining' in opt and opt.hard_mining and opt.lambda_triplet>0:

        fake_Y_source, C_X_source   =   DPL(self.G_X2Y_source,device_ids=self.gpu_ids)(self.data_X_source) 
        fake_Y_source_caffe=self.feature_extraction_model.rcg_func(fake_Y_source)
        face_feature_fake_Y_source, probe_fake_Y_source = DPL(self.feature_extraction_model,device_ids=self.gpu_ids)(fake_Y_source_caffe )



        nm_face_feature_fake_Y_source=face_feature_fake_Y_source[0]
        if opt.use_normalization:
            nm_face_feature_fake_Y_source=torch.nn.functional.normalize(nm_face_feature_fake_Y_source, p=2, dim=1, eps=1e-12)


        with torch.no_grad():
            
            fake_Y_source_positive_samples, C_X_source_positive_samples   =   DPL(self.G_X2Y_source,device_ids=self.gpu_ids)(self.source_positive_samples) 
            fake_Y_source_negative_samples, C_X_source_negative_samples   =   DPL(self.G_X2Y_source,device_ids=self.gpu_ids)(self.source_negative_samples) 

            
            fake_Y_source_positive_samples_caffe=self.feature_extraction_model.rcg_func(fake_Y_source_positive_samples)
            fake_Y_source_negative_samples_caffe=self.feature_extraction_model.rcg_func(fake_Y_source_negative_samples)

            
            face_feature_fake_Y_source_positive_samples, probe_fake_Y_source_positive_samples = DPL(self.feature_extraction_model,device_ids=self.gpu_ids)( fake_Y_source_positive_samples_caffe )
            face_feature_fake_Y_source_negative_samples, probe_fake_Y_source_negative_samples = DPL(self.feature_extraction_model,device_ids=self.gpu_ids)( fake_Y_source_negative_samples_caffe )

            # dis_pos=  torch.criterion_L1(probe_fake_Y_target, probe_fake_Y_target_positive_samples)
            # dis_neg=  torch.criterion_L1(probe_fake_Y_target, probe_fake_Y_target_negative_samples)

            # loss=  torch.max(0,  dis_pos - dis_neg + threshold_m )


            nm_face_feature_fake_Y_source_positive_samples=face_feature_fake_Y_source_positive_samples[0]
            nm_face_feature_fake_Y_source_negative_samples=face_feature_fake_Y_source_negative_samples[0]

            if opt.use_normalization:

                nm_face_feature_fake_Y_source_positive_samples=torch.nn.functional.normalize(nm_face_feature_fake_Y_source_positive_samples, p=2, dim=1, eps=1e-12)
                nm_face_feature_fake_Y_source_negative_samples=torch.nn.functional.normalize(nm_face_feature_fake_Y_source_negative_samples, p=2, dim=1, eps=1e-12)


        triplet_loss_source = self.criterion_triplet_loss(nm_face_feature_fake_Y_source, nm_face_feature_fake_Y_source_positive_samples, nm_face_feature_fake_Y_source_negative_samples) *opt.lambda_triplet


        self.update_loss(  'triplet_loss_source', triplet_loss_source.item() )
        print('triplet_loss_source {}'.format(triplet_loss_source.item() ) ) 
        
        triplet_loss_source.backward()



        fake_Y_target, C_X_target   =   DPL(self.G_X2Y_source,device_ids=self.gpu_ids)(self.data_X_target) 
        fake_Y_target_caffe=self.feature_extraction_model.rcg_func(fake_Y_target)
        face_feature_fake_Y_target, probe_fake_Y_target = DPL(self.feature_extraction_model,device_ids=self.gpu_ids)(fake_Y_target_caffe )


        nm_face_feature_fake_Y_target=face_feature_fake_Y_target[0]
        if opt.use_normalization:
            nm_face_feature_fake_Y_target=torch.nn.functional.normalize(nm_face_feature_fake_Y_target, p=2, dim=1, eps=1e-12)

        with torch.no_grad():

            fake_Y_target_positive_samples, C_X_target_positive_samples   =   DPL(self.G_X2Y_source,device_ids=self.gpu_ids)(self.target_positive_samples) 
            fake_Y_target_negative_samples, C_X_target_negative_samples   =   DPL(self.G_X2Y_source,device_ids=self.gpu_ids)(self.target_negative_samples) 

            fake_Y_target_positive_samples_caffe=self.feature_extraction_model.rcg_func(fake_Y_target_positive_samples)
            fake_Y_target_negative_samples_caffe=self.feature_extraction_model.rcg_func(fake_Y_target_negative_samples)

            face_feature_fake_Y_target_positive_samples, probe_fake_Y_target_positive_samples = DPL(self.feature_extraction_model,device_ids=self.gpu_ids)( fake_Y_target_positive_samples_caffe )
            face_feature_fake_Y_target_negative_samples, probe_fake_Y_target_negative_samples = DPL(self.feature_extraction_model,device_ids=self.gpu_ids)( fake_Y_target_negative_samples_caffe )


            nm_face_feature_fake_Y_target_positive_samples=face_feature_fake_Y_target_positive_samples[0]
            nm_face_feature_fake_Y_target_negative_samples=face_feature_fake_Y_target_negative_samples[0]

            if opt.use_normalization:
                nm_face_feature_fake_Y_target_positive_samples=torch.nn.functional.normalize(nm_face_feature_fake_Y_target_positive_samples, p=2, dim=1, eps=1e-12)
                nm_face_feature_fake_Y_target_negative_samples=torch.nn.functional.normalize(nm_face_feature_fake_Y_target_negative_samples, p=2, dim=1, eps=1e-12)


        triplet_loss_target = self.criterion_triplet_loss(nm_face_feature_fake_Y_target, nm_face_feature_fake_Y_target_positive_samples, nm_face_feature_fake_Y_target_negative_samples) *opt.lambda_triplet

        self.update_loss(  'triplet_loss_target', triplet_loss_target.item() )
        print('triplet_loss_target {}'.format(triplet_loss_target.item() ) ) 

        triplet_loss_target.backward()




    self.fake_Y_source_caffe=fake_Y_source_caffe.detach()
    self.fake_Y_target_caffe=fake_Y_target_caffe.detach()


    self.fake_Y_target=fake_Y_target.detach()



def optimize_step(self):
   

    self.feature_extraction_model.eval()
    self.optimizer_G.zero_grad()
    self.train_G()
    self.optimizer_G.step()


    if self.opt.use_GAN:
        self.optimizer_D.zero_grad()
        self.backward_D_target(fake_Y_target=self.fake_Y_target_pool.query(self.fake_Y_target)  )
        self.optimizer_D.step()
    


