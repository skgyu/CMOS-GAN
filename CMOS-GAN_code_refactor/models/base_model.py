import os
import torch
from torch import nn
from collections import OrderedDict
import numpy as np
from stools  import sutil
import csv
import copy
from  matplotlib import pyplot as plt
plt.switch_backend('agg')
import networks
from sync_batchnorm import  convert_model

class BaseModel(nn.Module):



    def name(self):

        return 'BaseModel'

    def initialize(self, opt):

        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.namedir = os.path.join(opt.checkpoints_dir, opt.name)

        self.save_dir= os.path.join(self.namedir,opt['train_step'])

        sutil.makedirs(self.save_dir)

        sutil.get_logger( os.path.join(opt.debug_info_dir, 'cls.log') , 'cls.log')
        sutil.get_logger( os.path.join(opt.debug_info_dir, 'cls_detail.log') , 'cls_detail.log')


        self.optimizer_names=['optimizer_D','optimizer_C','optimizer_G']


        plt.rcParams['savefig.dpi'] = 300  
        plt.rcParams['figure.dpi'] =  300 


    def get_image_paths(self):

        return self.image_paths

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        '''
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
        Copyright (c) 2016, Phillip Isola and Jun-Yan Zhu All rights reserved.
        '''
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)

        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])


    def save_optimizer(self, optimizer,optimizer_label, epoch_label):

        save_filename = '%s_optimizer_%s.pth' % (epoch_label, optimizer_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)


    def load_optimizer(self, optimizer,optimizer_label, epoch_label):

        save_filename = '%s_optimizer_%s.pth' % (epoch_label, optimizer_label)
        save_path = os.path.join(self.save_dir, save_filename)

        print(save_path)
        if not os.path.exists(save_path):
            print('{} load optimizer error, path does not exists'.format(optimizer_label)  )
            sutil.log('{} load optimizer error, path does not exists'.format(optimizer_label),'main')
            raise(RuntimeError('load optimizer error'))
            return False
        optimizer.load_state_dict(torch.load(save_path))

        return True


    def load_network(self, network, network_label, epoch_label):

        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)

        print(save_path)
        if not os.path.exists(save_path):
            print('{} load network error, path does not exists'.format(network_label)  )
            sutil.log('{} load network error, path does not exists'.format(network_label),'main')
            raise(RuntimeError('load network error'))
            return False

        network.load_state_dict(torch.load(save_path))

        return True

    def load_network_frompath(self, network, load_path):

        if not os.path.exists(load_path):
            print('{} have not been found and not been loaded'.format(load_path)  )
            sutil.log('{} have not been found and not been loaded'.format(load_path),'main')
            raise(RuntimeError('load error'))
            return False

        network.load_state_dict(torch.load(load_path))

        return True


    def update_learning_rate(self , epoch):

        for scheduler in self.schedulers:
            scheduler.step()
        
        self.record_learning_rate(epoch)

    def record_learning_rate(self,epoch):
        

        loc=os.path.join(self.opt.checkpoints_dir , self.opt.name , self.opt.train_step)
        loc=os.path.join(loc, 'learning_rate.txt')
        


        with open(loc, 'a') as f:

            f.write(  'epoch = {}\n'.format(epoch)  )

            for optimizer_name in self.optimizer_names:

                tmp_optimizer= getattr(self, optimizer_name)

                for param_id,param in enumerate(tmp_optimizer.param_groups):

                    lr = param['lr']

                    f.write( '{} '.format(optimizer_name) + 'param_id = {}'.format(param_id)   + ' , learning rate = %.7f\n' % lr)

                    print('lr={}'.format(lr))

        f.close()



    def clear_loss(self):

        self.sum_loss= OrderedDict()
        self.cnt_loss= OrderedDict()
        self.accuracy= np.array([0,0,0,0])
        self.accuracy_forG= np.array([0,0])


        self.sizes_C=np.array([0,0,0,0])
        self.sizes_C_forG=np.array([0,0])

    def update_loss(self,key,add,addcnt=1):

        if key not in  self.sum_loss:
            self.sum_loss[key]=0
            self.cnt_loss[key]=0

        self.sum_loss[key]+=  add
        self.cnt_loss[key]+=  addcnt




    def record_ave_errors(self,epoch=None):

        if epoch is None and hasattr(self,"epoch"):
            epoch=self.epoch


        loc=os.path.join(self.opt.expr_dir, 'loss_record.csv' )

        f=open(loc, 'a')
        writer = csv.writer(f)

        ret_item=['epoch']
        ret_value=[epoch if epoch is not None else '?']

        for key,value in self.sum_loss.items():
            ret_item.append(key)
            ret_value.append(self.sum_loss[key]/self.cnt_loss[key])
                
        writer.writerow(ret_item)
        writer.writerow(ret_value)
        f.close()

        csvfile=open(loc,'r')
        reader=csv.reader(csvfile)

        loss_names=[]
        losses={}
        epoches=[]


        for i,line in enumerate(reader):
            
            if i==0: 
                for loss_name in line[1:]:
                    loss_names.append(loss_name)
                    losses[loss_name]=[]
                continue
            
            if i%2==0:
                continue
                
            epoches.append( float(line[0]) )
            for i,loss_num in enumerate(line[1:]):
                loss_name= loss_names[i]
                losses[loss_name].append( float(loss_num) )
                
        csvfile.close()
 
        plt.figure()
        plt.title("Losses During Training")

        for loss_name in loss_names:
            plt.plot(epoches,losses[loss_name],label=loss_name)

        plt.xlabel("trained epochs")
        plt.ylabel("losses")
        plt.legend()
        saveloc=os.path.join(self.opt.debug_info_dir, 'losses_{}epochs.png'.format(epoch) )
        plt.savefig( saveloc) 
        plt.clf()
        



    def get_ave_errors(self,epoch=None):
        
        self.record_ave_errors(epoch)
        ret=copy.deepcopy(self.sum_loss)
        for key,value in ret.items():
             ret[key]/=self.cnt_loss[key]
        return  ret

    def save(self, epoch):

        self.saveall(epoch)


    def saveall(self, epoch):

        for name in self.generate_names+self.auxiliary_names:
            ret=getattr(self, name)
            self.save_network(ret,name, epoch , self.gpu_ids )

        for name in self.optimizer_names:
            ret=getattr(self,name)
            self.save_optimizer(ret,name, epoch)




    def init_loss(self):
        
        self.criterion_L1=torch.nn.L1Loss()
        self.criterion_MSE=torch.nn.MSELoss()
        self.criterion_cls= torch.nn.CrossEntropyLoss()




    def load_feature_extraction_model(self,opt):

        sutil.log('opt.num_identities={}'.format(opt.num_identities),'main')


        if opt['recog_state_dict']['type'] in ['resnet50','resnet50_depth']:
            self.feature_extraction_model = networks.get_ResNet50(input_dim=3,num_classes=opt.num_identities)
            state_dict=  torch.load(  opt.recog_state_dict.loc) 
            self.feature_extraction_model = convert_model(self.feature_extraction_model)
            self.feature_extraction_model.apply( networks.weights_init(opt['finetune']['init']  )  )
        else:
            rasie(RuntimeError('no such model type'))


        ret = self.feature_extraction_model
        if 'part' in  opt['recog_state_dict']:
            ret = getattr(ret,opt['recog_state_dict']['part'])
        
        ret.load_state_dict(state_dict)
        self.feature_extraction_model.cuda()
        self.feature_extraction_model.eval()

        self.feature_extraction_model.rcg_func= opt.rcg_func




    def init_optG(self):

        self.optimizer_G = torch.optim.Adam( self.G_X2Y_source.parameters(), lr=self.opt.optimizer_G.lr[0], betas=(self.opt.beta1, 0.999))



    def init_optC(self):

        if self.opt['recog_state_dict']['type'] in  ['resnet50_depth','resnet50']:
            #params_group1 = self.feature_extraction_model.backbone.conv1_7x7_s2.parameters()

            assert self.opt['recog_state_dict']['policy'] in ['split','normal']


            if  self.opt['recog_state_dict']['policy']=='split':

                params_group1 =  [params for params in self.feature_extraction_model.classifier_1.parameters()]
                    #[params for params in self.feature_extraction_model.backbone.conv1_7x7_s2.parameters()] +                
                params_group1_id = list(map(id, params_group1))
                params_group2 = filter(lambda p: id(p) not in params_group1_id, self.feature_extraction_model.parameters())
                self.optimizer_C = torch.optim.Adam( [{'params': params_group1, 'lr': self.opt.optimizer_C.lr[0]},
                                {'params': params_group2, 'lr': self.opt.optimizer_C.lr[1]}], lr=self.opt.optimizer_C.lr[1]
                                , betas=(self.opt.beta1, 0.999))


            elif self.opt['recog_state_dict']['policy']=='normal':

                
                self.optimizer_C = torch.optim.Adam( self.feature_extraction_model.parameters() , lr=self.opt.optimizer_C.lr[0], betas=(self.opt.beta1, 0.999))

            else:
                raise(RuntimeError('no such policy or not need train'))

        else:
            self.optimizer_C = torch.optim.Adam( self.feature_extraction_model.parameters() , lr=self.opt.optimizer_C.lr[0], betas=(self.opt.beta1, 0.999))        