#-*- coding:utf-8 -*-
import torch
from .base_model import BaseModel
import os,copy
from stools import sutil
import networks,shutil


class modelCrossModality(BaseModel):
    def name(self):

        return 'modelCrossModality'

    def init_loss(self):

        BaseModel.init_loss(self)
        self.clear_loss()  






    def init_optD(self):

        
        self.optimizer_D = torch.optim.Adam( self.dis_Y.parameters(),lr=self.opt.optimizer_D.lr[0], betas=(self.opt.beta1, 0.999) )



    def init_scheduler(self):

        self.optimizers = []

        for name in self.optimizer_names:
            if hasattr(self,name):
                self.optimizers.append(getattr(self,name))


        self.schedulers = []
        for optimizer_name in self.optimizer_names:
            optimizer=getattr(self, optimizer_name)
            self.schedulers.append(networks.get_scheduler(optimizer, optimizer_name, self.opt))


    def initialize(self, opt):

        BaseModel.initialize(self, opt)

        steps=[ 'main']

        sutil.add_functions(self,dirs='models',model_name='test_functions')
        sutil.add_functions(self,dirs='models',model_name= 'backwardD_functions')
        sutil.add_functions(self,dirs='models',model_name= 'backwardC_functions')
        sutil.add_functions(self,dirs='models',model_name= 'input_functions')

        for step in steps:
            if opt['train_step'].startswith(step):

                sutil.add_functions(self,dirs=os.path.join('models',self.name()+'s'),model_name=self.name()+'_'+step)
                shutil.copy(  os.path.join('models',self.name()+'s',self.name()+'_'+step+'.py') , os.path.join(opt.expr_dir,self.name()+'_'+step+'.py'))
                
                if  hasattr(opt,'model_files'):  
                    for  file in opt.model_files:
                        sutil.add_functions(self,dirs=os.path.join('models',self.name()+'s'),model_name=file)
                        shutil.copy(  os.path.join('models',self.name()+'s',file+'.py') , os.path.join(opt.expr_dir,file+'.py'))
                

                getattr(self,'initialize_'+step)(opt)

                return



    def update_learning_rate(self , epoch):

        BaseModel.update_learning_rate(self,epoch)








