#-*- coding:utf-8 -*-
import torch




def set_input_pre(self, input ,pre='source_'):


    self.image_paths=input[pre+'imageX_path']
    imageXs= input[pre+'imageX'].to(self.opt.device)
    setattr(self, 'data_X_'+pre[:-1]  ,imageXs)

    if hasattr(input,pre+'imageY'):
        imageYs= input[pre+'imageY'].to(self.opt.device)
        setattr(self, 'data_Y_'+pre[:-1]  ,imageYs)




def set_input(self, input):
    

    self.data_X_target = input['target_imageX'].to(self.opt.device)
    self.target_imageX_id=   input['target_imageX_id'].to(self.opt.device)
    self.target_imageX_paths=input['target_imageX_path']


    if 'target_imageY' in input:
        
        self.data_Y_target  =  input['target_imageY'].to(self.opt.device)
        self.target_imageY_id  =  input['target_imageY_id'].to(self.opt.device)
        self.target_imageY_paths  =  input['target_imageY_path']
    

    self.data_Y_source=input['source_imageY'].to(self.opt.device)    
    self.source_imageY_id=       input['source_imageY_id'].to(self.opt.device)
    self.source_imageY_paths=input['source_imageY_path']


    self.data_X_source = input['source_imageX'].to(self.opt.device)   
    self.source_imageX_id=       input['source_imageX_id'].to(self.opt.device)
    self.source_imageX_paths=input['source_imageX_path']

    
    self.image_paths=input['target_imageX_path']




