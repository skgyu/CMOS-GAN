## coding:utf-8
import os
import argparse
import torch
import shutil
from stools import sutil
import random
import numpy as np
import torchvision.transforms as transforms
import yaml
from easydict import EasyDict as edict

class BaseOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--no_recognition', action='store_true',help='')
        self.parser.add_argument('--config', type=str, default='options/S2P.yaml', help='Path to the config file.')
        self.parser.add_argument('--test_fre', default=1, type=int, help='test_fre')
        self.parser.add_argument('--read_mode', default='PIL', type=str, help='read_mode')
        self.parser.add_argument('--save_img_feature', type=bool, default=True, help='')
        self.parser.add_argument('--use_data_probe', default=0.5, type=float)
        self.parser.add_argument('--use_percent_dataset', default=False, type= bool)
        self.parser.add_argument('--manualSeed', type=int, default=30, help='')
        self.parser.add_argument('--no_target_imageYs', type=bool, default=True, help='')
        
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        opt = self.parser.parse_args()
        opt = edict(vars(opt))
        opt.isTrain = self.isTrain   # train or test


        config_addrs= opt.config.split(',')

        for config_addr in config_addrs:

            config_addr=config_addr.strip()

            with open(config_addr, 'r') as stream:
                config =yaml.load(stream)
                for key,value in  config.items():        
                    opt[key]=value


        gpus=''
        for i in range(len(opt.gpu_ids)):
            gpus+= str(opt.gpu_ids[i])
            if i!= len( opt.gpu_ids)-1:
                gpus+=','

        print(gpus)
        os.environ['CUDA_VISIBLE_DEVICES']=gpus


        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(0)
            opt.device_str="cuda:0" if torch.cuda.is_available() else "cpu"
            opt.device = torch.device( opt.device_str ) 

        else:
            opt.device= torch.device('cpu')

        opt.gpu_ids=   [i  for i in range(len(opt.gpu_ids)) ]
        
        print('####################')
        print(str(opt.device))
        print('####################')

        ################dataset############################
        if hasattr(opt,'dataset_ids'):
            opt.dataset_ids.sort()

        if hasattr(opt,'source_dataset_ids'):
            opt.source_dataset_ids.sort()

        if hasattr(opt,'test_dataset_ids'):
            opt.test_dataset_ids.sort()

        if hasattr(opt,'test_source_dataset_ids'):
            opt.test_source_dataset_ids.sort()




        expr_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.train_step) 
        sutil.makedirs(expr_dir)        
        file_name = os.path.join(expr_dir, 'opt.txt')
        opt.expr_dir=expr_dir

        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(opt.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
    

        config_name = 'config_train' if self.isTrain  else 'config_test'


        config_dir=os.path.join(expr_dir, config_name)
        cnt=1
        while os.path.exists(config_dir):
            config_dir=os.path.join(expr_dir, config_name + str(cnt))
            cnt+=1

        sutil.makedirs(config_dir)


        for config_addr in config_addrs:
            shortname= os.path.split(config_addr)[-1]
            shutil.copy(config_addr,   os.path.join(config_dir,shortname)   ) # copy config file to output folder




        opt.debug_image_dir= os.path.join( expr_dir,'image')
        opt.debug_info_dir= os.path.join( expr_dir,'info')
        sutil.makedirs(opt.debug_image_dir)
        sutil.makedirs(opt.debug_info_dir)

        sutil.get_logger(path=os.path.join(opt.debug_info_dir,opt.name+'.log' ),name='main' )
        sutil.get_logger(path=os.path.join(opt.expr_dir ,'datasetids.log' ),name='datasetids' )



        opt.typ='sketch_photo'
        opt.domainX='sketch'    
        opt.domainY='photo' 
        if any ( x in  sutil.read_yaml('setting/datasetid.yaml')['RGBD'] for x in opt.dataset_ids ):
            opt.typ='RGBD'
            opt.domainX='RGB'
            opt.domainY='D'
        if any ( x in  sutil.read_yaml('setting/datasetid.yaml')['NIR_VIS'] for x in opt.dataset_ids ):
            opt.typ='NIR_VIS'
            opt.domainX='NIR'
            opt.domainY='RGB'

            
        def to255(x):

            return (x+1)/2.0*255.0


        def get_rcg_func(key): #recog_state_dict

            nonlocal opt

            rcg_func1=  sutil.pytorch_rgb2bgr  if opt[key]['rgb2bgr'] else   lambda x: x
            rcg_func2=to255 if opt[key]['to255'] else lambda x:x

            if 'sub' in opt[key]:
                ret=opt[key]['sub']

                if len(ret)==3:
                    sub=torch.cat([ ret[0]*torch.ones( 1,opt.img_size,opt.img_size  ) , \
                        ret[1]*torch.ones( 1,opt.img_size,opt.img_size  ) , \
                        ret[2]*torch.ones( 1,opt.img_size,opt.img_size  )  ],0   ).to(opt.device)
                elif len(ret)==1:
                    sub= ret[0]*torch.ones( 1,opt.img_size,opt.img_size  ).to(opt.device)

                rcg_func3= lambda x: x-sub
            else:
                rcg_func3= lambda x: x

            #return  lambda x:  rcg_func3( rcg_func2( rcg_func1(x)))

            def return_func_3dim(x):

                if x.size(1)==1:
                    x=x.repeat(1,3,1,1)

                return rcg_func3( rcg_func2( rcg_func1(x)))

            def return_func(x):

                return rcg_func3( rcg_func2( rcg_func1(x)))

            if opt[key]['input_dim']==3:
                return return_func_3dim
            elif opt[key]['input_dim']==1:
                return return_func


        if 'recog_state_dict_rgb_for_fusion' in opt:
            opt.rgb_rcg_func_rgb_for_fusion =get_rcg_func('recog_state_dict_rgb_for_fusion')


        if 'recog_state_dict' in opt:
            opt.rcg_func         =get_rcg_func('recog_state_dict')


        opt.transfer_tensor_rgb= transforms.Compose( [transforms.Resize([opt.img_size,opt.img_size])   ,transforms.ToTensor(),transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  ])
        opt.transfer_tensor_gray= transforms.Compose( [transforms.Resize([opt.img_size,opt.img_size])   ,transforms.ToTensor(),transforms.Normalize(mean=[0.5], std= [0.5]  )  ])


        if opt.read_mode=='PIL':

            opt.readimg_to_tensor_rgb= sutil.PIL_readimg_to_tensor(img_size=opt.img_size,read_dim=3)
            opt.readimg_to_tensor_gray=  sutil.PIL_readimg_to_tensor(img_size=opt.img_size,read_dim=1)
            

        else:

            raise(RuntimeError('no such read model'))


        if opt.isTrain and opt.continue_train and  isinstance(opt.which_epoch,int):
            opt.start_epoch=int(opt.which_epoch)


        self.opt = opt

        sutil.upload_opt(opt)


        assert('manualSeed' in opt)

        random.seed(opt.manualSeed)
        os.environ['PYTHONHASHSEED'] = str(opt.manualSeed)
        np.random.seed(opt.manualSeed)
        torch.manual_seed(opt.manualSeed)
        torch.cuda.manual_seed(opt.manualSeed)
        torch.cuda.manual_seed_all(opt.manualSeed)    
        torch.backends.cudnn.benchmark = False            # if benchmark=True, deterministic will be False
        torch.backends.cudnn.deterministic = True
        # torch.set_deterministic(True)  


        return self.opt



       