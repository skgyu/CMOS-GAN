#coding: utf-8
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
from PIL import Image
from stools import sutil
import random



class BaseDataset(Dataset):


    def __init__(self ):

        self.dataset_map=sutil.read_yaml('setting/datasetid.yaml')

    
    def get_dict_id(self,x):

        if x not in self.id_set:
            num=len(self.id_set)
            self.id_set[x]=num
            
            sutil.log( 'self.__class__.__name__={}'.format(self.__class__.__name__),'datasetids')
            sutil.log( 'self.id_set[{}] = {}'.format(x,self.id_set[x]),'datasetids')

        return self.id_set[x]

    def normal_transfer(self, nchannel):

        return  transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean= (0.5, 0.5, 0.5) if nchannel==3 else [0.5], std=(0.5, 0.5, 0.5) if nchannel==3 else [0.5])
        ])



    def RS_transfer(self,img_size_H,img_size_W,nchannel):

        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((img_size_H + 30, img_size_W+30)),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5) if nchannel==3 else [0.5], std=(0.5, 0.5, 0.5) if nchannel==3 else [0.5])
        ])

    def pil_loader(self,path,read_dim):

        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB') if read_dim==3 else img.convert('L')  


    def default_loader(self,path, read_dim):
        
        return self.pil_loader(path,read_dim)



    
    def init_type(self):

        self.RGBD=False
        self.sketch_photo=False
        self.NIR_VIS=False


    def check_dataset(self,dataset_ids,source_dataset_ids=[]):

        self.init_type()

        ans=0

        for dataset in dataset_ids+source_dataset_ids:
            if dataset in self.dataset_map['RGBD']:
                self.RGBD=True
                ans|= 1<<0
            if dataset in self.dataset_map['sketch_photo']:
                self.sketch_photo=True
                ans|= 1<<1
            if dataset in self.dataset_map['NIR_VIS']:
                self.NIR_VIS=True
                ans|= 1<<2

        ran=[]

        for i in range(3):
            ran.append(1<<i)
        
        assert( ans in  ran  )

    def get_id_from_loc(self,locstr):
    
        return int(os.path.split(locstr)[1].split('.')[0].split('_')[0] )


    def aug_XandY(self,img_x,img_y=None):

        pflip, pcrop  = random.uniform(0, 1),random.uniform(0, 1)

        if pflip>0.5:

            img_x=transforms.functional.hflip(img_x)

            if img_y is not None:
                img_y=transforms.functional.hflip(img_y)


        if pcrop>0.5:

            img_x=transforms.functional.resize(img_x, self.img_size+30)

            if img_y is not None:
                img_y=transforms.functional.resize(img_y, self.img_size+30)

            starth= random.randint(0,30)
            startw= random.randint(0,30)


            img_x=transforms.functional.crop(img_x,starth,startw, self.img_size, self.img_size)
            if img_y is not None:
                img_y=transforms.functional.crop(img_y,starth,startw, self.img_size, self.img_size)


        return img_x,img_y



