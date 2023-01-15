import os
from .BaseDataset import BaseDataset
import torch



class PairedDataset(BaseDataset):

    def __init__(self,dataroot,dataset_ids,phase,img_size,opt,aug ,pre=''):

        super(PairedDataset,self).__init__()

        self.check_dataset(dataset_ids=dataset_ids)
        
        self.dim_X=opt.dim_X
        self.dim_Y=opt.dim_Y

        self.img_size=img_size

        if opt.read_mode=='PIL':

            self.imageX_loader= self.default_loader 
            self.imageY_loader= self.default_loader 

            self.norm_tsX=self.normal_transfer(nchannel=self.dim_X)
            self.norm_tsY=self.normal_transfer(nchannel=self.dim_Y)

        else:

            raise(RuntimeError('no such read model'))


        self.phase=phase

        self.aug=aug


        self.pre=pre
        self.dataset_ids=dataset_ids
        self.dataroot=dataroot
        
        self.no_target_imageYs=False

        print(str(dataset_ids))

        self.imageYs=[]
        self.imageXs=[]

        self.read_images_XandY(self.imageXs,self.imageYs)


    def read_images_XandY(self,imageXs,imageYs):

        if self.sketch_photo:
            domainX='sketch'    
            domainY='photo'    
        elif self.RGBD:
            domainX='RGB'
            domainY='D'
        elif self.NIR_VIS:
            domainX='NIR'
            domainY='RGB'

        self.id_set=dict()


        self.imageXs=[]

        for x in self.dataset_ids:
            datasetname=self.dataset_map[x]

            for root,dirs,files in os.walk(os.path.join(self.dataroot,datasetname,self.phase,domainX),topdown=True):
                files.sort()
                for name in files:
                    fileloc=os.path.join(root,name)

                    self.imageXs.append(fileloc)
                    
                    self.get_dict_id(self.get_id_from_loc(name))
                
        self.imageYs=[]

        for x in self.dataset_ids:
            datasetname=self.dataset_map[x]
        
            for root,dirs,files in os.walk(os.path.join(self.dataroot,datasetname,self.phase,domainY),topdown=True):
                files.sort()
                for name in files:
                    fileloc=os.path.join(root,name)
                    self.imageYs.append(fileloc)

        # assert len(imageXs)==len(imageYs)

        self.len_imageY=len(self.imageYs)
        self.len_imageX=len(self.imageXs)

        self.numbers= len(self.id_set)
        self.L=max(self.len_imageY,  self.len_imageX  )

    def __len__(self):
        return self.L

    def num_identities(self):
        return self.numbers

    def __getitem__(self, idx):
        

        idx_imageX=idx%self.len_imageX

        imageX=self.imageX_loader(self.imageXs[idx_imageX],self.dim_X) 

        if not self.no_target_imageYs:
            idx_imageY=idx%self.len_imageY
            imageY=self.imageY_loader(self.imageYs[idx_imageY],self.dim_Y)  
        else:
            imageY=None

        if self.phase=='train' and self.aug:
            imageX,imageY = self.aug_XandY(imageX, imageY)

        imageX = self.norm_tsX(imageX)  #resize and to tensor [-1,1]

        imageX=  imageX.float()  #dtype = torch.float32

        imageX_id =   self.id_set[ self.get_id_from_loc(self.imageXs[idx_imageX])  ]

        if not self.no_target_imageYs:
            imageY = self.norm_tsY(imageY)
            imageY=  imageY.float()
            imageY_id =   self.id_set[ self.get_id_from_loc(self.imageYs[idx_imageY]) ]


        sample = {self.pre+'imageX': imageX ,
            self.pre+'imageX_path':self.imageXs[idx_imageX],
            self.pre+'imageX_id':imageX_id
             }


        if not self.no_target_imageYs:
            sample[self.pre+'imageY']= imageY
            sample[self.pre+'imageY_path']= self.imageYs[idx_imageY]
            sample[self.pre+'imageY_id']= imageY_id


        return sample





