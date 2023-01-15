import torchvision.transforms as transforms
import numpy as np
import os
from torch.utils.data import Dataset
from .BaseDataset import BaseDataset
import random
from stools import sutil
import torch


class tsfUnpairedDataset(BaseDataset):
    """Face Landmarks dataset."""

    def __init__(self,dataroot,dataset_ids,source_dataset_ids,phase,img_size,opt ,aug, \
        no_target_imageYs=False, use_percent_dataset=False   ):

        super(tsfUnpairedDataset,self).__init__()

        
        self.check_dataset(dataset_ids=dataset_ids,source_dataset_ids=source_dataset_ids)
    

        self.phase=phase
        self.img_size=img_size
        
        self.aug=aug
    
        self.source_dataset_ids=source_dataset_ids
        self.dataset_ids=dataset_ids
        self.dataroot=dataroot
        self.opt=opt
        self.dim_X=opt.dim_X
        self.dim_Y=opt.dim_Y

        self.norm_tsX=self.normal_transfer(nchannel=self.dim_X)
        self.norm_tsY=self.normal_transfer(nchannel=self.dim_Y)
        self.aug_tsX=self.RS_transfer(self.img_size,self.img_size,nchannel=self.dim_X)
        self.aug_tsY=self.RS_transfer(self.img_size,self.img_size,nchannel=self.dim_Y)

        self.no_target_imageYs=no_target_imageYs


        if 'serial_probility' not in opt:
            self.serial_probility=0.25
        else:
            self.serial_probility= opt.serial_probility

        print(str(dataset_ids))
        print('serial_probility={}\n'.format(self.serial_probility) )
        sutil.log('self.serial_probility={}'.format(self.serial_probility), 'datasetids')


        self.source_imageYs=[]
        self.source_imageXs=[]
        self.target_imageXs=[]
        self.target_imageYs=[]


        self.use_percent_dataset =use_percent_dataset
        if use_percent_dataset:
            self.percent_of_dataset = opt.percent_of_dataset


        self.read_images_XandY()

        self.L=max(self.source_len,  self.target_len_imageX  )

    def read_images_XandY(self):

        opt=self.opt

        source_imageXs=[]
        source_imageYs=[]
        target_imageXs=[]
        target_imageYs=[]



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

        for x in self.source_dataset_ids:
            datasetname=self.dataset_map[x]

            for root,dirs,files in os.walk(os.path.join(self.dataroot,datasetname,self.phase,domainY),topdown=True):
                files.sort()
                for name in files:
                    fileloc=os.path.join(root,name)
                    
                    source_imageYs.append(fileloc)

                    self.get_dict_id(self.get_id_from_loc(name))

                    
        for x in self.source_dataset_ids:
            datasetname=self.dataset_map[x]
        
            for root,dirs,files in os.walk(os.path.join(self.dataroot,datasetname,self.phase,domainX),topdown=True):                
                files.sort()
                for name in files:
                    fileloc=os.path.join(root,name)

                    source_imageXs.append(fileloc)



        self.opt.source_identities = len(self.id_set)


        for x in self.dataset_ids:
            datasetname=self.dataset_map[x]
        
            for root,dirs,files in os.walk(os.path.join(self.dataroot,datasetname,self.phase,domainX),topdown=True):
                files.sort()
                for name in files:
                    fileloc=os.path.join(root,name)
                    
                    target_imageXs.append(fileloc)



        if not self.no_target_imageYs:
            for x in self.dataset_ids:
                datasetname=self.dataset_map[x]
            
                for root,dirs,files in os.walk(os.path.join(self.dataroot,datasetname,self.phase,domainY),topdown=True):
                    files.sort()
                    for name in files:
                        fileloc=os.path.join(root,name)

                        target_imageYs.append(fileloc)





        all_ids=set()
            
        for fileloc in target_imageXs:
            tmp_id = self.get_id_from_loc(fileloc)
            all_ids.add(tmp_id)

        all_ids=   list(all_ids)
        all_ids.sort()


        new_target_imageXs = []


        use_percent_dataset=self.use_percent_dataset
        
        if use_percent_dataset:

            percent_of_dataset=self.percent_of_dataset

            used_ids=[]


            if 'dataset_org_ids_txt' in opt and opt['dataset_org_ids_txt'] is not None:
                txtloc = opt['dataset_org_ids_txt']
            else:
                txtloc = os.path.join(self.opt.expr_dir, 'dataset_org_ids.txt') 



            if os.path.exists(txtloc):

                used_ids = self.read_ids(txtloc)

            else:
          

                n_used_ids=  int( percent_of_dataset* len(all_ids) + 0.5 ) 

                np.random.shuffle(all_ids)

                used_ids = all_ids[:n_used_ids]


                self.write_ids(used_ids,  txtloc)

        else:

            used_ids = all_ids



        for target_imageX in  target_imageXs:


            tmp_id = int(os.path.split(target_imageX)[-1].split('.')[0].split('_')[0])

            if tmp_id in used_ids:

                new_target_imageXs.append(target_imageX)

                self.get_dict_id(tmp_id)


        target_imageXs= new_target_imageXs


        if not self.no_target_imageYs:

            new_target_imageYs = []

            for target_imageY in  target_imageYs:


                tmp_id = int(os.path.split(target_imageY)[-1].split('.')[0].split('_')[0])

                if tmp_id in used_ids:

                    new_target_imageYs.append(target_imageY)



            target_imageYs= new_target_imageYs




        source_imageXs.sort()
        source_imageYs.sort()
        target_imageXs.sort()
        target_imageYs.sort()


        self.source_imageXs = source_imageXs

        self.source_imageYs = source_imageYs

        self.target_imageXs = target_imageXs

        self.target_imageYs = target_imageYs

        
        self.source_len=len(self.source_imageYs)                
        self.target_len_imageX=len(self.target_imageXs)
        self.numbers= len(self.id_set)


        self.opt.target_identities = len(self.id_set) -  self.opt.source_identities


        assert len(source_imageYs)==len(source_imageXs)

        if self.no_target_imageYs:
            assert(len(target_imageYs)==0)




    def read_ids(self,txtloc):

        used_ids=[]

        with open(txtloc,'r') as fr:
            lines=fr.readlines()

            firstline=lines[0] # 'contain %d ids:'

            nids = int(firstline.split()[1])

            for line in lines[1:]:
                used_ids.append( int(line.strip()) )

        fr.close()

        assert(len(used_ids) == nids)

        return used_ids

    def write_ids(self, used_ids,  txtloc):

        nids= len(used_ids)

        with open(txtloc,'w') as fw:

            fw.write('contain %d ids:\n'%nids)

            for tid in used_ids:

                fw.write('%d\n'%tid)

        fw.close()




    def __len__(self):

        return   self.L

    def num_identities(self):
        
        return self.numbers


    def no_match_random(self):

        self.serial_probility = -10000;


    def __getitem__(self, idx):

        source_idx=  idx %self.source_len

        p=random.uniform(0, 1)

        target_idx=  idx %self.target_len_imageX  if p<self.serial_probility else  np.random.randint(0, self.target_len_imageX)

                      
        
        source_imageY = self.default_loader(self.source_imageYs[source_idx],self.dim_Y)
        source_imageX = self.default_loader(self.source_imageXs[source_idx],self.dim_X)
        target_imageX = self.default_loader(self.target_imageXs[target_idx],self.dim_X)

        if not self.no_target_imageYs:
            target_imageY = self.default_loader(self.target_imageYs[target_idx],self.dim_Y)
        else:
            target_imageY=None

        
        if self.phase=='train' and self.aug:
            
            source_imageX,source_imageY= self.aug_XandY(source_imageX,source_imageY)
            target_imageX,target_imageY= self.aug_XandY(target_imageX,target_imageY)


        source_imageX= self.norm_tsX(source_imageX)
        source_imageY= self.norm_tsY(source_imageY)

        target_imageX= self.norm_tsX(target_imageX)

        if not self.no_target_imageYs:
            target_imageY= self.norm_tsY(target_imageY)

  

        source_id =   self.id_set[ self.get_id_from_loc(self.source_imageYs[source_idx])  ]
        target_id =   self.id_set[ self.get_id_from_loc(self.target_imageXs[target_idx]) ]


        sample = {'target_imageX': target_imageX ,
        'source_imageY': source_imageY, 'source_imageX': source_imageX, 
        'target_imageX_path':self.target_imageXs[target_idx],        
        'source_imageX_path':self.source_imageXs[source_idx],
        'source_imageY_path':self.source_imageYs[source_idx],
        'source_imageX_id':source_id,'source_imageY_id':source_id, 'target_imageX_id': target_id
         }



        if not self.no_target_imageYs:

            sample['target_imageY']=target_imageY
            sample['target_imageY_path']=self.target_imageYs[target_idx]
            sample['target_imageY_id']=target_id



        return sample




    def get_name2pos_pos2class(self):

        name2pos={}           #   mapping filenames (no extension) to the indexes of total list
        pos2class=[]          #   mapping indexes of total list to ids contained in files 

        pos=0
        for pathx in self.source_imageXs:
            name=sutil.get_file_name(pathx) #filename (no extension)
            name2pos[name]= pos # mapping the filename (no extension) to the index of total list
            pos2class.append(   self.get_id_from_loc(pathx) ) # mapping the index of total list to id contained in the file
            pos+=1


        for pathx in self.target_imageXs:
            name=sutil.get_file_name(pathx)
            name2pos[name]= pos
            pos2class.append(   self.get_id_from_loc(pathx) )
            pos+=1

        pos2class = np.array(pos2class)

        return name2pos, pos2class






    def prepare_id_images(self):

        '''
        target_id2images,source_id2images    
        
        target_id2images:

        mapping id in id_set to a list of indexes of target_imageXs

        source_id2images:

        mapping id in id_set to a list of indexes of source_id2images

        '''


        target_id2images={}

        target_imageXs=self.target_imageXs
        target_imageYs=self.target_imageYs

        for pos,image_loc in enumerate(target_imageXs):

            tid = self.id_set[ self.get_id_from_loc(image_loc) ]
            if tid not in target_id2images:
                target_id2images[tid]=[pos]
            else:
                target_id2images[tid].append(pos)

        self.target_id2images=target_id2images



        source_id2images={}

        source_imageXs=self.source_imageXs
        source_imageYs=self.source_imageYs

        for pos,image_loc in enumerate(source_imageXs):

            tid = self.id_set[ self.get_id_from_loc(image_loc) ]
            if  tid not in source_id2images:
                source_id2images[tid]=[pos]
            else:
                source_id2images[tid].append(pos)

        self.source_id2images=source_id2images


    

    def img_aug(self,img):

        if self.phase=='train' and self.aug:

            pflip, pcrop  = random.uniform(0, 1),random.uniform(0, 1)

            if pflip > 0.5:
                img=transforms.functional.hflip(img)

            if pcrop>0.5:

                starth= random.randint(0,30)
                startw= random.randint(0,30)
                img=transforms.functional.resize(img, self.img_size+30)
                img=transforms.functional.crop(img,starth,startw, self.img_size, self.img_size)
                
        return img



    def get_samples_from_loc(self,sample ):

        sample = self.default_loader(sample,self.dim_X)

        sample = self.img_aug(sample)
            
        sample= self.norm_tsX(sample)

        sample=sample.unsqueeze(0)

        return sample    





    def prepare_testdata(self, batchsize):
        
        idxs= [i for i in range(self.L)] 

        np.random.shuffle(idxs)

        idxs=idxs[:batchsize]


        dictions=[]

        for idx in idxs:

            dictions.append( self.__getitem__(idx))  


        sample={}

        for key in dictions[0].keys():

            val=[]

            ret=dictions[0][key]

            if isinstance(ret,int):
                for dict_x in dictions: 
                    val.append( torch.Tensor( [dict_x[key]]).long()    )
                val=torch.cat(val,dim=0)
            elif isinstance(ret,torch.Tensor):
                for dict_x in dictions: 
                    val.append(  dict_x[key] ) 
                val=torch.stack(val,dim=0)
            else:
                for dict_x in dictions: 
                    val.append(  dict_x[key] ) 
            sample[key]=val




        return sample