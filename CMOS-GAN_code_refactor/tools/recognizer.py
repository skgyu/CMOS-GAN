#coding: utf-8
#  cross modality cal CMC curves
import os,torch,networks
import numpy as NP
from tqdm import tqdm
from torch.autograd import Variable
from stools import sutil
from sync_batchnorm import  convert_model
from tools.functions import *
from torch.nn import DataParallel as DPL
from tools.struct import *

def cos_sim(x,y):

    return NP.dot(x,y)/NP.linalg.norm(x)/NP.linalg.norm(y)


###modified on 12_31
class Recognizer():

    def  __init__(self,opt):

        self.feature_extraction_model=None
        # TS_resize224= transforms.Compose( [transforms.Resize([224,224])   ,transforms.ToTensor(),transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  ])
        self.typ=opt.typ
        self.save_img_feature=opt.save_img_feature

        self.nobackgrounds=False if self.typ=='sketch_photo'  else True

        
        if 'nobackgrounds' in opt and opt['nobackgrounds']: 
            self.nobackgrounds=True


        # if self.typ=='sketch_photo':
        if not self.nobackgrounds:

            self.background_dir='../dataset/additional/10000backgrounds' 
            # self.background_readed_dir='../dataset/additional/PCSO_py35tf12'
            self.backgrounds=[]

            background_list= os.listdir(self.background_dir)
            background_list.sort()

            for file in background_list:
                self.backgrounds.append( os.path.join(self.background_dir,file  )  )

                

            # if not os.path.exists(self.background_readed_dir):
            #     os.makedirs(self.background_readed_dir)

            # if not os.path.exists(self.background_readed_dir+'_gray'):
            #     os.makedirs(self.background_readed_dir+'_gray')


        self.domainX_dirs= None
        self.imageXs= []
        self.imageYs=[]

        self.np=0
        self.npr=0
        self.RGBD_number2id=dict()
        self.probe= []
        self.gallery= []
        self.name=opt.name


        self.domainX=opt.domainX
        self.domainY=opt.domainY
        self.read_mode=opt.read_mode


    def load_simmatrix(self,outpath):

        
        #item_names=['imageXs','imageYs','probe','gallery','RGBD_number2id','sim_matrix']
        item_names=['imageXs','imageYs','RGBD_number2id','sim_matrix']
        att2loc={}
        for item_name in item_names:
            att2loc[item_name]=  os.path.join( outpath, item_name+'.npy')


        #fp=open( os.path.join(self.outpath, 'rcg_rate.txt') ,'w')

        loadimageXs=  list(NP.load(att2loc['imageXs'],allow_pickle=True))
        assert(loadimageXs ==  self.imageXs)


        loadimageYs=  list(NP.load(att2loc['imageYs'],allow_pickle=True))
        assert(loadimageYs ==  self.imageYs)


        self.npr  =  len(self.imageXs)
        self.ng  =   len(self.imageYs)


        self.RGBD_number2id =NP.load(att2loc['RGBD_number2id'],allow_pickle=True).item()

        self.sim_matrix= NP.load(att2loc['sim_matrix'],allow_pickle=True)

        self.load_success=True

    def save_simmatrix(self,outpath):


        #item_names=['imageXs','imageYs','probe','gallery','RGBD_number2id','sim_matrix']
        item_names=['imageXs','imageYs','RGBD_number2id','sim_matrix']
        
        for item_name in item_names:
            saveloc=  os.path.join( outpath, item_name+'.npy')
            NP.save(saveloc, getattr(self,item_name))


    def replacemodel(self,model):

        self.feature_extraction_model=model



    def init(self,opt,recog_item,rcg_func,img_dim=None,num_identities=None):
        

        if num_identities is None:
            num_identities= opt.num_identities

        assert(num_identities is not None)

        state_dict=  torch.load( opt[recog_item].loc )


        if opt[recog_item]['type'] in ['resnet50','resnet50_depth']:
            self.feature_extraction_model = networks.get_ResNet50(input_dim=opt[recog_item]['input_dim'] if not img_dim else img_dim, num_classes=num_identities)
            self.feature_extraction_model = convert_model(self.feature_extraction_model)
            self.feature_extraction_model.apply( networks.weights_init(opt['finetune']['init']  )  )
        else:
            rasie(RuntimeError('no such model type'))

        ret = self.feature_extraction_model
        if 'part' in  opt[recog_item] and opt[recog_item]['part']:
            ret = getattr(ret,opt[recog_item]['part'])

            
        ret.load_state_dict(state_dict)
        
        self.feature_extraction_model.cuda()
        self.feature_extraction_model.eval()

        self.feature_extraction_model.rcg_func= rcg_func
        

    def get_sim_matrix(self):
    
        print('get_sim_matrix.........')

        if self.load_success:
            return self.sim_matrix

        sim_matrix=   NP.zeros((self.npr,self.ng)).astype(NP.float64)


        step=5000
        for le in tqdm(range(0,self.npr,step)):
            ri= min(le+step,self.npr)
            sim_matrix[le:ri,:]=  NP.matmul(self.probe[le:ri], self.gallery[:].T)  


        self.sim_matrix=sim_matrix

        if self.save_img_feature:
            self.save_simmatrix(self.outpath)


        return sim_matrix

        

    def sort_matrix(self,sim_matrix):
        print('sort_matrix........')

        return NP.argsort(-sim_matrix,axis=1)


    def cal_dect(self,sorted_index):
        print('cal_dect................')

        principle_index=NP.zeros((self.npr,self.ng)).astype(NP.int64)

        for i in tqdm(range(self.npr)):
            ret_id = self.RGBD_number2id[get_id_from_loc(self.imageXs[i]) ]
            principle_index[i:i+1]=  NP.full( (1,self.ng) ,ret_id,dtype=NP.int64)


        detect_matrix = (principle_index==sorted_index).astype(NP.int64)
        max_poses=   NP.argmax( detect_matrix, axis=1 )

        for i in tqdm(range(max_poses.shape[0])):
            which_pos=max_poses[i]
            detect_matrix[i,which_pos: ]  = NP.full( self.ng-which_pos,1, dtype=NP.int64)


        return detect_matrix


    def calc_CMC(self,detect_matrix):
        
        print('calc_CMC...................')
        rates=1.0*detect_matrix.mean(0)
 

        fp=open( os.path.join(self.outpath, 'rcg_rate.txt') ,'w')
        for rank in tqdm(range(0,min(200,self.ng))):
            fp.write('Rank %d = %.6f\n' %( rank+1 , rates[rank]  ) )
        fp.close()



    def read_gallery(self,dataroot,dataset_ids,gallery_type,rcg_model_name,read_imageYs=None, nobackgrounds=False
        , reverse_option=False,gallery_gray=False, use_record=True,equilibrium=False,gpu_ids=[0]):


        nobackgrounds=self.nobackgrounds
        
        self.dataroot=dataroot
        dataset_ids.sort()
        self.dataset_ids=dataset_ids
        self.load_success=False

        self.gallery_type  = gallery_type
        self.reverse_option=reverse_option
        self.gallery_gray=gallery_gray

        if self.typ=='sketch_photo':
            rcg_model_name=  'modality='  + ('photo_' if not reverse_option else 'sketch_') + self.read_mode +'_'+rcg_model_name        
        elif self.typ=='NIR_VIS':
            rcg_model_name=  'modality='  + ('RGB_' if not reverse_option else 'NIR_') + self.read_mode +'_'+rcg_model_name
        elif self.typ=='RGBD':
            rcg_model_name=  'modality='  + ('D_' if not reverse_option else 'RGB_') + self.read_mode +'_'+rcg_model_name
        else:
            print('self.typ={}'.format(self.typ))
            raise(RuntimeError('read_gallery unknown type')  )
        self.rcg_model_name = rcg_model_name


        self.feature_extraction_model.eval()

        print('init...........')

        dataset_map=sutil.read_yaml('setting/datasetid.yaml')

        
        print(dataset_ids)


        if self.typ=='sketch_photo' :
            # search_name =  'sketch' if    reverse_option else 'photo'
            search_name =  self.domainX  if    reverse_option else self.domainY
            self.imageYs=[]
            for x in  dataset_ids:
                datasetname=dataset_map[x]

                for root,dirs,files in os.walk(os.path.join( dataroot,datasetname,'test',search_name),topdown=True):
                    files.sort()
                    for name in files:


                        fileloc=os.path.join(root,name)
                        self.imageYs.append(fileloc)

            if not nobackgrounds:  
                for i in tqdm(range(len(self.backgrounds)) ):
                    self.imageYs.append(self.backgrounds[i])
                    
            

            self.imageYs.sort(key=sort_by_name)

            self.gallery=[]
            self.ng=0

            for i in tqdm(range(len(self.imageYs)) ):

                ret_info = read_info ( self.imageYs[i],self.feature_extraction_model,rcg_model_name=self.rcg_model_name, isgray=gallery_gray, save_img_feature=self.save_img_feature )
                self.gallery.append(ret_info)
                self.RGBD_number2id[get_id_from_loc(self.imageYs[i])]=self.ng
                self.ng+=1 



            self.gallery=NP.array(self.gallery).astype(NP.float64)
                  
                    
        elif self.typ=='RGBD'  or self.typ=='NIR_VIS' :

            # search_name =  'RGB' if    reverse_option else 'D'
            search_name =  self.domainX  if    reverse_option else self.domainY


            if read_imageYs is not None and len(read_imageYs)!=0:
                self.imageYs=read_imageYs

            else:
                self.imageYs=[]
                for x in dataset_ids:
                    datasetname=dataset_map[x]
                    gallery_list_loc=   dataset_map['gallery'][x][search_name]
                    f=open(gallery_list_loc,'r')
                    for line in f.readlines():
                        loc=line.strip().split(' ')[0]
                        self.imageYs.append( os.path.join( dataroot,datasetname ,loc)  )
                    f.close()


            if not nobackgrounds:  
                for i in tqdm(range(len(self.backgrounds)) ):
                    self.imageYs.append(self.backgrounds[i])


            self.imageYs.sort(key=sort_by_name)

            self.gallery=[]

            self.ng=0



            for i in tqdm(range(len(self.imageYs)) ):

                ret_info = read_info ( self.imageYs[i],self.feature_extraction_model,rcg_model_name=self.rcg_model_name, isgray=gallery_gray, save_img_feature=self.save_img_feature )
                self.gallery.append(ret_info)
                self.RGBD_number2id[get_id_from_loc(self.imageYs[i])]=self.ng
                self.ng+=1 

            self.gallery=NP.array(self.gallery).astype(NP.float64)

            try:
                assert(self.gallery.shape[0]==len(self.imageYs))

            except:    
                raise(RuntimeError('self.gallery.shape[0] = {} , len(self.imageYs)= {}'
                    .format(self.gallery.shape[0], len(self.imageYs) ) ) )
        else:

            raise(RuntimeError('read_gallery unknown type'))





    def  get_imageXs(self,probe_gray=False,batchsize=1,gpu_ids=[0]):


        print('get_imageXs')




        if self.typ  ==  'sketch_photo':

            self.imageXs=[]


            for domainX_dir in self.domainX_dirs:
                ret_imageXs=os.listdir(domainX_dir)
                for ret_image_x in ret_imageXs:

                    self.imageXs.append( os.path.join( domainX_dir,ret_image_x) )
            
            self.imageXs.sort(key=sort_by_name)

            print(len(self.imageXs))
            print(len(self.imageYs))

            
            # if not self.nobackgrounds:
            #     assert len(self.imageXs)+10000==len(self.imageYs),'len(imageXs)!=len(imageYs)'
            # else:
            #     assert len(self.imageXs) == len(self.imageYs),'len(imageXs)!=len(imageYs)'
            


        elif self.typ in ['RGBD','NIR_VIS']:

            
            self.imageXs=[]

            for domainX_dir in self.domainX_dirs:
                ret_imageXs=os.listdir(domainX_dir)

                for ret_image_x in ret_imageXs:
                    self.imageXs.append(os.path.join( domainX_dir,ret_image_x))
        
            self.imageXs.sort(key=sort_by_name)


        else:
            raise(RuntimeError("get_imageXs unknown type"))


        self.load_success=False
        if os.path.exists(os.path.join(self.outpath, 'sim_matrix.npy' )  ):
            self.load_simmatrix(self.outpath)
            return
                

        self.probe=[]  ############# editing ############################


            
            
        # fp=open( os.path.join(self.outpath, 'RGBD_number2id.txt') ,'w')
        # for key,val in self.RGBD_number2id.items():
        #     fp.write('RGBD_number2id[{}]={}\n'.format(key,val))
        # fp.close()


        Nimages=len(self.imageXs)
        if batchsize==1:
            for i in tqdm(range(Nimages)):

                ret_info = read_info ( self.imageXs[i],self.feature_extraction_model,rcg_model_name=self.rcg_model_name, isgray=probe_gray, save_img_feature=self.save_img_feature )
                self.probe.append(ret_info)

        else:

            for le in tqdm(range(0,Nimages,batchsize)):
                ri= min(le+batchsize,Nimages)

                need_extract=False

                for i in range(le,ri): 
                    saveloc=pre_loc(self.imageXs[i],rcg_model_name=self.rcg_model_name, isgray=probe_gray)
                    if os.path.exists(saveloc):
                        pass
                    else:
                        need_extract=True
                        paf=get_parent_abs_folder(self.imageXs[i],rcg_model_name=self.rcg_model_name, isgray=probe_gray)
                        sutil.makedirs(paf)
                        
                        break



                if not need_extract:
                    for i in range(le,ri): 
                        saveloc=pre_loc(self.imageXs[i],rcg_model_name=self.rcg_model_name, isgray=probe_gray)
                        ret_info=NP.load(saveloc,allow_pickle=True)
                        self.probe.append(ret_info)

                else:  # if need extract

                    batch_tensor=[]

                    for i in range(le,ri): 


                        img =    sutil.readimg_to_tensor_rgb(self.imageXs[i]) if not probe_gray else  sutil.readimg_to_tensor_gray(self.imageXs[i])
                        img =  img.cuda()

                        img= Variable(img.unsqueeze(0))

                        # img=self.feature_extraction_model.rcg_func(img)

                        # (r, g, b) = torch.chunk(img, 3, dim = 0)
                        # img = torch.cat((b, g, r), dim = 0).unsqueeze(0) # convert RGB to BGR 
                        batch_tensor.append(img)

                    batch_tensor =  torch.cat(batch_tensor,dim=0)

                    batch_tensor=self.feature_extraction_model.rcg_func(batch_tensor)


                    with torch.no_grad():
                        fc =DPL(self.feature_extraction_model,device_ids=gpu_ids)(batch_tensor)[0][0] 
                        fc=torch.nn.functional.normalize(fc, p=2, dim=1, eps=1e-12)
                        
                    for i in range(le,ri):



                        ret_info = fc[i-le].detach().cpu().float().numpy()
                        self.probe.append(ret_info)

                        saveloc=pre_loc(self.imageXs[i],rcg_model_name=self.rcg_model_name, isgray=probe_gray)

                        if self.save_img_feature:
                            NP.save( saveloc, ret_info)



            
        self.probe=NP.array(self.probe).astype(NP.float64)
    
        self.npr=Nimages

        try:
            assert(self.probe.shape[0]==len(self.imageXs) )
        except:
            raise(RuntimeError('self.probe.shape[0] = {} , len(self.imageXs)= {}'
                .format(self.probe.shape[0], len(self.imageXs) ) ))









    def run(self,train_step,probe_type, epochL=-1,epochR=-1, \
        add=1,probe_gray=False,load_epoch=None,\
        original_data=[],batchsize=1,gpu_ids=[0],save_dir_name=''):

        
        nobackgrounds=self.nobackgrounds
        
        self.probe_type = probe_type
        rcg_model_name  = self.rcg_model_name
        gallery_gray=self.gallery_gray


        name=self.name
        
        self.feature_extraction_model.eval()

        best_epoch,best_rate=-1,-1
        rcg_rates_at_rank1=[]
        rcg_epochs=[]

        print(epochL,epochR)

        for epoch in range(epochL,epochR+1,add):

            # domainX_dir= 'results/'+name +'/test_'+str(epoch)+'/images'


            if len(original_data)!=0:
                original_data.sort()
                self.domainX_dirs= original_data
            else:
                if not self.reverse_option:
                    self.domainX_dirs= [os.path.join('results',name,  train_step ,'test_'+str(load_epoch if load_epoch else epoch ),'images')  ]
                else: #reverse 


                    if  self.typ=='sketch_photo':
                        modality=   'sketch'
                        raise(RuntimeError('disable reverse option for sketch-to-photo')  )

                    if  self.typ=='NIR_VIS':
                        modality=   'NIR'
                        raise(RuntimeError('disable reverse option for NIR-to-VIS')  )

                    elif self.typ=='RGBD':
                        modality=  'RGB'
                        dataset_map=sutil.read_yaml('setting/datasetid.yaml')
                        self.domainX_dirs=[]
                        for x in self.dataset_ids:
                            datasetname=dataset_map[x]
                            self.domainX_dirs+=  [os.path.join( self.dataroot,datasetname,'test',modality)]
                    else:
                        raise(RuntimeError('unknown type')  )

            skip=False
            for domainX_dir in self.domainX_dirs:
                if not os.path.exists(domainX_dir):
                    skip=True
                    break

            if skip:
                continue

            # self.outpath= 'rcgrate/'+str(name)+('noback' if  nobackgrounds else 'withback')+'/test_'+str(epoch)
            

            HOME=  os.path.join('recognition rate',name, train_step, self.probe_type+"_"+self.gallery_type, "probe_{}_gallery_{}_{}".format( 'gray' if probe_gray else 'color'  \
                , 'gray' if gallery_gray else 'color', save_dir_name ), ('noback' if  nobackgrounds else 'withback')  ) 

            self.outpath=  os.path.join(HOME,'test_'+str(epoch)  ) 
       
            if not os.path.exists(self.outpath):
                os.makedirs(self.outpath)


            #elif os.path.exists(os.path.join( self.outpath,'rcg_rate.txt' )  ):
            if epoch!=epochR:
                assert(   os.path.exists(os.path.join(self.outpath, 'rcg_rate.txt'))   )
            else:
                self.get_imageXs(probe_gray=probe_gray,batchsize=batchsize,gpu_ids=gpu_ids)

            
            


            if  os.path.exists(os.path.join( self.outpath,'rcg_rate.txt' )  ):
                fp=open( os.path.join(self.outpath, 'rcg_rate.txt') ,'r')

                ret_rate=float(fp.readline().strip('\n').split()[3])

                rcg_rates_at_rank1.append(ret_rate)
                rcg_epochs.append(epoch)

                fp.close()


                file_loc=os.path.join(self.outpath, 'rank_1_detected_name.txt')

                    
                continue

            
            


            fp=open(os.path.join(self.outpath,'g_order.txt'),'w' )
            for x in self.imageYs:
                fp.write(x+'\n')

            # if hasattr(self, 'backgrounds'):
            #     for x in self.backgrounds:
            #         fp.write(x+'\n')
            fp.close()


            fp=open(os.path.join(self.outpath,'p_order.txt') ,'w')
            for x in self.imageXs:
                fp.write(x+'\n')
            fp.close()



            sim_matrix=self.get_sim_matrix()
            
            sorted_index=self.sort_matrix(sim_matrix)
            
            detect_matrix=self.cal_dect(sorted_index)
            self.calc_CMC(detect_matrix)



            fp=open( os.path.join(self.outpath, 'rcg_rate.txt') ,'r')

            ret_rate=float(fp.readline().strip('\n').split()[3])
            rcg_rates_at_rank1.append(ret_rate)
            rcg_epochs.append(epoch)

            fp.close()  ###to be continue

        rcg_rates_at_rank1=NP.array(rcg_rates_at_rank1)

        best_rate= NP.max(rcg_rates_at_rank1)
        best_epoch=  rcg_epochs[NP.argmax(rcg_rates_at_rank1)] 

        worst_rate= NP.min(rcg_rates_at_rank1)
        worst_epoch=  rcg_epochs[NP.argmin(rcg_rates_at_rank1)] 

        mean_rate=  rcg_rates_at_rank1.mean()
        std_rate=  rcg_rates_at_rank1.std()

        fp=open(os.path.join(HOME,'best_rcg_rate.txt'),'w')

        
        fp.write('best_epoch: '+str(best_epoch)+' best_rate: '+str(best_rate)+'\n')
        fp.write('worst_epoch: '+str(worst_epoch)+' worst_rate: '+str(worst_rate)+'\n')
        fp.write('mean_rate: '+str(mean_rate)+'\n')
        fp.write('std_rate: '+str(std_rate)+'\n')

        for i in range(len(rcg_epochs) ):
            fp.write('epoch '+str(rcg_epochs[i])+' rcg_rate_at_rank_1= '+str(rcg_rates_at_rank1[i])+' \n')
        fp.close()


        fp=open(os.path.join(HOME,'ave_rcg_rate.txt'),'w')
        for i in range(len(rcg_epochs) ):
            fp.write('epoch '+str(rcg_epochs[i])+' ave_rcg_rate_at_rank_1= '+str( NP.mean(rcg_rates_at_rank1[:i]) )+' \n')
        fp.close()

 