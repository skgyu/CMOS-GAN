#coding: utf-8
#  cross modality cal CMC curves
import os
import numpy as NP
from tqdm import tqdm
from tools.functions import *
from tools.struct import *

def cos_sim(x,y):


    return NP.dot(x,y)/NP.linalg.norm(x)/NP.linalg.norm(y)



class FusionRecognizer():

    def  __init__(self,recognizer_d,recognizer_rgb):

        self.recognizer_d = recognizer_d
        self.recognizer_rgb = recognizer_rgb


    def run(self,train_step, epochL=-1,epochR=-1,
        add=1,load_epoch=None,
        original_data=[],batchsize=1,gpu_ids=[0],save_dir_name=''):

        nobackgrounds=self.recognizer_d.nobackgrounds
        assert(save_dir_name!='')


        self.gallery_type= self.recognizer_d.gallery_type+'_and_'+self.recognizer_rgb.gallery_type
        self.probe_type = self.recognizer_d.probe_type+'_and_'+self.recognizer_rgb.probe_type


        assert(self.recognizer_d.RGBD_number2id==self.recognizer_rgb.RGBD_number2id)
        self.RGBD_number2id= self.recognizer_d.RGBD_number2id

        assert(self.recognizer_d.ng==self.recognizer_rgb.ng)
        self.ng= self.recognizer_d.ng

        assert(self.recognizer_d.npr==self.recognizer_rgb.npr)
        self.npr= self.recognizer_d.npr


        name=self.recognizer_d.name

    
        best_epoch,best_rate=-1,-1
        rcg_rates_at_rank1=[]
        rcg_epochs=[]

        print(epochL,epochR)

        for epoch in range(epochL,epochR+1,add):

            if len(original_data)!=0:
                original_data.sort()
                self.domainX_dirs= original_data
            else:
                ret_addr= os.path.join('results',name,  train_step ,'test_'+str(load_epoch if load_epoch else epoch ),'images') 
                self.domainX_dirs= [ ret_addr ]
          

            for domainX_dir in self.domainX_dirs:
                if not os.path.exists(domainX_dir):
                    raise(RuntimeError('no domainX_dir {}'.format(domainX_dir)))


            # self.outpath= 'rcgrate/'+str(name)+('noback' if  nobackgrounds else 'withback')+'/test_'+str(epoch)
            

            HOME=  os.path.join('recognition rate',name, train_step, 'fusion_'+self.probe_type+"_and_"+self.gallery_type, 
                save_dir_name, ('noback' if  nobackgrounds else 'withback')  ) 

            self.outpath=  os.path.join(HOME,'test_'+str(epoch)  ) 
       
            if not os.path.exists(self.outpath):
                os.makedirs(self.outpath)

            if epoch!=epochR:

                assert(os.path.exists(os.path.join( self.outpath,'rcg_rate.txt' )  ) )

            if  os.path.exists(os.path.join( self.outpath,'rcg_rate.txt' )  ):

                fp=open( os.path.join(self.outpath, 'rcg_rate.txt') ,'r')

                ret_rate=float(fp.readline().strip('\n').split()[3])

                rcg_rates_at_rank1.append(ret_rate)
                rcg_epochs.append(epoch)

                fp.close()

                continue





            sim_matrix1=self.recognizer_d.get_sim_matrix()
            sim_matrix2=self.recognizer_rgb.get_sim_matrix()
            sim_matrix=(sim_matrix1+sim_matrix2)/2.0


            sorted_index=self.sort_matrix(sim_matrix)      ###############
            
            detect_matrix=self.cal_dect(sorted_index)       ###############
            self.calc_CMC(detect_matrix)  ##################



            fp=open( os.path.join(self.outpath, 'rcg_rate.txt') ,'r')

            ret_rate=float(fp.readline().strip('\n').split()[3])

            rcg_rates_at_rank1.append(ret_rate)
            rcg_epochs.append(epoch)

            fp.close()

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
            fp.write('epoch '+str(rcg_epochs[i])+' ave_rcg_rate_at_rank_1= '+str( NP.mean(rcg_rates_at_rank1[:i+1]) )+' \n')

        fp.close()    



    def sort_matrix(self,sim_matrix):

        print('sort_matrix........')


        return NP.argsort(-sim_matrix,axis=1)



    def cal_dect(self,sorted_index):

        print('cal_dect................')



        principle_index=NP.zeros((self.npr,self.ng)).astype(NP.int64)

        for i in tqdm(range(self.npr)):
            ret_id = self.RGBD_number2id[get_id_from_loc(self.recognizer_d.imageXs[i]) ]
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