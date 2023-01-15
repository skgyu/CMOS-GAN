#coding: utf-8
#  cross modality cal CMC curves
import os,copy
import numpy as NP
from tqdm import tqdm
from torch.autograd import Variable
from stools import sutil
from tools.functions import *
from  matplotlib import pyplot as plt
plt.switch_backend('agg')
import shutil
from collections import  Counter, OrderedDict
from pytorch_msssim import ssim as func_cal_ssim



class MAEcaler():

    def  __init__(self):

        self.domainX_dirs= None
        self.imageXs= []
        self.imageYs=[]

        
    def read_gallery(self, dataroot,dataset_ids, reverse_option=False,gallery_gray=False, use_record=True,typ='sketch_photo'):

        print('init...........')
        
        self.gallery_gray=gallery_gray
        
        dataset_map=sutil.read_yaml('setting/datasetid.yaml')
        dataset_ids.sort()
        print(dataset_ids)


        if typ=='sketch_photo':
            search_name =  'sketch' if    reverse_option else 'photo'
        elif typ=='RGBD' :
            search_name =  'RGB' if    reverse_option else 'D'
        elif typ=='NIR_VIS':
            search_name =  'NIR'  if  reverse_option else 'RGB'
        else:
            raise(RuntimeError('read_gallery unknown type'))

        self.imageYs=[]
        imageYs=self.imageYs
        for x in  dataset_ids:
            datasetname=dataset_map[x]
            print( os.path.join( dataroot,datasetname,'test',search_name) )
            for root,dirs,files in os.walk(os.path.join( dataroot,datasetname,'test',search_name),topdown=True):
                files.sort()
                for name in files:
                    fileloc=os.path.join(root,name)
                    imageYs.append( fileloc)
        imageYs.sort(key=sort_by_name)





    def get_imageXs(self, probe_gray=False,batchsize=1,verbose=False):

        
        imageYs=self.imageYs
        domainX_dirs=self.domainX_dirs

        self.imageXs=[]
        imageXs=self.imageXs
        for domainX_dir in domainX_dirs:
            ret_imageXs=os.listdir(domainX_dir)
            for ret_image_x in ret_imageXs:
                imageXs.append( os.path.join( domainX_dir,ret_image_x) )
        imageXs.sort(key=sort_by_name)

        ####tobe  ########
        print(len(imageXs))
        print(len(imageYs))

        assert len(imageXs)==len(imageYs),'len(imageXs)!=len(imageYs)'

        NimageYs=len(imageYs)
        NimageXs=len(imageXs)


        ans_mae=0
        ans_mse=0
        ans_sqrtmse=0
        ans_psnr=0
        ans_ssim=0


        delete_psnr=0

        if verbose:
            S_deltanums= NP.zeros(256)

            global org_dict
            org_dict=OrderedDict()
            for i in range(256):
                org_dict[ NP.int64(i) ]= 0

            if os.path.exists(verbose_path):
                shutil.rmtree(verbose_path)
            sutil.makedirs(verbose_path)

        opt=sutil.get_opt()

        for i in tqdm(range(0,NimageYs) ):


            Ycolor2gray2color = True if self.istarget and 'GTY_color2gray2color' in opt and opt['GTY_color2gray2color']  else False

            img_y= sutil.readimg_to_tensor_rgb(imageYs[i],Ycolor2gray2color) if not probe_gray else  sutil.readimg_to_tensor_gray(imageYs[i],Ycolor2gray2color)


            img_y= Variable(img_y.unsqueeze(0))
            img_y_01= (img_y+1)/2
            img_y=     img_y_01*255.0
            numpy_y=img_y[0].detach().cpu().float().numpy().astype(NP.float64)



            Xcolor2gray2color = True if self.istarget and 'SynY_color2gray2color' in opt and opt['SynY_color2gray2color']  else False

            img_x= sutil.readimg_to_tensor_rgb(imageXs[i],Xcolor2gray2color) if not probe_gray else  sutil.readimg_to_tensor_gray(imageXs[i],Xcolor2gray2color)

            img_x= Variable(img_x.unsqueeze(0))
            img_x_01= (img_x+1)/2
            img_x=     img_x_01*255.0
            numpy_x=img_x[0].detach().cpu().float().numpy().astype(NP.float64)

            ans_mae+=   NP.abs(numpy_x-numpy_y).mean()

            ret_mse= NP.mean((numpy_x - numpy_y) ** 2 )

            ans_mse+=   ret_mse
            ans_sqrtmse+=   NP.sqrt( ret_mse)

            ret_ssim = NP.float64( func_cal_ssim(img_x_01, img_y_01).item() ) 
            ans_ssim+= ret_ssim

            if ( abs(ret_mse)<0.00001 ):

                print('ret_mse {}'.format(ret_mse) )
                
                delete_psnr+=1

            else:
                ans_psnr+=  10 * NP.log10(255.0**2/ret_mse)

            if verbose:

                img_shortname = os.path.split(imageYs[i])[1].split('.')[0]

                des_path = os.path.join(verbose_path, img_shortname+'_per_element_mae_dis')
                

                delta_nums=verbose_debug(NP.abs(numpy_x-numpy_y), des_path)


                S_deltanums+=delta_nums


        ans_mae/=NimageXs
        ans_mse/=NimageXs
        ans_sqrtmse/=NimageXs
        ans_psnr/= (NimageXs-delete_psnr)
        ans_ssim/=NimageXs

        if verbose:
            S_deltanums/=NimageXs
            des_path = os.path.join(verbose_path, 'mean_per_element_mae_dis.png')
            #verbose_debug(S_deltanums, des_path, scale=1)

            plt.figure()
            plt.title('distribution of per element distance')
            plt.xlim(0,255)
            xs=NP.arange(0,256,1)
            plt.plot(xs,S_deltanums)
            plt.xlabel("distance")
            plt.ylabel("frequence")
            plt.savefig(des_path)
            plt.clf()
            plt.close()

            f=open( os.path.join(verbose_path, 'mean_per_element_mae_dis.txt'),'w')
            for key,val in zip(xs,S_deltanums):
                f.write( '{}:{}\n'.format(key,val))
            f.close()


            des_path = os.path.join(verbose_path, 'mean_per_element_mae_dis2.png')
            #verbose_debug(S_deltanums, des_path, scale=1)

            plt.figure()
            plt.title('distribution of per element distance')
            plt.ylim(0,3000)
            plt.xlim(0,255)
            xs=NP.arange(0,256,1)
            plt.plot(xs,S_deltanums)
            plt.xlabel("distance")
            plt.ylabel("frequence")
            plt.savefig(des_path)
            plt.clf()
            plt.close()


        #probe=NP.array(probe).astype(NP.float64)
        #probe=NP.array(probe)

        # global self.outpath
        # fp=open( os.path.join(self.outpath, 'mae.txt') ,'w')
        # fp.write('mae  =  {}\n'.format(ans_mae) )
        # fp.close()

        fp=open( os.path.join(self.outpath, 'mae_mse_sqrtmse_psnr_ssim.txt') ,'w')
        fp.write('mae  =  {}\n'.format(ans_mae) )
        fp.write('mse  =  {}\n'.format(ans_mse) )
        fp.write('sqrtmse  =  {}\n'.format(ans_sqrtmse) )
        fp.write('psnr  =  {}\n'.format(ans_psnr) )
        fp.write('ssim  =  {}\n'.format(ans_ssim) )
        fp.write('delete_psnr nums  =  {}\n'.format(delete_psnr) )
        fp.close()

        return ans_mae,ans_mse,ans_sqrtmse,ans_psnr,ans_ssim
 


    def run(self, istarget,name,train_step, epochL=-1,epochR=-1, \
        add=1,probe_gray=False,load_step=None ,load_epoch=None,\
        typ='sketch_photo',original_data=[],batchsize=1,gpu_ids=[0],verbose=False):

        self.istarget=istarget

        best_epoch=-1
        best_mae=-1

        maes,mses,sqrtmses,psnrs,ssims=[],[],[],[],[]
        mae_epochs,mse_epochs,sqrtmse_epochs,psnr_epochs,ssim_epochs=[],[],[],[],[]
        gallery_gray=self.gallery_gray


        print(epochL)
        print(epochR)
        

        for epoch in range(epochL,epochR+1,add):

            print('epoch={}'.format(epoch))

            if load_step is None:
                load_step= train_step


            if len(original_data)!=0:
                original_data.sort()
                self.domainX_dirs= original_data
            else:
                self.domainX_dirs= [os.path.join('results',name,  load_step ,'test_'+str(load_epoch if load_epoch else epoch ),'images')  ]

            skip=False
            for domainX_dir in self.domainX_dirs:
                if not os.path.exists(domainX_dir):
                    skip=True
                    break    

            if skip:
                continue

            opt=sutil.get_opt()

            pg_type=  "probe_{}_gallery_{}".format( 'gray' if probe_gray else 'color' , 'gray' if gallery_gray else 'color'  ) 

            Ycolor2gray2color = True if self.istarget and 'GTY_color2gray2color' in opt and opt['GTY_color2gray2color']  else False

            Xcolor2gray2color = True if self.istarget and 'SynY_color2gray2color' in opt and opt['SynY_color2gray2color']  else False

            if Ycolor2gray2color:
                pg_type+='_GTYcgc'

            if Xcolor2gray2color:
                pg_type+='_SynYcgc'


            self.outpath=  os.path.join('objective image quality',name, train_step,  pg_type  ,'test_'+str(epoch)  ) 



            global verbose_path
            verbose_path = os.path.join(self.outpath, 'verbose')
       
            if not os.path.exists(self.outpath):
                os.makedirs(self.outpath)


            if epoch!=epochR:
                assert(  os.path.exists(os.path.join(self.outpath, 'mae_mse_sqrtmse_psnr_ssim.txt'))   )


            if os.path.exists(os.path.join( self.outpath,'mae_mse_sqrtmse_psnr_ssim.txt' )  ) and not verbose:
            

                fp=open( os.path.join(self.outpath, 'mae_mse_sqrtmse_psnr_ssim.txt') ,'r')
                ret_mae=float(fp.readline().strip('\n').split()[2])
                ret_mse=float(fp.readline().strip('\n').split()[2])
                ret_sqrtmse=float(fp.readline().strip('\n').split()[2])
                ret_psnr=float(fp.readline().strip('\n').split()[2])
                ret_ssim=float(fp.readline().strip('\n').split()[2])


                maes.append(ret_mae)
                mses.append(ret_mse)
                sqrtmses.append(ret_sqrtmse)
                psnrs.append(ret_psnr)
                ssims.append(ret_ssim)



                mae_epochs.append(epoch)
                mse_epochs.append(epoch)
                sqrtmse_epochs.append(epoch)
                psnr_epochs.append(epoch)
                ssim_epochs.append(epoch)


                fp.close()
                    
                continue

            ans_mae,ans_mse,ans_sqrtmse,ans_psnr,ans_ssim=self.get_imageXs(probe_gray=probe_gray, batchsize=batchsize,verbose=verbose)

            imageXs=self.imageXs
            imageYs=self.imageYs


            fp=open(os.path.join(self.outpath,'g_order.txt'),'w' )
            for x in imageYs:
                fp.write(x+'\n')
            fp.close()

            fp=open(os.path.join(self.outpath,'p_order.txt') ,'w')
            for x in imageXs:
                fp.write(x+'\n')
            fp.close()

            maes.append(ans_mae)
            mses.append(ans_mse)
            sqrtmses.append(ans_sqrtmse)
            psnrs.append(ans_psnr)
            ssims.append(ans_ssim)


            mae_epochs.append(epoch)
            mse_epochs.append(epoch)
            sqrtmse_epochs.append(epoch)
            psnr_epochs.append(epoch)
            ssim_epochs.append(epoch)

            fp.close()

        maes=NP.array(maes)
        mses=NP.array(mses)
        sqrtmses=NP.array(sqrtmses)
        psnrs=NP.array(psnrs)
        ssims=NP.array(ssims)



        best_mae,best_mse,best_sqrtmse,best_psnr,best_ssim= NP.min(maes),NP.min(mses),NP.min(sqrtmses),NP.max(psnrs),NP.max(ssims)
        wrost_mae,wrost_mse,wrost_sqrtmse,wrost_psnr,wrost_ssim= NP.max(maes),NP.max(mses),NP.max(sqrtmses),NP.min(psnrs),NP.min(ssims)


        best_epoch_mae,best_epoch_mse,best_epoch_sqrtmse,best_epoch_psnr,best_epoch_ssim=  mae_epochs[NP.argmin(maes)] \
            ,mse_epochs[NP.argmin(mses)],sqrtmse_epochs[NP.argmin(sqrtmses)],psnr_epochs[NP.argmax(psnrs)], ssim_epochs[NP.argmax(ssims)]

        wrost_epoch_mae,wrost_epoch_mse,wrost_epoch_sqrtmse,wrost_epoch_psnr,wrost_epoch_ssim=  mae_epochs[NP.argmax(maes)] \
            ,mse_epochs[NP.argmax(mses)],sqrtmse_epochs[NP.argmax(sqrtmses)],psnr_epochs[NP.argmin(psnrs)], ssim_epochs[NP.argmin(ssims)]

        fp=open(os.path.join('objective image quality', name, train_step , pg_type,'best_mae.txt'),'w')
        fp.write('best_epoch: {} best_mae: {}\n'.format(best_epoch_mae,best_mae) )
        fp.write('wrost_epoch: {} wrost_mae: {}\n'.format(wrost_epoch_mae,wrost_mae) )
        fp.write('mean_mae: {}\n'.format( maes.mean() ) )
        fp.write('std_mae: {}\n'.format( maes.std() ) )

        for i in range(len(mae_epochs) ):
            fp.write('epoch '+str(mae_epochs[i])+' mae= '+str(maes[i])+' \n')
        fp.close()

        fp=open(os.path.join('objective image quality', name, train_step , pg_type,'best_mse.txt'),'w')
        fp.write('best_epoch: {} best_mse: {}\n'.format(best_epoch_mse,best_mse) )
        fp.write('wrost_epoch: {} wrost_mse: {}\n'.format(wrost_epoch_mse,wrost_mse) )
        fp.write('mean_mse: {}\n'.format( mses.mean() ) )
        fp.write('std_mse: {}\n'.format( mses.std() ) )

        for i in range(len(mse_epochs) ):
            fp.write('epoch '+str(mse_epochs[i])+' mse= '+str(mses[i])+' \n')
        fp.close()

        fp=open(os.path.join('objective image quality', name, train_step , pg_type,'best_sqrtmse.txt'),'w')
        fp.write('best_epoch: {} best_sqrtmse: {}\n'.format(best_epoch_sqrtmse,best_sqrtmse) )
        fp.write('wrost_epoch: {} wrost_sqrtmse: {}\n'.format(wrost_epoch_sqrtmse,wrost_sqrtmse) )
        fp.write('mean_sqrtmse: {}\n'.format( sqrtmses.mean() ) )
        fp.write('std_sqrtmse: {}\n'.format( sqrtmses.std() ) )
        for i in range(len(sqrtmse_epochs) ):
            fp.write('epoch '+str(sqrtmse_epochs[i])+' sqrtmse= '+str(sqrtmses[i])+' \n')
        fp.close()


        fp=open(os.path.join('objective image quality', name, train_step , pg_type,'best_psnr.txt'),'w')
        fp.write('best_epoch: {} best_psnr: {}\n'.format(best_epoch_psnr,best_psnr) )
        fp.write('wrost_epoch: {} wrost_psnr: {}\n'.format(wrost_epoch_psnr,wrost_psnr) )
        fp.write('mean_psnr: {}\n'.format( psnrs.mean() ) )
        fp.write('std_psnr: {}\n'.format( psnrs.std() ) )
        for i in range(len(psnr_epochs) ):
            fp.write('epoch '+str(psnr_epochs[i])+' psnr= '+str(psnrs[i])+' \n')
        fp.close()

        fp=open(os.path.join('objective image quality', name, train_step , pg_type,'best_ssim.txt'),'w')
        fp.write('best_epoch: {} best_ssim: {}\n'.format(best_epoch_ssim,best_ssim) )
        fp.write('wrost_epoch: {} wrost_ssim: {}\n'.format(wrost_epoch_ssim,wrost_ssim) )
        fp.write('mean_ssim: {}\n'.format( ssims.mean() ) )
        fp.write('std_ssim: {}\n'.format( ssims.std() ) )
        for i in range(len(ssim_epochs) ):
            fp.write('epoch '+str(ssim_epochs[i])+' ssim= '+str(ssims[i])+' \n')
        fp.close()

def verbose_debug(delta, des_path):

    txt_path=des_path+'.txt'
    png_path=des_path+'.png'


    #if os.path.exists(txt_path):

    # for loc in [txt_path,png_path]:
    #     if os.path.exists(loc):
    #         shutil.remove(loc)

        

    plt.figure()
    plt.title('distribution of per element distance')
    delta= delta.reshape(-1)
    plt.xlim(0,255)

    delta=(delta+0.5).astype(NP.int64 )

    delta_nums=copy.deepcopy(org_dict)

    # print(len(delta_nums))

    delta_nums.update  ( Counter(delta) ) 


    f=open(txt_path,'w')
    for key,val in delta_nums.items():
        f.write( '{}:{}\n'.format(key,val))
    f.close()


    # print(len(delta_nums))

    delta_nums = NP.array( list( delta_nums.values() ) )  

    # print(delta_nums.shape)
    

    plt.hist(delta, bins=255,color='g')
    plt.xlabel("distance")
    plt.ylabel("frequence")
    plt.savefig(png_path)
    plt.clf()
    plt.close()



    return delta_nums

















    

