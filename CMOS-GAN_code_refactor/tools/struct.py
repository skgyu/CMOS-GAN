#coding: utf-8
import os
import torch
import numpy as NP
from torch.autograd import Variable
from stools import sutil



def get_parent_abs_folder(loc,rcg_model_name,isgray=False):

    pfolder=os.path.abspath(os.path.join(loc, ".."))

    pfolder_path,pfolder_name= os.path.split(pfolder)
    pfolder_name=pfolder_name+'_{}_preload_'.format( 'gray' if isgray else 'color' )  +rcg_model_name

    return os.path.join(pfolder_path,pfolder_name) 


def pre_loc(loc,rcg_model_name,isgray=False):

    pfolder=os.path.abspath(os.path.join(loc, ".."))
    filename= os.path.split(loc)[1].split('.')[0]+'.npy'

    pfolder_path,pfolder_name= os.path.split(pfolder)
    # pfolder_name=pfolder_name+'_preload_'+rcg_model_name
    pfolder_name=pfolder_name+'_{}_preload_'.format( 'gray' if isgray else 'color' )  +rcg_model_name
    newloc= os.path.join(pfolder_path,pfolder_name,filename)

    return newloc




def read_info(loc  ,feature_extraction_model,rcg_model_name, isgray=False,save_img_feature=True):

    pfolder=os.path.abspath(os.path.join(loc, ".."))
    filename= os.path.split(loc)[1].split('.')[0]+'.npy'

    pfolder_path,pfolder_name= os.path.split(pfolder)
    pfolder_name=pfolder_name+'_{}_preload_'.format( 'gray' if isgray else 'color' )  +rcg_model_name
    newloc= os.path.join(pfolder_path,pfolder_name,filename)

    if os.path.exists(newloc):
        return NP.load(newloc)


    sutil.makedirs( os.path.join(pfolder_path,pfolder_name) )


    img=sutil.readimg_to_tensor_rgb(loc) if not isgray else   sutil.readimg_to_tensor_gray(loc)

    img=img.cuda()

    img=Variable(img.unsqueeze(0))

    img= feature_extraction_model.rcg_func(img)

    fc = feature_extraction_model(img)[0][0]  #feature,out

    fc = torch.nn.functional.normalize(fc, p=2, dim=1, eps=1e-12)

    info=   fc[0].detach().cpu().float().numpy()  #get a 2048 dim numpy vector

    NP.save( newloc, info)

    return info


