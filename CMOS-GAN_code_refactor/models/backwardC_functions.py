#-*- coding:utf-8 -*-
import torch
from stools import sutil



def classify_loss_forC(self,predict, target,idx,lambda_cls=None):

    if lambda_cls is None:
        lambda_cls =  self.opt.lambda_cls

    
    loss_cls=self.criterion_cls(predict, target ) *lambda_cls
    
    
    batchsize=predict.size()[0]
    self.sizes_C[idx]+= batchsize

    
    predict_numpy= (torch.max(predict.detach().cpu(),dim=1)[1]).numpy()
    target_numpy=target.detach().cpu().numpy()   
    self.accuracy[idx]+=    (predict_numpy==target_numpy).sum(0)
    print('classify_loss_forC idx{} correct num= {}/{} '.format(idx, (predict_numpy==target_numpy).sum(0), batchsize) )

    if self.epoch==0:
        sutil.log(  'classify_loss_forC epoch{}: idx={}'.format(self.epoch, idx)  ,'cls_detail.log')
        sutil.log(  'predict'  ,'cls_detail.log')
        sutil.log(  predict  ,'cls_detail.log')
        sutil.log(  'target'  ,'cls_detail.log')
        sutil.log(  target  ,'cls_detail.log')

    return loss_cls


def classify_loss_forG(self,predict,target,idx,lambda_cls=None ):
    
    if lambda_cls is None:
        lambda_cls =  self.opt.lambda_cls


    loss_cls=self.criterion_cls(predict, target ) *lambda_cls

    batchsize=predict.size()[0]
    self.sizes_C_forG[idx]+= batchsize



    predict_numpy= (torch.max(predict.detach().cpu(),dim=1)[1]).numpy()
    target_numpy=target.detach().cpu().numpy()
    self.accuracy_forG[idx]+=    (predict_numpy==target_numpy).sum(0)
    print('classify_loss_forG idx{} correct num= {}/{} '.format(idx, (predict_numpy==target_numpy).sum(0), batchsize) )


    if self.epoch==0:
        sutil.log(  'classify_loss_forG epoch{}: idx={}'.format(self.epoch, idx)  ,'cls_detail.log')
        sutil.log(  'predict'  ,'cls_detail.log')
        sutil.log(  predict  ,'cls_detail.log')
        sutil.log(  'target'  ,'cls_detail.log')
        sutil.log(  target  ,'cls_detail.log')

    return loss_cls