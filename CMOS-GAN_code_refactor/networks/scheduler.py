# -*- coding: utf-8 -*-
from torch.optim import lr_scheduler
###############################################################################
# Functions
###############################################################################



def get_scheduler(optimizer, optimizer_name, opt  ):

    if not opt.continue_train:
        last_epoch=-1

    elif isinstance(opt.which_epoch,int):
        last_epoch=int(opt.which_epoch)
        for i,x in enumerate(optimizer.param_groups): 
            x['initial_lr']=opt[optimizer_name]['lr'][i]
    else:
        raise(RuntimeError('opt.which_epoch error') )

    lr_strategy =  opt[optimizer_name]
    lr_policy = lr_strategy['lr_policy']


    if lr_policy == 'lambda2':
        def lambda_rule(epoch):           
            lr_l = 1.0 - max(0, epoch  + 1 - lr_strategy.niter) / float(lr_strategy.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)  

    elif lr_policy == 'origin':
        def lambda_rule(epoch):
            lr_l = 1
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch ) 

    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_strategy.step_size, gamma=lr_strategy.gamma , last_epoch=last_epoch )
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)

    
    print('lr={}'.format(optimizer.param_groups[0]['lr']) )

    return scheduler












        

       




