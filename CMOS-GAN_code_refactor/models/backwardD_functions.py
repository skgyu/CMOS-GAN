#-*- coding:utf-8 -*-
from torch.autograd import Variable




def backward_D_target(self,fake_Y_target):

    loss_D_Y=self.dis_Y.calc_dis_loss( input_fake=Variable(fake_Y_target.data), input_real= self.data_Y_source ) *self.opt.lambda_D


    self.update_loss('loss_D_Y' , loss_D_Y.item() )
    print('loss_D_Y=', loss_D_Y.item())


    loss_D_Y.backward()




