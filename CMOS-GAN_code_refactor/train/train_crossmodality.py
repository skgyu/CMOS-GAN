#coding: utf-8
import os
import torch
from data.PairedDataset import PairedDataset
from data import tsfUnpairedDataset
from options.train_options import TrainOptions
from models.models import create_model
from stools import sutil  
from tqdm import tqdm
from tools.MAE_caler import MAEcaler
maecaler=MAEcaler()

def tf_test(model,opt,epoch,epochR,dataset_ids,
    recognizer, domain, rcg_model_name ,save_dir_name,testloader=None,calmae=True):


    model.evalstate()

    opt.phase='test'

    original_data=[]

    dataset_map=sutil.read_yaml('setting/datasetid.yaml')

    assert(opt.typ in ['sketch_photo','NIR_VIS'])

    if domain in  ["_source","_target"] :


        assert( testloader is not None )

        des_dir = os.path.join(opt.results_dir, opt.name, opt.train_step+domain,'%s_%s' % ("test", epoch),'images')  

        sutil.makedirs(des_dir)
        


        print('test'+domain)


        need_len=0
        for dataset in dataset_ids:
            gallery_dir= os.path.join(opt.dataroot,dataset_map[dataset],"test",opt.domainY) 
            need_len+=  len(os.listdir(gallery_dir))

        if need_len == len( os.listdir(des_dir) ):
            pass

        else:

            for i, data in tqdm(enumerate(testloader)):


                if domain=="_source":
                    model.set_input_pre(data,'source_')
                    visuals=model.test_input_output_network('data_X_source','fake_Y_source','G_X2Y_source')

                else:
                    model.set_input_pre(data,'target_')
                    visuals=model.test_input_output_network('data_X_target','fake_Y_target','G_X2Y_source')

                sutil.save_images(visuals=visuals,image_paths=model.get_image_paths(),des_dir=des_dir)


        if not opt.no_recognition or calmae:

            tmp_dir=os.path.join(opt.results_dir, opt.name, opt.train_step+domain,'%s_%s' % ("test", epoch),'images')
            original_data.append( tmp_dir )
            original_data.sort()


        if not opt.no_recognition:

            recognizer.read_gallery(dataroot=opt.dataroot,dataset_ids=dataset_ids,gallery_type='gallery_{}'.format(opt.domainY),
                rcg_model_name=rcg_model_name[opt.domainY],
                read_imageYs=[],nobackgrounds=False, reverse_option=False, use_record=False,gallery_gray= opt.dim_Y==1) 
            

            recognizer.run(train_step=opt.train_step+domain, probe_type='probe_estimated_{}'.format(opt.domainY), epochL=opt.test_fre
                ,epochR=epochR,add=opt.test_fre, original_data=original_data, batchsize=opt.testbatchSize, gpu_ids=opt.gpu_ids
                , probe_gray=opt.dim_X==1, save_dir_name=save_dir_name[opt.domainY])


        if calmae:

            maecaler.read_gallery(dataroot=opt.dataroot, dataset_ids = dataset_ids, reverse_option=False, use_record=False,typ=opt.typ,gallery_gray=opt.dim_Y==1)  #target_dataset_ids

            maecaler.run(istarget= (domain=='_target'), name=opt.name,train_step=opt.train_step+domain, epochL=opt.test_fre,epochR=epochR,add=opt.test_fre, \
                typ=opt.typ, original_data=original_data, batchsize=opt.testbatchSize, gpu_ids=opt.gpu_ids,probe_gray=opt.dim_X==1)

        

    elif domain=='_all':

        print('test all')

        des_dir_source = os.path.join(opt.results_dir, opt.name, opt.train_step+'_source','%s_%s' % ("test", epoch),'images')  
        des_dir_target = os.path.join(opt.results_dir, opt.name, opt.train_step+'_target','%s_%s' % ("test", epoch),'images')  



        if not opt.no_recognition or calmae and  all(x not in [4,34] for x in dataset_ids):

            tmp_dir=os.path.join(opt.results_dir, opt.name, opt.train_step+"_source",'%s_%s' % ("test", epoch),'images')
            original_data.append( tmp_dir )
            tmp_dir=os.path.join(opt.results_dir, opt.name, opt.train_step+"_target",'%s_%s' % ("test", epoch),'images')
            original_data.append( tmp_dir )
            original_data.sort()


        if not opt.no_recognition:


            recognizer.read_gallery(dataroot=opt.dataroot,dataset_ids=dataset_ids,gallery_type='gallery_{}'.format(opt.domainY),
                rcg_model_name=rcg_model_name[opt.domainY],
                read_imageYs=[],nobackgrounds=False, reverse_option=False, use_record=False,gallery_gray= opt.dim_Y==1) 

            recognizer.run(train_step=opt.train_step+domain, probe_type='probe_estimated_{}'.format(opt.domainY), epochL=opt.test_fre,epochR=epochR,add=opt.test_fre, \
                original_data=original_data, batchsize=opt.testbatchSize, gpu_ids=opt.gpu_ids,probe_gray=opt.dim_X==1
                ,save_dir_name=save_dir_name[opt.domainY])



        if calmae and  all(x not in [4,34] for x in dataset_ids):

            maecaler.read_gallery(dataroot=opt.dataroot, dataset_ids = dataset_ids, reverse_option=False, use_record=False,typ=opt.typ,gallery_gray=opt.dim_Y==1)  #target_dataset_ids

            maecaler.run(istarget= (domain=='_target'), name=opt.name,train_step=opt.train_step+domain, epochL=opt.test_fre,epochR=epochR,add=opt.test_fre, \
                typ=opt.typ, original_data=original_data, batchsize=opt.testbatchSize, gpu_ids=opt.gpu_ids,probe_gray=opt.dim_X==1)
        

    opt.phase='train'





if __name__ == '__main__':



    opt = TrainOptions().parse()

    assert(opt.model=='modelCrossModality')


    if 'no_target_imageYs' in opt and  opt['no_target_imageYs']:
        no_target_imageYs=True
    else:
        no_target_imageYs=False


    face_dataset = tsfUnpairedDataset(dataroot=opt.dataroot,dataset_ids=opt.dataset_ids,source_dataset_ids=opt.source_dataset_ids \
        ,phase='train',img_size=opt.img_size,opt=opt,aug=opt.aug, no_target_imageYs=no_target_imageYs, use_percent_dataset=opt.use_percent_dataset)

    trainloader = torch.utils.data.DataLoader(face_dataset, batch_size=opt.batchSize,pin_memory=True, shuffle=True, num_workers=opt.num_workers)


    val_dataset = tsfUnpairedDataset(dataroot=opt.dataroot,dataset_ids=opt.dataset_ids,source_dataset_ids=opt.source_dataset_ids \
        ,phase='train',img_size=opt.img_size,opt=opt,aug=False, no_target_imageYs=no_target_imageYs, use_percent_dataset=opt.use_percent_dataset)
    val_dataset.no_match_random()

    t_dataset = tsfUnpairedDataset(dataroot=opt.dataroot,dataset_ids=opt.dataset_ids,source_dataset_ids=opt.source_dataset_ids \
        ,phase='test',img_size=opt.img_size,opt=opt,aug=False, no_target_imageYs=no_target_imageYs)
    t_dataset.no_match_random()




    if 'hard_mining' in opt and opt.hard_mining:
        face_dataset.prepare_id_images()



    opt.num_identities = face_dataset.num_identities()
    len_face_dataset=len(face_dataset)



    print(len(face_dataset))
    sutil.log(len(face_dataset),'main')
    sutil.log('num_identities','main')
    sutil.log(opt.num_identities,'main')
    sutil.log('num_identities record','main')


    recognizer_rgb=None
    if not opt.no_recognition:
        from tools.recognizer import Recognizer 
        recognizer_rgb=Recognizer(opt)
        recognizer_rgb.init(opt,recog_item='recog_state_dict_rgb_for_fusion',rcg_func=opt.rgb_rcg_func_rgb_for_fusion,img_dim=opt.dim_X)


    test_dataset_source = PairedDataset(dataroot=opt.dataroot,dataset_ids=opt.test_source_dataset_ids \
        ,phase='test',img_size=opt.img_size,opt=opt,aug=False ,pre='source_')
    testloader_source = torch.utils.data.DataLoader(test_dataset_source, batch_size=opt.testbatchSize,pin_memory=True, shuffle=False, num_workers=1)


    test_dataset = PairedDataset(dataroot=opt.dataroot,dataset_ids=opt.test_dataset_ids \
        ,phase='test',img_size=opt.img_size,opt=opt,aug=False ,pre='target_')
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.testbatchSize,pin_memory=True, shuffle=False, num_workers=1)


  
    if 'hard_mining' in opt and opt.hard_mining:
        opt.face_dataset=face_dataset


    model = create_model(opt)

    if not opt.no_recognition:
        recognizer_rgb.replacemodel(model.feature_extraction_model)



    testdata=t_dataset.prepare_testdata(opt.batchSize)

    valdata=val_dataset.prepare_testdata(opt.batchSize)
    

    if 'test_at_beginning' in opt and opt.test_at_beginning:
        
        model.evalstate()

        epoch=opt.which_epoch

        if 'policy' in opt.recog_state_dict and opt.recog_state_dict.policy!='only_last':

            rcg_model_name='{}_{}_finetuned_{}epochs_'.format(opt.name, opt.train_step, epoch) +opt.recog_state_dict.name
            save_dir_name= 'finetuned_'+opt.recog_state_dict.name

        else:

            rcg_model_name=  opt.recog_state_dict.name
            save_dir_name=   opt.recog_state_dict.name

        calmae=False if ('cal_MAE' in opt and not opt.cal_MAE) else True


        tf_test(model,opt,epoch,epoch,dataset_ids=opt.test_dataset_ids,domain='_target',recognizer=recognizer_rgb,testloader=testloader,
            rcg_model_name= { opt.domainY: rcg_model_name },
            save_dir_name=  { opt.domainY: save_dir_name },calmae=calmae)

        tf_test(model,opt,epoch,epoch,dataset_ids=opt.test_source_dataset_ids,domain='_source',recognizer=recognizer_rgb,testloader=testloader_source,
            rcg_model_name= { opt.domainY: rcg_model_name },
            save_dir_name=  { opt.domainY: save_dir_name },calmae=calmae)


        tf_test(model,opt,epoch,epoch,dataset_ids=opt.test_source_dataset_ids+opt.test_dataset_ids,domain='_all',recognizer=recognizer_rgb, 
            rcg_model_name= { opt.domainY: rcg_model_name },
            save_dir_name=  { opt.domainY: save_dir_name } ,calmae=calmae)




    model.record_learning_rate(opt.start_epoch)

    iter_num=0


    for epoch in range( opt.start_epoch,opt.max_epoch+1):

        model.epoch=epoch

        model.clear_loss()


        for i, data in enumerate(trainloader):
            
            

            model.trainstate()
  
            print('enumerate',i)
            model.set_input(data)
            model.optimize_step()

            iter_num+= model.data_X_source.size(0)
            if iter_num>=len_face_dataset//3:
               model.evalstate()  #
               model.set_input(testdata)
               model.test(epoch=epoch+1, phase='test', imgy_repeat3= (opt.dim_X==3*opt.dim_Y) )
               model.set_input(valdata)
               model.test(epoch=epoch+1, phase='train', imgy_repeat3= (opt.dim_X==3*opt.dim_Y))
               iter_num-=len_face_dataset//3


        if (epoch+1) % opt.save_epoch_freq == 0:
            model.saveall(epoch+1)
            

        errors = model.get_ave_errors(epoch)


      
        model.evalstate()
        model.testC(epoch)

        if not opt.no_recognition:
            recognizer_rgb.replacemodel(model.feature_extraction_model)  ########################

        if (epoch+1)%opt.test_fre == 0 :


            if 'policy' in opt.recog_state_dict and opt.recog_state_dict.policy!='only_last':

                rcg_model_name='{}_{}_finetuned_{}epochs_'.format(opt.name, opt.train_step, epoch+1) +opt.recog_state_dict.name
                save_dir_name= 'finetuned_'+opt.recog_state_dict.name

            else:

                rcg_model_name=  opt.recog_state_dict.name
                save_dir_name=   opt.recog_state_dict.name

            calmae=False if ('cal_MAE' in opt and not opt.cal_MAE) else True


            tf_test(model,opt,epoch+1,epoch+1,dataset_ids=opt.test_dataset_ids,domain='_target',recognizer=recognizer_rgb,testloader=testloader,
                rcg_model_name= { opt.domainY: rcg_model_name },
                save_dir_name=  { opt.domainY: save_dir_name },calmae=calmae)

            tf_test(model,opt,epoch+1,epoch+1,dataset_ids=opt.test_source_dataset_ids,domain='_source',recognizer=recognizer_rgb,testloader=testloader_source,
                rcg_model_name= { opt.domainY: rcg_model_name },
                save_dir_name=  { opt.domainY:save_dir_name },calmae=calmae)


            tf_test(model,opt,epoch+1,epoch+1,dataset_ids=opt.test_source_dataset_ids+opt.test_dataset_ids,domain='_all',recognizer=recognizer_rgb, 
                rcg_model_name= { opt.domainY: rcg_model_name },
                save_dir_name=  { opt.domainY: save_dir_name } ,calmae=calmae)


        model.update_learning_rate(epoch+1)



    model.saveall(opt.max_epoch+1)


      
