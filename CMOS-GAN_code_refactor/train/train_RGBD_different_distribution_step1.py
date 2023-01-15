#coding: utf-8
import os
import torch
import os
from data.PairedDataset import PairedDataset
from data import tsfUnpairedDataset
from options.train_options import TrainOptions
from models.models import create_model
from stools import sutil 
from tqdm import tqdm
from tools.MAE_caler import MAEcaler
maecaler=MAEcaler()


def tf_test(model,opt,epoch,epochR,dataset_ids,
    recognizer_fusion, domain, rcg_model_name ,save_dir_name,testloader=None):


    model.evalstate()

    opt.phase='test'

    original_data=[]

    dataset_map=sutil.read_yaml('setting/datasetid.yaml')


    if domain in  ["_source","_target"] :


        assert( testloader is not None )

        des_dir = os.path.join(opt.results_dir, opt.name, opt.train_step+domain,'%s_%s' % ("test", epoch),'images')  

        sutil.makedirs(des_dir)
        

        print('test'+domain)

        image_domain_ids= dataset_ids


        need_len=0
        for dataset in image_domain_ids:
            ret= os.path.join(opt.dataroot,dataset_map[dataset],"test",'D') 
            need_len+=  len(os.listdir(ret))

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




        tmp_imageY=[]
        for x in dataset_ids:
            datasetname=dataset_map[x]
            gallery_list_loc=   dataset_map['gallery'][x]['D']  #_fake_Y_source.png

            f=open(gallery_list_loc,'r')
            for line in f.readlines():

                filename=line.strip().split(' ')[0].split('/')[2].split('.')[0]+'_fake_Y{}.png'.format(domain)
                tmp_loc=os.path.join( des_dir ,filename)
        
                assert ( os.path.exists(tmp_loc) )

                tmp_imageY.append( tmp_loc   )

            f.close()
        tmp_imageY.sort()

        print('detph')

        recognizer_fusion.recognizer_d.read_gallery(dataroot=opt.dataroot,dataset_ids=dataset_ids,gallery_type='gallery_Depth_estimated',
            rcg_model_name=rcg_model_name['d'],
            read_imageYs=tmp_imageY,nobackgrounds=True, reverse_option=False, use_record=False,gallery_gray= opt.dim_Y==1) 
        
        tmp_dir=os.path.join(opt.results_dir, opt.name, opt.train_step+domain,'%s_%s' % ("test", epoch),'images')
        original_data.append( tmp_dir )
        # tmp_dir=os.path.join(opt.results_dir, opt.name, opt.train_step+"_target",'%s_%s' % ("test", epoch),'images')
        # original_data.append( tmp_dir )
        original_data.sort()

        recognizer_fusion.recognizer_d.run(train_step=opt.train_step+domain, probe_type='probe_depth_estimated', epochL=opt.test_fre,epochR=epochR,add=opt.test_fre, \
            original_data=original_data, batchsize=opt.testbatchSize, gpu_ids=opt.gpu_ids,probe_gray=opt.dim_Y==1
            ,save_dir_name=save_dir_name['d'])


        print('rgb')

        recognizer_fusion.recognizer_rgb.read_gallery(dataroot=opt.dataroot,dataset_ids=dataset_ids, gallery_type='gallery_RGB_groundtruth'
            ,rcg_model_name=rcg_model_name['rgb']
            ,nobackgrounds=True, reverse_option=True, use_record=False,gallery_gray= opt.dim_X==1)  #target_dataset_ids

        recognizer_fusion.recognizer_rgb.run(train_step=opt.train_step+domain, probe_type='probe_RGB_groundtruth', epochL=opt.test_fre,epochR=epoch,add=opt.test_fre
            , batchsize=opt.testbatchSize, gpu_ids=opt.gpu_ids,probe_gray=opt.dim_X==1
            ,save_dir_name=save_dir_name['rgb'])


        print('fusion')


        recognizer_fusion.run(train_step=opt.train_step+domain, epochL=opt.test_fre,epochR=epochR,add=opt.test_fre, \
            original_data=original_data, batchsize=opt.testbatchSize, gpu_ids=opt.gpu_ids
            ,save_dir_name='fusion_'+save_dir_name['d']+"_and_"+save_dir_name['rgb'])





        maecaler.read_gallery(dataroot=opt.dataroot, dataset_ids = dataset_ids, reverse_option=False, use_record=False,typ='RGBD',gallery_gray=opt.dim_Y==1)  #target_dataset_ids

        maecaler.run(istarget= (domain=='_target'), name=opt.name,train_step=opt.train_step+domain, epochL=opt.test_fre,epochR=epochR,add=opt.test_fre, \
            typ='RGBD', original_data=original_data, batchsize=opt.testbatchSize, gpu_ids=opt.gpu_ids,probe_gray=opt.dim_Y==1)

        

    elif domain=='_all':

        print('test all')

        des_dir_source = os.path.join(opt.results_dir, opt.name, opt.train_step+'_source','%s_%s' % ("test", epoch),'images')  
        des_dir_target = os.path.join(opt.results_dir, opt.name, opt.train_step+'_target','%s_%s' % ("test", epoch),'images')  


        tmp_imageY=[]
        for x in dataset_ids:
            datasetname=dataset_map[x]
            gallery_list_loc=   dataset_map['gallery'][x]['D']  #_fake_Y_source.png

            f=open(gallery_list_loc,'r')
            for line in f.readlines():

                filename=line.strip().split(' ')[0].split('/')[2].split('.')[0]+'_fake_Y_source.png'  
                tmp_loc=os.path.join( des_dir_source ,filename)
                if not os.path.exists(tmp_loc):
                    filename=line.strip().split(' ')[0].split('/')[2].split('.')[0]+'_fake_Y_target.png' 
                    tmp_loc=os.path.join( des_dir_target ,filename)

                assert ( os.path.exists(tmp_loc) )

                tmp_imageY.append( tmp_loc   )
            f.close()
            

        tmp_imageY.sort()

        print('detph')

        recognizer_fusion.recognizer_d.read_gallery(dataroot=opt.dataroot,dataset_ids=dataset_ids,gallery_type='gallery_Depth_estimated',
            rcg_model_name=rcg_model_name['d'],
            read_imageYs=tmp_imageY,nobackgrounds=True, reverse_option=False, use_record=False,gallery_gray= opt.dim_Y==1) 


       
        tmp_dir=os.path.join(opt.results_dir, opt.name, opt.train_step+"_source",'%s_%s' % ("test", epoch),'images')
        original_data.append( tmp_dir )
        tmp_dir=os.path.join(opt.results_dir, opt.name, opt.train_step+"_target",'%s_%s' % ("test", epoch),'images')
        original_data.append( tmp_dir )
        original_data.sort()

        recognizer_fusion.recognizer_d.run(train_step=opt.train_step+domain, probe_type='probe_depth_estimated', epochL=opt.test_fre,epochR=epochR,add=opt.test_fre, \
            original_data=original_data, batchsize=opt.testbatchSize, gpu_ids=opt.gpu_ids,probe_gray=opt.dim_Y==1
            ,save_dir_name=save_dir_name['d'])

                
        print('rgb')



        recognizer_fusion.recognizer_rgb.read_gallery(dataroot=opt.dataroot,dataset_ids=dataset_ids, gallery_type='gallery_RGB_groundtruth'
            ,rcg_model_name=rcg_model_name['rgb']
            ,nobackgrounds=True, reverse_option=True, use_record=False,gallery_gray= opt.dim_X==1)  #target_dataset_ids

        recognizer_fusion.recognizer_rgb.run(train_step=opt.train_step+domain, probe_type='probe_RGB_groundtruth', epochL=opt.test_fre,epochR=epoch,add=opt.test_fre
            , batchsize=opt.testbatchSize, gpu_ids=opt.gpu_ids,probe_gray=opt.dim_X==1
            ,save_dir_name=save_dir_name['rgb'])


        print('fusion')


        recognizer_fusion.run(train_step=opt.train_step+domain, epochL=opt.test_fre,epochR=epochR,add=opt.test_fre, \
            original_data=original_data, batchsize=opt.testbatchSize, gpu_ids=opt.gpu_ids
            ,save_dir_name='fusion_'+save_dir_name['d']+"_and_"+save_dir_name['rgb'])


             
        maecaler.read_gallery(dataroot=opt.dataroot, dataset_ids = dataset_ids, reverse_option=False, use_record=False,typ='RGBD',gallery_gray=opt.dim_Y==1)  #target_dataset_ids

        maecaler.run(istarget= (domain=='_target'), name=opt.name,train_step=opt.train_step+domain, epochL=opt.test_fre,epochR=epochR,add=opt.test_fre, \
            typ='RGBD', original_data=original_data, batchsize=opt.testbatchSize, gpu_ids=opt.gpu_ids,probe_gray=opt.dim_Y==1)
        

    opt.phase='train'





if __name__ == '__main__':



    opt = TrainOptions().parse()


    if 'no_target_imageYs' in opt and  opt['no_target_imageYs']:
        no_target_imageYs=True
    else:
        no_target_imageYs=False


    face_dataset = tsfUnpairedDataset(dataroot=opt.dataroot,dataset_ids=opt.dataset_ids,source_dataset_ids=opt.source_dataset_ids \
        ,phase='train',img_size=opt.img_size,opt=opt,aug=opt.aug, no_target_imageYs=no_target_imageYs)
    trainloader = torch.utils.data.DataLoader(face_dataset, batch_size=opt.batchSize,pin_memory=True, shuffle=True, num_workers=opt.num_workers)

    opt.num_identities = face_dataset.num_identities()
    len_face_dataset=len(face_dataset)



    print(len(face_dataset))
    sutil.log(len(face_dataset),'main')
    sutil.log('num_identities','main')
    sutil.log(opt.num_identities,'main')
    sutil.log('num_identities record','main')



    from tools.recognizer import Recognizer 
    from tools.recognizer_fusion import FusionRecognizer 
    recognizer_d=Recognizer(opt)

    recognizer_rgb=Recognizer(opt)
    recognizer_rgb.init(opt,recog_item='recog_state_dict_rgb_for_fusion',rcg_func=opt.rgb_rcg_func_rgb_for_fusion,img_dim=opt.dim_X)

    recognizer_fusion=FusionRecognizer(recognizer_d,recognizer_rgb)


    test_dataset_source = PairedDataset(dataroot=opt.dataroot,dataset_ids=opt.test_source_dataset_ids \
        ,phase='test',img_size=opt.img_size,opt=opt,aug=False ,pre='source_')
    testloader_source = torch.utils.data.DataLoader(test_dataset_source, batch_size=opt.testbatchSize,pin_memory=True, shuffle=False, num_workers=1)


    test_dataset = PairedDataset(dataroot=opt.dataroot,dataset_ids=opt.test_dataset_ids \
        ,phase='test',img_size=opt.img_size,opt=opt,aug=False ,pre='target_')
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.testbatchSize,pin_memory=True, shuffle=False, num_workers=1)




    val_dataset = tsfUnpairedDataset(dataroot=opt.dataroot,dataset_ids=opt.dataset_ids,source_dataset_ids=opt.source_dataset_ids \
        ,phase='train',img_size=opt.img_size,opt=opt,aug=False, no_target_imageYs=no_target_imageYs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batchSize,pin_memory=True, shuffle=False, num_workers=opt.num_workers)

    t_dataset = tsfUnpairedDataset(dataroot=opt.dataroot,dataset_ids=opt.dataset_ids,source_dataset_ids=opt.source_dataset_ids \
        ,phase='test',img_size=opt.img_size,opt=opt,aug=False, no_target_imageYs=no_target_imageYs)
    t_loader = torch.utils.data.DataLoader(t_dataset, batch_size=opt.batchSize,pin_memory=True, shuffle=False, num_workers=opt.num_workers)






    testdata=None
    for i, data in enumerate(t_loader): 
        testdata=data
        break

    valdata=None
    for i, data in enumerate(val_loader): 
        valdata=data
        break
  

    model = create_model(opt)
    recognizer_d.replacemodel(model.feature_extraction_model)



    if 'test_at_beginning' in opt and opt.test_at_beginning:

        model.evalstate()

        epoch=opt.start_epoch

        epoch=0

        tf_test(model,opt,epoch+1,epoch+1,dataset_ids=opt.test_dataset_ids,domain='_target',recognizer_fusion=recognizer_fusion,testloader=testloader,
                    rcg_model_name= {'rgb': opt.recog_state_dict_rgb_for_fusion.name, 'd': '{}_{}_finetuned_{}epochs_'.format(opt.name, opt.train_step, epoch+1) +opt.recog_state_dict.name },
                    save_dir_name=  {'rgb':opt.recog_state_dict_rgb_for_fusion.name, 'd':'finetuned_'+opt.recog_state_dict.name})

        tf_test(model,opt,epoch+1,epoch+1,dataset_ids=opt.test_source_dataset_ids,domain='_source',recognizer_fusion=recognizer_fusion,testloader=testloader_source,
            rcg_model_name= {'rgb': opt.recog_state_dict_rgb_for_fusion.name, 'd': '{}_{}_finetuned_{}epochs_'.format(opt.name, opt.train_step, epoch+1) +opt.recog_state_dict.name },
            save_dir_name=  {'rgb':opt.recog_state_dict_rgb_for_fusion.name, 'd':'finetuned_'+opt.recog_state_dict.name})


        tf_test(model,opt,epoch+1,epoch+1,dataset_ids=opt.test_source_dataset_ids+opt.test_dataset_ids,domain='_all',recognizer_fusion=recognizer_fusion, 
            rcg_model_name= {'rgb': opt.recog_state_dict_rgb_for_fusion.name, 'd': '{}_{}_finetuned_{}epochs_'.format(opt.name, opt.train_step, epoch+1) +opt.recog_state_dict.name },
            save_dir_name=  {'rgb':opt.recog_state_dict_rgb_for_fusion.name, 'd':'finetuned_'+opt.recog_state_dict.name} )




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

        recognizer_d.replacemodel(model.feature_extraction_model)  

        if (epoch+1)%opt.test_fre == 0 :


            tf_test(model,opt,epoch+1,epoch+1,dataset_ids=opt.test_dataset_ids,domain='_target',recognizer_fusion=recognizer_fusion,testloader=testloader,
                rcg_model_name= {'rgb': opt.recog_state_dict_rgb_for_fusion.name, 'd': '{}_{}_finetuned_{}epochs_'.format(opt.name, opt.train_step, epoch+1) +opt.recog_state_dict.name },
                save_dir_name=  {'rgb':opt.recog_state_dict_rgb_for_fusion.name, 'd':'finetuned_'+opt.recog_state_dict.name })

            tf_test(model,opt,epoch+1,epoch+1,dataset_ids=opt.test_source_dataset_ids,domain='_source',recognizer_fusion=recognizer_fusion,testloader=testloader_source,
                rcg_model_name= {'rgb': opt.recog_state_dict_rgb_for_fusion.name, 'd': '{}_{}_finetuned_{}epochs_'.format(opt.name, opt.train_step, epoch+1) +opt.recog_state_dict.name },
                save_dir_name=  {'rgb':opt.recog_state_dict_rgb_for_fusion.name, 'd':'finetuned_'+opt.recog_state_dict.name })


            tf_test(model,opt,epoch+1,epoch+1,dataset_ids=opt.test_source_dataset_ids+opt.test_dataset_ids,domain='_all',recognizer_fusion=recognizer_fusion, 
                rcg_model_name= {'rgb': opt.recog_state_dict_rgb_for_fusion.name, 'd': '{}_{}_finetuned_{}epochs_'.format(opt.name, opt.train_step, epoch+1) +opt.recog_state_dict.name },
                save_dir_name=  {'rgb':opt.recog_state_dict_rgb_for_fusion.name, 'd':'finetuned_'+opt.recog_state_dict.name } )


        model.update_learning_rate(epoch+1)



    model.saveall(opt.max_epoch+1)


      
