#coding: utf-8
import os,logging,sys
import torch,yaml
import importlib,time,shutil
from types import MethodType
import numpy as np
from PIL import Image
import cv2


opt=None



def upload_opt(option):
    global opt
    opt=option

def get_opt():
    return opt


def makedirs(x):
    if not os.path.exists(x):
        os.makedirs(x)



def get_file_name(pth):
    return  os.path.split(pth)[1].split('.')[0]




def get_logger(path,name='mylogger'):

    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)  

    loc=os.path.join(path) 

    f_handler = logging.FileHandler(loc)
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(logging.Formatter(
    fmt="%(asctime)s %(name)s %(filename)s %(message)s",
    datefmt="%Y/%m/%d %X"
    ))


    filter=logging.Filter(name)
    logger.addFilter(filter)
    
    logger.addHandler(f_handler)
    logger.propagate = False

    
    return logger





def find_model_using_name( dirs, model_name ): 

    model_filename = dirs + '.' +model_name   # filename

    modellib = importlib.import_module(model_filename)

    model= None     
    for name, cls in modellib.__dict__.items():
        if name  == model_name:
            model = cls

    if model is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return model


def add_functions(model_ins, dirs ,model_name):
    
    assert not dirs.endswith('/')

    model_filename =    '.'.join(dirs.split('/')) + '.' +model_name   

    functions_py = importlib.import_module(model_filename)


    for name ,val in functions_py.__dict__.items():
        if str(val).startswith('<function'):
            setattr(model_ins,name, MethodType(val,model_ins)  )



def log(x ,name):

    logger = logging.getLogger(name)

    logger.info(x)




def tensor2im(image_tensor, imtype=np.uint8):

    image_numpy = image_tensor[0].cpu().float().numpy() if len(image_tensor.shape) ==4 else  image_tensor.cpu().float().numpy()
    
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0 # HWC

    return image_numpy.astype(imtype)


def tensor2images(image_tensor, imtype=np.uint8):

    assert len(list(image_tensor.size()))==4

    images_numpy = image_tensor.cpu().float().numpy()

    images_numpy = (np.transpose(images_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0 # HWC

    return images_numpy.astype(imtype)


def save_image(image_numpy, image_path ,isgray=False ): 

    #image_numpy: HWC

    print(image_path)

    if image_numpy.shape[2]==1:
        image_numpy=image_numpy.squeeze(2)

    image_pil = Image.fromarray(image_numpy)

    if isgray:
        image_pil=image_pil.convert('L')

    image_pil.save(image_path)

def save_images(visuals,image_paths,des_dir):

    for label, image_numpys in visuals.items():
        for i,image_numpy in enumerate(image_numpys):

            image_path=image_paths[i]
            basename=os.path.split(image_path)[1]
            shortname=os.path.splitext(basename)[0]
            image_name = '%s_%s.png' % (shortname, label)   # (    epoch100_fake_p)
            save_image(image_numpy, os.path.join(des_dir,image_name))



def pytorch_rgb2bgr(x,dim=1):

    (r, g, b) = torch.chunk(x, 3, dim = dim)

    return torch.cat((b, g, r), dim = dim) # convert RGB to BGR 




def transfer_tensor_rgb(x):

    return opt.transfer_tensor_rgb(x)


def transfer_tensor_gray(x):

    return opt.transfer_tensor_gray(x)


def readimg_to_tensor_rgb(loc,color2gray2color=False):

    return opt.readimg_to_tensor_rgb(loc,color2gray2color)


def readimg_to_tensor_gray(loc,color2gray2color=False):

    return opt.readimg_to_tensor_gray(loc,color2gray2color)



class PIL_readimg_to_tensor:

    def __init__(self,img_size,read_dim):

        assert (read_dim in [1,3])
        self.img_size = img_size
        self.read_dim=read_dim


    def __call__(self,loc,color2gray2color=False):


        img = Image.open(loc)

        img= img.convert('RGB') if self.read_dim==3 else   img.convert('L')

        if color2gray2color:
            assert(self.read_dim==3)
            img_gray=img.convert('L')
            img=img_gray.convert('RGB')

        img= transfer_tensor_rgb(img)  if self.read_dim==3 else  transfer_tensor_gray(img)

        return img




def read_yaml(loc):
    
    with open(loc, 'r') as stream:
        config =yaml.load(stream)
    return config


