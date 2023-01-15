# **CMOS-GAN: Semi-supervised Generative Adversarial Model for Cross-Modality Face Image Synthesis**

We provide PyTorch implementations for CMOS-GAN [(Paper)](<https://ieeexplore.ieee.org/document/9975261>). This PyTorch implementation produces results comparable to our original results reported in our paper. We will continue optimize the code to fix any issues reported. 



If you use this code for your research, please cite:

S. Yu, H. Han, S. Shan and X. Chen, "CMOS-GAN: Semi-Supervised Generative Adversarial Model for Cross-Modality Face Image Synthesis," in IEEE Transactions on Image Processing, vol. 32, pp. 144-158, 2023, doi: 10.1109/TIP.2022.3226413.



## Prerequisites

- Linux or macOS
- Python 3 (We used python 3.7/3.5)
- CPU or NVIDIA GPU + CUDA  (We used cuda 11.1/10.2/10.0)



## Getting Started

### Installation

- Clone this repo:

```
git clone  https://github.com/skgyu/CMOS-GAN
cd CMOS-GAN
cd CMOS-GAN_code_refactor
```

- Install [PyTorch](http://pytorch.org/) 0.4+ (we use PyTorch 1.7.1/1.8.2/1.2.0) and other dependencies (e.g., torchvision and  [visdom](https://github.com/facebookresearch/visdom)).



### Run CMOS-GAN 


- Download the pre-trained RGB face recognition network.
  - We previously downloaded a pre-trained face recognition network from <https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/>, but this model maybe no longer publicly available.
  - So we suggest use [Our Pretrained models](https://drive.google.com/file/d/1BEYaFX_kW6pWTkcNiwRg5vGereXvtTVw/view?usp=share_link) for PyTorch are converted from [Caffe models](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) authors of [1] provide.
  - Put the downloaded model (named 'ResNet50Backbone.pth') at 'CMOS-GAN_code_refactor/pretrained_model/ResNet50Backbone.pth'.



- sketch-to-photo synthesis

  - We download [CUFS](<http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html>)  [CUFSF](<http://mmlab.ie.cuhk.edu.hk/archive/cufsf/>) [3-4] dataset. We use CUFS as the paired data $S_p$, and use CUFSF as the unpaired data $S_u$.

  - We performed face alignment, cropping and augmentation. Our processed data can be downloaded [here](https://drive.google.com/file/d/1Fk5gdC9bpUXDdMTa6GTX-wE22JHfQ9yE/view?usp=share_link). Put the downloaded dataset folders (named 'AUG_3_9_AR', 'AUG_3_9_CUFSF', 'AUG_3_9_CUHK', and 'AUG_3_9_XM2VTS') at 'CMOS-GAN/dataset/Viewed/'. The complete relative paths are as follows.

    - 'CMOS-GAN/dataset/Viewed/AUG_3_9_AR'
    - 'CMOS-GAN/dataset/Viewed/AUG_3_9_CUFSF'
    - 'CMOS-GAN/dataset/Viewed/AUG_3_9_CUHK'
    - 'CMOS-GAN/dataset/Viewed/AUG_3_9_XM2VTS'

  - (Optional) If you want to perform face recognition using the 10,000 background images like the experiment in our paper, you need to download MORPH [2] to build the background set of the gallery, which contains 10000 RGB images. We aligned and cropped these images in the same way as for CUFS and CUFSF. Place the folder of gallery set (named '10000backgrounds') at 'CMOS-GAN/dataset/additional/10000backgrounds'. 

  - Run the following script to start training and testing. 

    ```
    bash script/S2P_CUFS_CUFSF/S2P_CUFS_CUFSF.sh
    ```

    

  - If you do not want to use the 10,000 background images when performing face recognition, you can run the following script to start training and testing. 

    ```
    bash script/S2P_CUFS_CUFSF_nobackgrounds/S2P_CUFS_CUFSF_nobackgrounds.sh
    ```

    

  - If you do not want to perform face recognition, you can run the following scripts to start training and testing. 	

    ```	
    bash script/S2P_CUFS_CUFSF/S2P_CUFS_CUFSF_no_recognition.sh
    ```

    



Similarly, NIR-to-VIS synthesis and RGB-to-depth synthesis can be performed using the following scripts.

- NIR-to-VIS synthesis

  - VIPLMumoFace3K

    ```
    bash script/RGBD_VIPLMumoFace3K/RGBD_VIPLMumoFace3K.sh
    ```




- RGB-to-depth synthesis

  - RealSenseII

    ```
    bash script/RGBD_RealSenseII/RGBD_RealSenseII.sh
    ```

  - BUAA

    ```
    bash script/RGBD_BUAA/RGBD_BUAA.sh
    ```

  - VIPLMumoFace3K

    ```
    bash script/RGBD_VIPLMumoFace3K/RGBD_VIPLMumoFace3K.sh
    ```

  - RealSenseII_VIPLMumoFace3K

    ```
    bash script/RGBD_RealSenseII_VIPLMumoFace3K/RGBD_RealSenseII_VIPLMumoFace3K.sh
    ```







## References

[1] Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman

[VGGFace2: A dataset for recognising faces across pose and age ](https://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/) 

International Conference on Automatic Face and Gesture Recognition, 2018

[2] K. Ricanek and T. Tesafaye, “Morph: A longitudinal image database of
normal adult age-progression,” in IEEE FG, pp. 341–345, 2006.

[3] X. Wang and X. Tang. Face Photo-Sketch Synthesis and Recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI)*, vol. 31, no. 11, pages 1955-1967, 2009.

[4] W. Zhang, X. Wang and X. Tang. Coupled Information-Theoretic Encoding for Face Photo-Sketch Recognition. *Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2011.