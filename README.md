# MMeViT
This is an official implemetnation of arXiv paper: "[MMeViT: Multi-Modal ensemble ViT for Post-Stroke Rehabilitation Action Recognition](https://arxiv.org/abs/2509.23044)"

## Overview
We propose a multimodal(3-axis accelerometer and gyrometer from IMU sensors, skeleton from RGBD camera) deep learning model specialized for the Human Action Recognition of post-stroke patients. The sensor data are converted into 3-channel 2D images, and the skeleton data is converted Gaussian map to be respectively used as an input of a ViT-based model that classifies the label(action).

## Codes
```
./codes
│      function.py              // functions for train_test.py
│      train_test.py            // train and test MLP head
│  
├─imu
│      function.py              // functions for dealing IMU data
│      model.py                 // part of IMU model in MMeViT 
│      preprocessing.py         // preprocessing functions of IMU data
│      
└─skeleton
        function.py             // functions for dealing skeleton data
        multi_feature.py        // features of skeleton data 
        
```

### Preprocessing
![preprocessing](./figures/preprocessing.png)


### Model 
![model](./figures/model.png) 

## Citations
If you found this repo useful, please consider citing our paper: 
<!-- 나중에수정하기-->
```
@article{kim2025MMeViT,
    title = {MMeViT: Multi-Modal ensemble ViT for Post-Stroke Rehabilitation Action Recognition},
    author = {Kim, Ye-eun and Lim, Suhyeon and Choi, Andrew J.},
    year={2025},
    eprint={2509.23044},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2509.23044},
} 
```
