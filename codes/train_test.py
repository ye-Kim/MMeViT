import torch 
import function as mmvvF

import os
import argparse
import yaml


os.environ["CUDA_VISIBLE_DEVICES"]="1"

torch.manual_seed(443)
torch.cuda.manual_seed(443)
 
parse = argparse.ArgumentParser()
parse.add_argument("--config", type=str, default="./NRC_mmVViT.yaml")

argpars = parse.parse_args()
#load config yaml

with open(argpars.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


# load imu model
IMU = mmvvF.ImuWrapper(config)
_, imu_loader = IMU.load_Dataset()
window_size, num_channels, data_depth = imu_loader.dataset[0][0].shape
imu_model = IMU.load_Model(window_size=window_size, num_channels=num_channels, data_depth=data_depth)

# load skeleton model
SKELETON = mmvvF.SkeletonWrapper(config)
skeleton_Dataset = SKELETON.load_Dataset()
skeleton_model = SKELETON.load_Model()


# train mlp 
INFERENCE = mmvvF.ResultWrapper(config)

_, imu_trainloader = IMU.load_Dataset(train=True)
skeleton_trainset = SKELETON.load_Dataset(train=True)

mlp = mmvvF.mlp_head(config['model']['skeleton']['data_emb_size'], config['n_classes'])
INFERENCE.mlp_train_and_save(imu_model, imu_trainloader, imu_loader, skeleton_model, skeleton_trainset, skeleton_Dataset, mlp)

loss, acc, cm = INFERENCE.get_inference(imu_model, imu_loader, skeleton_model, skeleton_Dataset)

# mlp = torch.load(config['model']['mlp']['dir']) # load mlp 
# loss, acc, cm = INFERENCE.get_inference(imu_model, imu_loader, skeleton_model, skeleton_Loader)

# save confusion matrix
print(cm)
INFERENCE.show_confusion_matrix(cm)
