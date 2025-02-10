# FFMT: Unsupervised RGB-D Point Cloud Registration via Fusion Feature Matching with Transformer

## Instructions
This code has been tested on 
- Python 3.8, PyTorch 1.12.1, CUDA 11.3, GeForce RTX 4090

## Requirements
To create a virtual environment and install the required dependences please run:
```shell
git clone https://github.com/yuyuyuyy11/FFMT.git
conda create --name ffmt python=3.8
conda activate ffmt
pip install -r requirements.txt
```
## Make Dataset
You need to download the RGB-D version of 3DMatch dataset and ScanNet dataset in advance.
Details can refer to [URR](https://github.com/mbanani/unsupervisedRR/blob/main/docs/datasets.md).

### 3DMatch
```shell
python create_3dmatch_rgbd_dict.py --data_root 3dmatch_train.pkl train
python create_3dmatch_rgbd_dict.py --data_root 3dmatch_valid.pkl valid
python create_3dmatch_rgbd_dict.py --data_root  3dmatch_test.pkl test
```


### ScanNet
```shell
python create_scannet_dict.py --data_root scannet_train.pkl train
python create_scannet_dict.py --data_root scannet_valid.pkl valid
python create_scannet_dict.py --data_root scannet_test.pkl test 
```