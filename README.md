# OCT-Multitask
## Multi-Task Joint Segmentation of Retinal Layers and Fluid Lesions in Optical Coherence Tomography ##
## 1. Environment
- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.
## 2. Prepare OCT datasets 
- You can download the datasets from the [Baidu Netdisk](https://pan.baidu.com/s/1SzuMVWN1AteufH0JIj6lhw?pwd=ra2z) (access code: ra2z) and decompress it to ["./datasets"](datasets/). Please go to ["./datasets/README.md"](datasets/README.md) for details. 
#### (1) Run pre-processing step for OCT B-scans
- The pre-processing step contains OCT flattening and intensity normalization.
- An example of pre-processing on Duke dataset:
```bash
python terminal.py --preprocessing --dataset duke
```
- If you want to do this process on other dataset, you can set the parameter value of dataset to "yifuyuan" or "retouch". Specially, If you want to do this on Retouch dataset, you have to set a specific device type of --oct_device, including "Cirrus", "Spectralis" and "Topcon".
```bash
python terminal.py --preprocessing --dataset retouch --oct_device Cirrus
```
#### (2) Split dataset into train and test datasets 
- The clinical and Duke dataset are automatically splited into training set and test set based on the index json files in ["./datasets/yifuyuan/"](datasets/yifuyuan/) and ["./datasets/duke/"](datasets/duke/). To do these, you can type the following:
```bash
python terminal.py --split_dataset --dataset yifuyuan
```
```bash
python terminal.py --split_dataset --dataset duke
```
## 3. Train and test the proposed model on our clinical and Duke dataset
- If you have done the above two steps for specific dataset, you can type the training and testing commands on the corresponding dataset given below:
#### (1) Our clinical dataset
```bash
python terminal.py --training --testing --dataset yifuyuan --backbone resnetv2 --epoch 65
```
#### (2) Duke dataset
```bash
python terminal.py --training --testing --dataset duke --backbone resnetv2 --epoch 65
```
## 4. Semi-supervised learning on Retouch dataset
- To do this, You must first get a pre-training weights .pth file by training on a fully annotated dataset as a source dataset, like our clinical dataset.
- After getting pre-training weights, you can generate pseudo labels of layer and fluid on Retouch Cirrus/Spectralis/Topcon dataset by using the following command (in the case of Cirrus):
```bash
python terminal.py --testing --generate_pseudo --dataset retouch --oct_device Cirrus --backbone resnetv2 --pretrain_path ./datasets/yifuyuan/result/yifuyuan_resnetv2_seed_8830/weights_final.pth
```
- where pretrain_path is the folder path of pre-trained weights.
- Then, if you need to perform 6-fold cross validation on Retouch Cirrus/Spectralis/Topcon dataset, you need to execute the following two lines of commands in sequence, to separate the dataset and train the model (in the case of Cirrus):
```bash
python terminal.py --split_cross_valid --dataset retouch --oct_device Cirrus --k 6
```
```bash
python terminal.py --cross_valid --training --testing --dataset retouch --oct_device Cirrus --backbone resnetv2 --k 6 --epoch 25 --pretrain_path ./datasets/yifuyuan/result/yifuyuan_resnetv2_seed_8830/weights_final.pth
```
- Finally, if you want to verify the performance of semi-supervised model on the source dataset (our clinical dataset), you can execute the following two commands to merge the training sets of source and target dataset, then train the model on this merged training set and test it on the test set of the source data set:
```bash
python terminal.py --merge_dataset --dataset retouch --oct_device Cirrus
```
```bash
python terminal.py --training --testing --dataset retouch --oct_device Cirrus --backbone resnetv2 --epoch 25 --pretrain_path ./datasets/yifuyuan/result/yifuyuan_resnetv2_seed_8830/weights_final.pth
```
