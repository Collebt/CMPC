# Learning to Find the Optimal Correspondence between SAR and Optical Image Patches


## Project Page
See it from [CMPC](hhttps://Collebt.github.io/CMPC/)

## Overview
We design a cross-modal visual geo-localization framework to prdict the accurate location of SAR patch from satellite database with cross-modal feature embedding and scene-graph modules.


## Dataset 
The Optical-SAR Patch Correspondence Dataset
Download it from [Google Drive](https://drive.google.com/drive/folders/1KXbHCG47QnmvWWBGzroBZdkIy79tDXZp), and extract it to the directory `data/SN6-CMPC/`.

## Setup
```
conda env create --file environment.yml
conda activate cmpc
```

## Train


- train.py: train the whole paradigm with the embedding model and refine the module, with the dataset named SN6-CMPC. The dataset provides JSON files including the position and raw information of the query/reference images.

- train_emb.py: Only the embedding model is trained for comparative experiments with other methods. 

- train_refine.py: Only train the refinement model and use it after obtaining the trained embedding model and the output features of the images.

Train the whole model
```
python train.py exp_name experiments/same_advwd_gnn_e2e.json
```


## Test
- test.py: meter-level test and cmc plot.
```
python test.py param_path experiments/same_advwd_gnn_e2e.json
```

## BibTeX
```
@ARTICLE{li2023learning,
  author={Li, Haoyuan and Xu, Fang and Yang, Wen and Yu, Huai and Xiang, Yuming and Zhang, Haijian and Xia, Gui-Song},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Learning to Find the Optimal Correspondence Between SAR and Optical Image Patches}, 
  year={2023},
  volume={16},
  number={},
  pages={9816-9830},
  keywords={Optical sensors;Optical imaging;Task analysis;Feature extraction;Visualization;Synthetic aperture radar;Remote sensing;Adversarial training;cross-modal image retrieval;graph neural network;synthetic aperture radar (SAR)},
  doi={10.1109/JSTARS.2023.3324768}}
```
