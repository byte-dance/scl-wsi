# Coupling Self-Supervised and Supervised Contrastive Learning for Multiple Classification of Cervical Cytological Whole Slide Images

A three stage framework for Cervical Cytological Whole Slide Image Multi-classification.

## Object Detection and Classification
We use [YOLOv7](https://github.com/WongKinYiu/yolov7) to detect the suspicious cells in the WSI and [EfficientNet](https://github.com/qubvel/efficientnet) to further classify
the suspicious cells.

## Self-supervised learning on suspicious cells
We use [dino](https://github.com/facebookresearch/dino) to train a cell-level feature extractor ([ResNet50](https://arxiv.org/abs/1512.03385)). 

## Supervised Contrastive learning based Whole Slide Image Classification

Before training, each WSI should be saved as a .csv file. Suppose a WSI has 100 cells,
each cell is projected to a 2048-dimension vector by the pretrained ResNet50, the csv file is shape of 100*2048.

To pre-compute targets, please run
```commandline
python target_generation.py
```

To start training, please run 
```commandline
python train.py --epochs 100 --train_set train.csv --val_set valid.csv --model_path <PATH/TO/OUTPUT/DIR>
```
Example train.csv
```text
匹配细胞号,TBS级别,label,path
EXN0030626,ASC-H,3,/home/nas2/path/datasets/dino-r50-e200-cell-feature-data-augmentation/EXN0030626
EXN0051344,LSIL,2,/home/nas2/path/datasets/dino-r50-e200-cell-feature-data-augmentation/EXN0051344
EHG0069357,ASC-US,1,/home/nas2/path/datasets/dino-r50-e200-cell-feature-data-augmentation/EHG0069357
EJZ0033815,ASC-US,1,/home/nas2/path/datasets/dino-r50-e200-cell-feature-data-augmentation/EJZ0033815
EXN0007419,ASC-US,1,/home/nas2/path/datasets/dino-r50-e200-cell-feature-data-augmentation/EXN0007419
EXG0170912,NILM,0,/home/nas2/path/datasets/dino-r50-e200-cell-feature-data-augmentation/EXG0170912
```
