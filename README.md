# HSI-DHL
The code for `Boosting Hyperspectral Image Classification with Dual Hierarchical Learning`

## Requirements

This tool is compatible with Python 2.7 and Python 3.5+.

It is based on the [PyTorch](http://pytorch.org/) deep learning and GPU computing framework and use the [Visdom](https://github.com/facebookresearch/visdom) visualization server.

## Hyperspectral datasets

Several public hyperspectral datasets are available on the [UPV/EHU](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes) wiki. Users can download those beforehand or let the tool download them. The default dataset folder is `./Datasets/`, although this can be modified at runtime using the `--folder` arg.

At this time, the tool automatically downloads the following public datasets:
  * Pavia University
  * Pavia Center
  * Indian Pines
  * Salinas-A

SalinasA_gt.mat
SalinasA_corrected.mat

The [Data Fusion Contest 2018 hyperspectral dataset]() is also preconfigured, although users need to download it on the [DASE](http://dase.ticinumaerospace.com/) website and store it in the dataset folder under `DFC2018_HSI`.

An example dataset folder has the following structure:
```
Datasets
├── Salinas-A
│   ├── SalinasA_gt.mat
│   └── SalinasA_corrected.mat
├── IndianPines
│   ├── Indian_pines_corrected.mat
│   └── Indian_pines_gt.mat
├── PaviaC
│   ├── Pavia_gt.mat
│   └── Pavia.mat
└── PaviaU
    ├── PaviaU_gt.mat
    └── PaviaU.mat
```

### Adding a new dataset

Adding a custom dataset can be done by modifying the `custom_datasets.py` file. Developers should add a new entry to the `CUSTOM_DATASETS_CONFIG` variable and define a specific data loader for their use case.

## Usage

Run `python main.py --DATASET "KSC" --DHCN_LAYERS 2 --SAMPLE_PERCENTAGE 10 --GPU '0,1,2,3'`.
