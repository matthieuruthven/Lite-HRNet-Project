# Lite-HRNet Installation and Method Extension for Compatability with SPEED+ Dataset

[Lite-HRNet](https://github.com/HRNet/Lite-HRNet) is a 2D pose estimation method whose implementation depends on several packages from [OpenMMLab](https://github.com/open-mmlab), notably [MMCV](https://github.com/open-mmlab/mmcv) and [MMPose](https://github.com/open-mmlab/mmpose). Since Lite-HRNet was made publicly available, installation of the method has become more complicated due to major upgrades to MMCV and MMPose. In addition, by default Lite-HRNet is compatible with the [COCO](https://cocodataset.org/#home) and [MPII Human Pose](http://human-pose.mpi-inf.mpg.de/) datasets but not with the [SPEED+](https://zenodo.org/record/5588480) dataset. The method must therefore be extended to make it compatible with the SPEED+ dataset. 

## Lite-HRNet Installation Instructions

### 1. Create conda environment and activate it

```
conda create -n openmmlab python=3.8
conda activate openmmlab
```

### 2. Install PyTorch

Install PyTorch 1.11.0 using following command:
```
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
```
For more information, see the installation instructions on the PyTorch [website](https://pytorch.org/get-started/locally/).

### 3. Install MIM

Install [MIM](https://github.com/open-mmlab/mim), a package management software, using the following command:
```
pip install -U openmim
```

### 4. Install MMCV using MIM

Install version 1.7.0 of the old release of MMCV using the following command:
```
mim install mmcv-full==1.7.0
```
For more information, see the installation instructions [here](https://github.com/open-mmlab/mmcv/tree/1.x).

### 5. Install MMPose using pip

Install version 0.29.0 of the old release of MMPose using the following command:
```
pip install mmpose==0.29.0
```
For more information, see the installation instructions [here](https://mmpose.readthedocs.io/en/0.x/install.html#installation).

### 6. Clone the Lite-HRNet GitHub repository

```
git clone https://github.com/HRNet/Lite-HRNet.git
```

### 7. Install remaining build requirements

```
pip install -r requirements.txt
```

## Extending Lite-HRNet for SPEED+ Dataset Compatibility


### 1. Add custom Python class

For more information, see the instructions [here](https://mmpose.readthedocs.io/en/0.x/tutorials/2_new_dataset.html).

1. Add the `topdown_speedplus_dataset.py` file from this GitHub repository to the `mmpose/datasets/datasets/top_down` folder

2. When running Lite-HRNet for the first time, modify line 15 of the `topdown_speedplus_dataset.py` file in the following way:

```
@DATASETS.register_module(name='TopDownSpeedPlusDataset')
```

3. Update the `__init__.py` file in the `mmpose/datasets/datasets/top_down` and `mmpose/datasets/datasets` folders

### 2. Add custom dataset transformation

For more information, see the instructions [here](https://mmpose.readthedocs.io/en/0.x/tutorials/3_data_pipeline.html).

1. Add the `my_transforms.py` file from this GitHub repository to the `mmpose/datasets/pipelines` folder

2. When running Lite-HRNet for the first time, modify line 13 of the `my_transforms.py` file in the following way:

```
@PIPELINES.register_module(name='TopDownBBoxCrop')
```
3. Update the `__init__.py` file in the `mmpose/datasets/pipelines` folder

### 3. Additional modifications

For code modifications required because of MMCV and MMPose upgrades, see the following files in this GitHub repository:

- *configs/_base_/datasets/speedplus.py*
- *models/backbones/litehrnet.py*
- *tools/train.py*
- *tools/test.py*