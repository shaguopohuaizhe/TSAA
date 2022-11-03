# Transferable Sparse Adversarial Attack

Pytorch Implementation of our CVPR2022 paper "Transferable Sparse Adversarial Attack".

### Table of Contents  
1) [Dependencies](#Dependencies) <a name="Dependencies"/>
2) [Pretrained-Generators](#Pretrained-Generators) <a name="Pretrained-Generators"/>
3) [Datasets](#Datasets) <a name="Datasets"/>
4) [Training/Eval](#Training)  <a name="Training"/>

## Dependencies
1. Install [pytorch](https://pytorch.org/). This repo is tested with pytorch==1.6.0.
2. Install python packages using following command:
```
pip install -r requirements.txt
```
This repo is tested with python==3.8.5.

## Pretrained-Generators
Download pretrained adversarial generators from [here](https://drive.google.com/drive/folders/1Fo3xGEPadvWnXt0eGEI9qdzBEAoBqjbG?usp=sharing).

Adversarial generators are trained against following two models.
* Inceptionv3
* ResNet50

These models are trained on ImageNet and available in Pytorch. 
  
## Datasets
* Training data:
  * [ImageNet](http://www.image-net.org/) Training Set.
  
* Evaluations data:
  * randomly selected 5k images from [ImageNet](http://www.image-net.org/) Validation Set.
  You can download evaluations data from [here](https://drive.google.com/drive/folders/1z6fMGd-NFvKi1-tVG59ow7ZxHyEGfEGI?usp=sharing).
  
  
## Training
<p align="justify"> Run the following command

```
  python train.py --train_dir [path_to_train] --model_type incv3 --eps 255 --target -1
```
This will start trainig a generator trained on one dataset (--train_dir) against Inceptionv3 (--model_type) under perturbation budget $\ell_\infty$=255 (--eps) in a non-targeted setting (--target).<p>

## Evaluations
<p align="justify"> Run the following command

```
  python eval.py --test_dir [path_to_val] --model_type incv3 --model_t res50 --eps 255 --target 971 --checkpoint [path_to_checkpoint]
```
This will load a generator trained against Inceptionv3 (--model_type) and evaluate clean and adversarial accuracy of ResNet50 (--model_t) under perturbation budget 255 (--eps) in a targeted setting (--target). <p>

## Citation
If you find this repo useful, please cite our paper.
```bibtex
@InProceedings{He_2022_CVPR,
    author    = {He, Ziwen and Wang, Wei and Dong, Jing and Tan, Tieniu},
    title     = {Transferable Sparse Adversarial Attack},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {14963-14972}
}
```







