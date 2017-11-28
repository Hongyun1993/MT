# MT−A Kind of Rapid Training Method by Mixing Data

This repository contains the code for MT introduced in the paper [" MT−A Kind of Rapid Training Method by Mixing Data"]

### Citing MT
If you find MT useful in your research, please consider citing:

	@inproceedings{Hongyun2018MT,
	  title={A Kind of Rapid Training Method by Mixing Data},
	  author={Hongyun Li, Yafeng Yang, Jianlin Zhang, Zhiyong Xu },
	  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
	  year={2018}
	}

Note that we only listed some early implementations here, and all results are obtained on DenseNet-40(k=12). 

## Contents
1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results on CIFAR](#results-on-cifar)
5. [Updates](#updates)


## Introduction
MT is a new training or data compression, it randomly chooses two samples to generate a new one by weighted average method and uses the new samples to train the model.

`<img src="https://cloud.githubusercontent.com/assets/8370623/17981494/f838717a-6ad1-11e6-9391-f0906c80bc1d.jpg" width="480">`

Figure 1: A homotopy sample


`![densenet](https://cloud.githubusercontent.com/assets/8370623/17981496/fa648b32-6ad1-11e6-9625-02fdd72fdcd3.jpg)`
Figure 2: A deep DenseNet with three dense blocks. 


## Usage 
0. Install tensforflow and required dependencies like cuDNN. See the instructions in official website of tensorflow[here](www.tensorflow.org/) for a step-by-step guide.
1. Clone this repo: ```git clone https://github.com/Hongyun1993/MT.git```
2. run the main script

## Results on CIFAR
The table below shows the results of MT on CIFAR datasets. The "+" mark at the end denotes for standard data augmentation (random crop after zero-padding, and horizontal flip). For a DenseNet model, L denotes its depth and k denotes its growth rate. *indicates results run by ourselves.

Model | CIFAR-10 | CIFAR-10+ | CIFAR-100 | CIFAR-100+ 
-------|:-------:|:--------:|:--------:|:--------:|
DenseNet (L=40, k=12) |93.00 |94.76 | 72.45|75.58
DenseNet (L=40, k=12)* |93.21 |94.31 | 72.19|73.32
MT|**93.51** |**94.83** |**72.91** |75.01


## Updates
**11/15/2017:**

1. Add supporting code, so one can simply *git clone* and run.

## Contact
lihongyun1993@qq.com  
yangyafeng17@qq.com  
Any discussions, suggestions and questions are welcome!
