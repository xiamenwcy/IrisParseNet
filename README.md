# IrisParseNet
## Joint Iris Segmentation and Localization Using Deep Multi-task Learning Framework

Created by [Caiyong Wang](http://wangcaiyong.com/) at Institute of Automation Chinese Academy of Sciences (**CASIA**)

<img src='compare.png' width="820px">

As shown in the figure above, we present a novel deep learning based joint iris segmentation and localization method, named as **IrisParseNet**. Unlike the former many FCN based iris segmentation methods, this method not only realizes accurate iris sgementation, but also solves the most import problem in the iris preprocessing step, namely iris localization. Besides, this method achieves state-of-the-art performances on various benchmarks. To promote the research on iris preprocessing (especially iris segmentation and iris localization), we have made our ground-truth annotations, annotation codes and evaluation protocols freely available to the community. 

## Citation
If you use this code or ground-truth annotations for your research, please cite our papers.

```
@article{casiairis2019,
  title={Joint Iris Segmentation and Localization Using Deep Multi-task Learning Framework},
  author={Caiyong Wang and Yuhao Zhu and Yunfan Liu and Ran He and Zhenan Sun},
  journal={arXiv preprint abs/1901.11195},
  year={2019}
}
```
## Prerequisites
- [Halcon](https://www.mvtec.com/products/halcon/) 10.0/13.0 or above  for **labeling iris inner/outer boundary**
- matlab R2016a (other versions are ok) for **evaluating the performance of the proposed method**

## Getting Started

### Annotation codes
we use the interactive development environment (HDevelop) provided by the machine vision software, i.e. MVTec Halcon. Before labeling, you need to install Halcon software. Halcon is a paid software, but it allows to try out for free, please refer to the page:
https://www.mvtec.com/products/halcon/now/.

Our halcon based annotation codes can be found in **code/annotation**. The code can help us to label
iris inner/outer bounadry and output a variety of kinds of annatation results as much as possible.

### Annotation ground truths  
A large benchmark database with ground truths will be released in the future!

### Evaluation protocols
The iris segmentation and localization evaluation codes are provided. During realizing the 
evaluation protocols, we've referenced a lot of open source codes. Here, we thank them, especially
[USIT Iris Toolkit v2](http://www.wavelab.at/sources/Rathgeb16a/), [TVM-iris segmentation](https://www4.comp.polyu.edu.hk/~csajaykr/tvmiris.htm), [GlaS Challenge Contest](https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/evaluation/).

Please read our paper for detailes. The evaluation codes can be found in **code/evaluation**.

## Reference 
[1] Zhao, Zijing, and Ajay Kumar. "An Accurate Iris Segmentation Framework Under Relaxed Imaging Constraints Using Total Variation Model." international conference on computer vision (2015): 3828-3836.

[2] Liu, Nianfeng, et al. "Accurate iris segmentation in non-cooperative environments using fully convolutional networks." Biometrics (ICB), 2016 International Conference on. IEEE, 2016.

[3] Hu, Yang, Konstantinos Sirlantzis, and Gareth Howells. "Improving colour iris segmentation using a model selection technique." Pattern Recognition Letters 57 (2015): 24-32.

## Questions
Please contact wangcaiyong2017@ia.ac.cn
  
  
 


  
