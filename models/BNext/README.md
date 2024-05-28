# Join The High Accuracy Club on ImageNet With A Binary Neural Network Ticket

This is the official code of the paper [Guo, N., Bethge, J., Meinel, C. and Yang, H., 2022. Join the High Accuracy Club on ImageNet with A Binary Neural Network Ticket. arXiv preprint arXiv:2211.12933.](https://arxiv.org/abs/2211.12933).
### 1.Introduction

Binary neural networks are the extreme case of network quantization, which has long been thought of as a potential edge machine learning solution. However, the significant accuracy gap to the full-precision counterparts restricts their creative potential for mobile applications. In this work, we revisit the potential of binary neural networks and focus on a compelling but unanswered problem: how can a binary neural network achieve the crucial accuracy level (e.g., 80%) on ILSVRC-2012 ImageNet? We achieve this goal by enhancing the optimization process from three complementary perspectives: (1) We design a novel binary architecture BNext based on a comprehensive study of binary architectures and their optimization process. (2) We propose a novel knowledge-distillation technique to alleviate the counter-intuitive overfitting problem observed when attempting to train extremely accurate binary models. (3) We analyze the data augmentation pipeline for binary networks and modernize it with up-to-date techniques from full-precision models. The evaluation results on ImageNet show that BNext, for the first time, pushes the binary model accuracy boundary to 80.57% and significantly outperforms all the existing binary networks.

![Pipeline](https://user-images.githubusercontent.com/24189567/205317106-9a2448f7-116b-4f89-bdfb-c2d148287f52.jpg)
* **Figure**: The architecture of our design, constructed based on an optimized MobileNet backbone and the proposed modules. “Processor” is the core binary convolution module, enhanced using channel-wise mix-to-binary attention branch, and “BN” represents batch normalization layer. The basic block is consisted of an element attention module and a binary feed forward network.

![Convolution Comparison](https://user-images.githubusercontent.com/24189567/204559496-1729c13d-4149-43b5-b674-d0e3df81a72a.jpg)
* **Figure**: Convolution module comparison. a) is the basic module of XNOR Net. b) is the basic module of Real2Binary Net. c) is the core convolution processor in our design with adaptive mix2binary attention.

## 2.Installation

To build the environment for this project on your system, run the following command:
```
sudo pip install -r requirements.txt
```

This will install all the packages listed in the requirements.txt file.


## 3.Pretrained Models and Comparison With Existing Designs
#### Pretrained Model
|Method |BOPs (G)|Binarization Degree (BOPs/(64*OPs))|Top-1 Acc  |Pretrained Models| 
|:----:    | :---: | :---: | :---:  | :---:               |
|BNext-T| 0.077 |97.81% |72.4 % |  [BNext-T](https://drive.google.com/file/d/1CJ0XOEhoHuNe-tDYJaAOd1j4YyNXuyas/view?usp=sharing)                  |  
|BNext-S| 0.172 |98.47% |76.1 % |  [BNext-S](https://drive.google.com/file/d/1NcVM5Qb1K9Oq_sjEA1lGtp7kVbsfTLsa/view?usp=sharing)                  |
|BNext-M| 0.317 |99.02% |78.3 % |  [BNext-M](https://drive.google.com/file/d/1xyKnA6SsG4ZpguNQQrB6Yz-J5dzXYfKE/view?usp=sharing)                  |
|BNext-L| 0.819 |99.49% |80.6 % |  [BNext-L](https://drive.google.com/file/d/1XGKcX2Zl_fIU9wPBDjTTxstOBfwQH8xc/view?usp=sharing)                  |

#### Comparison With Existing Designs
Methods | with BN | with PReLU | with SE | Quantization | KD |Degree of Binarization (BOPs/(64*Ops))
:----: | :---: | :----: | :----: | :----: | :----: | :-----: 
BNN | Yes |  |  |  | Yes |18.07%
XNOR-Net | Yes |  |  |  | Yes |18.07%
BiRealNet-18 | Yes | Yes |  |  | Yes | 17.86%
MeiliusNet-18 | Yes |  |  |  | Yes |16.76%
Real2BinaryNet | Yes | Yes | Yes |  | Yes |14.34%
ReActNet-BiR18 | Yes | Yes |  |  | Yes |13.89%
ReActNet-A | Yes | Yes |  |  | Yes |86.50%
PokeBNN-2.0x | Yes | Yes | Yes | Yes | Yes| 97.15%
BNext-18 | Yes | Yes | Yes | Yes | Yes| 61.04%
BNext-T (ours) | Yes | Yes | Yes | Yes | Yes| 97.81%
BNext-S (ours) | Yes | Yes | Yes | Yes | Yes| 98.47%
BNext-M (ours) | Yes | Yes | Yes | Yes | Yes| 99.02%
BNext-L (ours) | Yes | Yes | Yes | Yes | Yes|99.49%

## 4.Training Procedure
![Training Procedure](https://user-images.githubusercontent.com/24189567/204558527-04de1a26-bfce-4a16-87f9-f781b13988f7.jpg)
* **Figure**: The loss curve, accuracy curve and temperature curve during the optimization process 

### 5. Feature Visualization
![Feature_Visualization_BNext_Tiny](https://user-images.githubusercontent.com/24189567/205326008-fde4e29b-e52a-4a90-81f9-88a45e736c8e.jpg)
* **Figure**: Visualizing the diversity of binary features in BNext-T model. The input image is resized as 1024x1024.

### 6. Reference
If you find our code useful for your research, please cite our paper as follows:
```
@article{guo2022join,
  title={Join the High Accuracy Club on ImageNet with A Binary Neural Network Ticket},
  author={Guo, Nianhui and Bethge, Joseph and Meinel, Christoph and Yang, Haojin},
  journal={arXiv preprint arXiv:2211.12933},
  year={2022}
}
```
