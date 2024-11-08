# FusionGCN (Expert Systems With Applications, 2025)：
  
#### 欢迎参考和引用我们的工作(Welcome to refer to and cite our work)  
#### 文章发表在Expert Systems with Applications Volume 262, 1 March 2025上
#### Code for paper [“FusionGCN: Multi-focus image fusion using superpixel features generation GCN and pixel-level feature reconstruction CNN”](https://www.sciencedirect.com/science/article/pii/S0957417424025326).  
  
# Delivery timeline  
-   Received 27 July 2024;
-   Received in revised form 27 October 2024;
-   Accepted 29 October 2024;
-   Available online 7 November 2024
  
# Highlights  
-   To our knowledge, this is the first time that GCN has been used to solve the problem of multi-focus image fusion.
-   Compared with existing methods, FusionGCN provides a new solution for multi-focus image fusion tasks running on devices with limited computing resources.
-   FusionGCN has achieved good results on multiple datasets.
  
# Reference information  
```  
@article{Ouyang2025FusionGCN,
  title={FusionGCN: Multi-focus image fusion using superpixel features generation GCN and pixel-level feature reconstruction CNN},  
  author={Yuncan Ouyang and Hao Zhai and Hanyue Hu and Xiaohang Li and Zhi Zeng},  
  journal={Expert Systems with Applications},  
  pages={125665},  
  year={2025},  
  publisher={Elsevier}  
}
```
  
# Dependencies  
-   python >= 3.6
-   pytorch >= 1.5.0
-   CUDA >= 12.0
-   train.py -- 训练我们的网络(Train our network)
-   inference.py -- 利用训练好的网络参数进行图像融合(Fusion images through network)

# How to use
-   代码提供Lytro, MFFW, MFI-WHU, GrayScale和Low-Light5个数据集进行测试
-   The code provides five datasets for testing: Lytro, MFFW, MFI-WHU, GrayScale, and Low-Light
-   仅需替换inference.py中的数据集名称即可一键运行
-   Simply replace the dataset name in the inference.py to run with one click
  
# Results
The output results will be stored in `./Result/(datasetname)`.

Our results in Lytro, MFFW, MFI-WHU, GrayScale datasets can be downloaded.
  
# Contact information  
E-mail addresses: 2023210516060@stu.cqnu.edu.cn (Y. Ouyang)
