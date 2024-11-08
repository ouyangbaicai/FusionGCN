# FusionGCN (Expert Systems With Applications, 2025)：
Code for paper [FusionGCN: Multi-focus image fusion using superpixel features generation GCN and pixel-level feature reconstruction CNN](https://www.sciencedirect.com/science/article/pii/S0957417424025326).  
你可以在此处下载所有代码(You can download all the code in this branch)  
  
Train.py -- 训练我们的网络（Train our network）  
inference.py -- 利用训练好的网络参数进行图像融合（Fusion images through network）

# -------------------Reference information-------------------  
如果我们的工作对您有所启发，欢迎引用以下信息。  
If this work is helpful to you, please citing our work as follows:  
  
```  
@article{OUYANG2025125665,
  title={FusionGCN: Multi-focus image fusion using superpixel features generation GCN and pixel-level feature reconstruction CNN},  
  author={Yuncan Ouyang and Hao Zhai and Hanyue Hu and Xiaohang Li and Zhi Zeng},  
  journal={Expert Systems with Applications},  
  pages={125665},  
  year={2025},  
  publisher={Elsevier}  
}
```
# ----------------------HighLight----------------------  
· To our knowledge, this is the first time that GCN has been used to solve the problem of multi-focus image fusion. In FusionGCN, we proposed a clever approach that combines block segmentation with pixel optimization, allowing the decision map to transition from coarse to fine, reducing the difficulty of network inference.  
· We have carefully designed a superpixel-based graph decoder and a pixel-based CNN basic extraction block, enabling GCN and CNN to cooperate in the same network, alleviating the incompatibility between non-Euclidean spatial data and Euclidean spatial data.  
· Thanks to the automatic clustering feature of GCN on features, FusionGCN only requires 0.13M parameters and performs 50.76G floating-point operations per second when processing a pair of 520 × 520 multi-focus images, even without using any methods to reduce network parameters. This provides a new solution for running multi-focus image fusion tasks on computationally limited devices.  
· In order to effectively evaluate the efficiency of each method, we have proposed a novel evaluation method based on objective evaluation indicators, making the efficiency assessment results comparable.  
