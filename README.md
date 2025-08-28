# DAMFusion: A Degradation-Aware Adaptive Multi-Task Fusion Framework for Robust Image Integration  

> This repository provides a non-official and simplified version of our implementation, intended solely for academic communication and preliminary understanding.

     
> To protect the originality of our work and ensure compliance with data and intellectual property considerations, the **complete codebase will be released after the paper is officially published**.

     
> We appreciate your understanding and support!

## 🔹 Framework Overview

The structure of **DAMFusion** is illustrated in the figure below:

![image](https://github.com/Wohaizainuli/DAMFusion/blob/main/Figure/3.jpg)

*Fig. 1: Overall Architecture of DAMFusion. For the detailed network structure, please refer to `model/vir_branch.py`.*  



The structures of EDMoE and FMoE are further illustrated in the figure below.  

![image](https://github.com/Wohaizainuli/DAMFusion/blob/main/Figure/4.jpg)  

*Fig. 2: The architectures of EDMoE and FMoE. Both are composed of three components: Gating, Expert Layer and Aggregation Layer. For the detailed network structure, please refer to `model/vir_branch.py`,`FMoEGate.py` and `Mltransformer.py` *  

## 🔹 Dataset
To begin, please first acquire the datasets. This project uses 5 publicly available datasets：  

    
Infrared-visible image fusion datasets:
- **LLVIP**：http://bupt-ai-cz.github.io/LLVIP/
- **M3FD**：https://github.com/dlut-dimt/TarDAL
- **MSRS**： https://github.com/Linfeng-Tang/MSRS
- **FLIR**：https://oem.flir.com/zh-cn/solutions/automotive/adas-dataset-form/、

      
Medical datasets：
- **Havard Medical Image Fusion Datasets**：https://github.com/xianming-gu/Havard-Medical-Image-Fusion-Datasets
