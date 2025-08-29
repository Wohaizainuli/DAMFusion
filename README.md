# DAMFusion: A Degradation-Aware Adaptive Multi-Task Fusion Framework for Robust Image Integration  

> DAMFusion is a Degradation-Aware Adaptive Multi-Task Fusion Framework designed to enhance robust image integration across various imaging tasks.
> It leverages a two-stage autoencoder architecture in combination with a Mixture of Experts (MoE) network to adaptively handle multiple tasks.

  

> This repository provides a non-official and simplified version of our implementation, intended solely for academic communication and preliminary understanding.

     
> To protect the originality of our work and ensure compliance with data and intellectual property considerations, the **complete codebase will be released after the paper is officially published**.

     
> We appreciate your understanding and support!

## 🔹 Framework Overview

The structure of **DAMFusion** is illustrated in the figure below:

![image](https://github.com/Wohaizainuli/DAMFusion/blob/main/Figure/3.jpg)

*Fig. 1: Overall Architecture of DAMFusion. For the detailed network structure, please refer to `model/vir_branch.py`.*  



The structures of EDMoE and FMoE are further illustrated in the figure below.  

![image](https://github.com/Wohaizainuli/DAMFusion/blob/main/Figure/4.jpg)  

*Fig. 2: The architectures of EDMoE and FMoE. Both are composed of three components: Gating, Expert Layer and Aggregation Layer. For the detailed network structure, please refer to `model/vir_branch.py`,`FMoEGate.py` and `Mltransformer.py`.*  

## 🔹 Dataset
To begin, please first acquire the datasets. This project uses 5 publicly available datasets：  

    
Infrared-visible image fusion datasets:
- **LLVIP**：http://bupt-ai-cz.github.io/LLVIP/
- **M3FD**：https://github.com/dlut-dimt/TarDAL
- **MSRS**： https://github.com/Linfeng-Tang/MSRS
- **FLIR**：https://oem.flir.com/zh-cn/solutions/automotive/adas-dataset-form/、


    
Medical datasets：
- **Havard Medical Image Fusion Datasets**：https://github.com/xianming-gu/Havard-Medical-Image-Fusion-Datasets


Please refer to the official sources of each dataset for download and usage instructions.

## 🔹 Model Training Pipeline
Due to multiple considerations, including data confidentiality and resource constraints, we are currently unable to provide the trained model weights. We understand that this may cause some inconvenience, and we sincerely appreciate your interest and support.  



The trained models are saved under the `run/` directory. The overall training process consists of three main stages:  

 **Step 1: LoRA Fine-tuning on a Pretrained CLIP Classifier**  
We first perform LoRA-based fine-tuning on a pretrained CLIP classifier to enable the model to generate degradation-aware image embeddings directly from the input images. The resulting model weights are saved as `w1`.      
For more details on this step, please refer to the author's related repository: SGAFusion(https://github.com/Wohaizainuli/SGAFusion)  



      
 **Step 2: Training Stage I – Image Restoration Network**  
Next, we load `w1` and use `train_main0.py` to train the Stage I image restoration network. After training, we save the encoder weights of the final model as `w2`.      



     
 **Step 3: Training Stage II – Joint Fusion Network**  
Finally, we load both `w1` and `w2`, and run `train_main1.py` to train the Stage II joint fusion network. The final model weights from this stage are saved as `w3`.   

  

## 🔹 Model Inference  
To perform inference, run `test_parms_fps.py` with the pretrained weights `w1` and `w3`.
After placing the images to be fused in the specified input directory, the fusion results will be automatically generated and saved to the corresponding output directory.  


## 🔹 Result
We selected a subset of the dataset and their corresponding fusion results, which are provided in the `/Image` directory. Figures 3 and 4 below illustrate the fusion results of the proposed algorithm in comparison with those of the baseline methods.
![image](https://github.com/Wohaizainuli/DAMFusion/blob/main/Figure/result1.jpg)
*Fig .3: Visual Comparison on the Infrared-Visible Dataset.*

![image](https://github.com/Wohaizainuli/DAMFusion/blob/main/Figure/result2.jpg)
*Fig .4: Visual Comparison on the Medical Dataset.*

## 🔹 Extension Experiment  

Furthermore, to assess the effectiveness of the proposed fusion method, we conduct object detection experiments using the YOLOv8 model (https://github.com/ultralytics/ultralytics) as a downstream task.  
  
    
    The visual comparison of the detection results is presented in the figure below.
    
![image](https://github.com/Wohaizainuli/DAMFusion/blob/main/Figure/Detection.jpg)
*Fig .5: Visual Comparison of Object Detection. The LLVIP dataset only provides labels for the "Person" class.*



## 🔹 Contributing & Contact

Thank you for reading and supporting DAMFusion! If you have any questions or encounter issues, feel free to open an issue in this repository or contact me directly at **junjiema_xmtra@163.com**.  
We welcome contributions—please fork the repo, make your changes, and submit a pull request.
