# DFN：Distributed Feedback Network for Single-Image  Deraining

## Introduction 
Recently, deep convolutional neural network has achieved great success for single-image deraining. However, affected by the intrinsic overlapping between rain streaks and background texture patterns, most of these methods tend to more or less remove texture details in rain-free regions and lead oversmoothing effect in the recovered background. In order to generate reasonable rain streak layer and improve reconstruc- tion quality of background, we propose a distributed feedback network (DFN) in recurrent structure. A novel feedback block is designed to implement feedback mechanism. In each feedback block, hidden state with high-level information (output) will flow into the next iteration to correct the low-level representations (in- put). By stacking multiple feedback blocks, the proposed network where the hidden states are distributed can extract powerful high- level representations for rain streak layer. Curriculum learning is employed to connect the loss of each iteration and ensure hidden states contain the notion of output. In addition, a self-ensemble strategy for rain removal task which can keep the vertical character of rain streak is explored to maximize the potential performance of deraining model. Extensive experimental results demonstrate the superiority of DFN in comparison with the state- of-the-art methods.

![image](https://github.com/Hang-Z/DFN/blob/master/Images/structure.png)

## Requirements

*Python 3.7,Pytorch >= 0.4.0  
*Requirements: opencv-python  
*Platforms: Ubuntu 18.04,cuda-10.2  
*MATLAB for calculating PSNR and SSIM 

## Datasets
DFN is trained and tested on five datasets: Rain100L[1],Rain100H[1],RainLight[2],RainHeavy[2],Rain12[3]. It is worth mentioning that several the state-of-the-arts were trained on a strict Rain100H, which contains 1254 pairs of images. We re-trained these competing methods on whole Rain100H.

*Note: 

(i) The authors of [1] updated the Rain100L and Rain100H, we call the new datasets as RainLight and RainHeavy here.

(ii) The Rain12 contains only 12 pairs of testing images, we use the model trained on Rain100L to test on Rain12.

## Getting Started
### Test
All the pre-trained models were placed in `./models/`.

Run the `test.py` to obtain the deraining images. Then, you can calculate the evaluation metrics by run the MATLAB scripts in `./statistics/`. For example, if you want to compute the average PSNR and SSIM on Rain100L, you can run the `Rain100L.m`.

### Train
If you want to train the models, you can run the `train.py` and don't forget to change the `args` in this file. Or, you can run in the terminal by the following code.

`python train.py --save_path path_to_save_trained_models  --data_path path_to_training_dataset`

### Results

Average PSNR and SSIM values of DFN on five datasets are shown:


Datasets | GMM|DDN| ResGuideNet|JORDER-E|SSIR|PReNet|BRN|MSPFN|DFN|DFN+
----|----|----|----|----|----|----|----|----|----|----
Rain100L|28.66/0.865|32.16/0.936|33.16/0.963|-|32.37/0.926|37.48/0.979|38.16/0.982|37.5839/0.9784|39.22/0.985|39.85/0.987
Rain100H|15.05/0.425|21.92/0.764|25.25/0.841|-|22.47/0.716|29.62/0.901|30.73/0.916|30.8239/0.9055|31.40/0.926|31.81/0.930
RainLight|-|31.66/0.922|-|39.13/0.985|32.20/0.929|37.93/0.983|38.86/0.985|39.7540/0.9862|39.53/0.987|40.12/0.988
RainHeavy|-|22.03/0.713|-|29.21/0.891|22.17/0.719|29.36/0.903|30.27/0.917|30.7112/0.9129|31.07/0.927|31.47/0.931
Rain12|32.02/0.855|31.78/0.900|29.45/0.938|-|34.02/0.935|36.66/0.961|36.74/0.959|35.7780/0.9514|37.19/0.961|37.55/0.963



![image](https://github.com/Hang-Z/DFN/blob/master/Images/results.png)

## References
[1]Yang W, Tan R, Feng J, Liu J, Guo Z, and Yan S. Deep joint rain detection and removal from a single image. In IEEE CVPR 2017.

[2]Yang W, Tan R, Feng J, Liu J, Yan S, and Guo Z. Joint rain detection and removal from a single image with contextualized deep networks. IEEE T-PAMI 2019.

[3]Li Y, Tan RT, Guo X, Lu J, and Brown M. Rain streak removal using layer priors. In IEEE CVPR 2016.
