---
layout:     post                    # 使用的布局（不需要改）
title:     Deep Convolutional Nets for Semantic Image Segmentation with Deep Gaussian CRFs               # 标题 
subtitle:   Image segmentation, Convolutional Nets, G-CRF, Multi-Scale Prediction  #副标题
date:       2017-09-25             # 时间
author:     Brian                      # 作者
header-img: img/railway.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                                   #标签
          - 研究方向
---


## 1、会议论文汇报PPT


![](https://i.imgur.com/6uH8w0w.png)
![](https://i.imgur.com/C8me9BC.png)
![](https://i.imgur.com/QwqHDgS.png)
![](https://i.imgur.com/IF0SwJB.png)
![](https://i.imgur.com/udxmhiP.png)

![](https://i.imgur.com/FaxK2y8.png)
![](https://i.imgur.com/XyQ8xif.png)
![](https://i.imgur.com/avRSuT0.png)
![](https://i.imgur.com/9jS4zHB.png)
![](https://i.imgur.com/6FcoDDB.png)

![](https://i.imgur.com/ain5alI.png)
![](https://i.imgur.com/A2GwDd5.png)
![](https://i.imgur.com/jr60kCH.png)
![](https://i.imgur.com/Ss1ZAfH.png)
![](https://i.imgur.com/loU2aSh.png)

![](https://i.imgur.com/ZiKLFsW.png)
![](https://i.imgur.com/zv0EudO.png)
![](https://i.imgur.com/fVa3S3F.png)
![](https://i.imgur.com/7K53LxF.png)
![](https://i.imgur.com/2wuLmdN.png)

___

## 2、中文演讲稿

各位教授，大家好

1. 我汇报的题目是**Deep Convolutional Nets for Semantic Image Segmentation with Deep Gaussian CRFs**,
2. 我将从以下四个方面介绍我的研究成果，第一方面介绍我们需要解决难题和我们用到的基本方法，第二方面和第三方面分别介绍解决该问题的前期卷积神经网络处理过程和后期高斯条件随机场处理过程，最后对我们研究的结果做了基本总结。    
3. 图像语义分割是针对图片的每一个像素进行分类，该技术是场景分类，目标追踪等领域的基石，目前语义分割的方法是通过卷积神经网络进行像素分类，该方法的缺点是由于卷积神经网络的空间不变性导致的定位精度不高，难以得到理想的结果。为了克服这一缺点，达到卷积神经网络的分类准确性和定位精度的平衡，我们采用以下三种方法：多尺度分割，降低卷积神经网络的感受野，条件随机场做后期处理。
4. 整个语义分割过程分为两个部分：第一个部分是采用卷积神经网络作为前期处理来处理分类问题，我们通过降低感受野方法来缩短计算时间和多尺度预测获得更多低维特征提高边界定位精度。
5. 第二个部分是采用高斯条件随机场作为后期处理来解决定位问题，我们通过恢复丢失的边缘信息得到准确的语义分割结果。
6. 接下来这个图展示我们整个研究过程的思路，红色虚线左边是作为语义分割前期处理的卷积神经网络，红色虚线右边是作为语义分割后期处理的高斯条件随机场。用来训练整个网络的数据库是The VOC PASCAL 2012
7. 接下来介绍卷积神经网络作为语义分割前期处理的具体内容：
8. 降低感受野的目的为了减少运行时间，如红色的方框中，需要对每一个第一个全连接层进行子采样。
9. 为了获得更多低维特征采用多尺度来方法提高边界定位精度, 前面四个最大池化层中的每一个的输出和输入图像被附加到两层多层感知器（MLP）。
10. 接下来介绍高斯条件随机场作为语义分割后期处理的具体内容
11. 整个语义分割的后期处理过程采用高斯条件随机场，我们将条件随机场的能量函数定义为下面形式，要保证全局最小的条件是 $(A+\lambda I)x = B$, 为了达到此条件让整个模型收敛，可以通过链式求导规则求解出A和B的梯度。
12. 下面我们采用了另一种方法可以让整个模型收敛的更快, 通过Potts-type pairwise模型将二元能量形式转化为每个像素匹配一个标签，再将收敛条件的等式转化为矩阵形式，
13. 然后通过链式求导法和一系列推导得出$\hat{A}$的梯度。
14. 以上得到权重的梯度后，最后用Softmax进行分类，对每个像素和标签的匹配给出概率值，这些概率值由交叉熵进行惩罚处理，最后随着训练次数的增加，交叉熵的损失值越小，模型的预测会越精准。
15. 下面对实验的部分进行介绍，数据集采用PASCAL VOC 2012,分为训练集，验证集和测试集，一共有20个物体类和1个背景，分割的标准是交除并，在训练过程中，我们采用在ImageNet上预训练好的DCNN，交叉熵损失值的下降方式采用共轭梯度下降，经过不断试验最后选定的Mini-batch = 20, Momentum = 0.9, Weight decay = $5*10^{-4}$, Learning rate = 0.001并且每迭代2000次学习率降低10倍。
16. 下面是我们提出方法的具体实验数据，表一是增强训练数据集训练的模型在验证集下的准确率，我们看到Basenet的准确率是62.25%，每一个红色矩形部分中的数据比上一组数据高大约1%左右，最好的结果采用多尺度，二次优化和条件随机场达到76.40%。表二和表一的区别在于表二扩大了数据集，达到最高的准确率78.3%。由此得出，我们采用方法的有效性。
17. 下面是我们将原始图片分割后的效果，第一列是原始图片，第二列是手工标注的图片，第三列是二次优化分割的效果，第四列是多尺度和二次优化分割的效果，最后一列是将多尺度，二次优化和条件随机场做结合分割效果图，可以观察到结合这三种方法捕捉到了更多细节和获得了更好的分割效果。
18. 最后对我们的研究成果进行总结
19. 我们的结论主要分为三个方面: 通过控制感受野的大小，能在不损失性能的情况下减少运算的时间；多尺度分割方法能够提高边缘定位精度；最后高斯条件随机场能够进一步捕捉细腻的边缘信息获得更好的分割效果。
20. 我的汇报结束，谢谢大家的聆听。
___

## 3、英文演讲稿

Dear professors, good morning/afternoon:

1. Thank you for giving me an opportunity to speak about what I have done at this special occasion, what I am going to talk about today is my topic ,Deep Convolutional Nets for Semantic Image Segmentation with Deep Gaussian CRFs.
2. I will introduce our research results by the following four parts. The  introduction is about what kind of problem we need to solve  and how we deal with it , the second part is about convolutional neural network as pre-processing of solution to the problem, the third part is about Gaussian condition random fields as post-processing, Finally, we summarize the results of our study.
3. Image semantic segmentation is to classify each pixel of the picture, which is the cornerstone of scene classification and target tracking. The current main semantic segmentation method is to classify pixels by convolutional neural network. The disadvantage of this method is the spatial invariance of the convolutional neural network,which can cause limited positioning accuracy, what is more, it is difficult to get the  desired results. In order to overcome this shortcoming, to achieve balance between the classification and the positioning accuracy of of the convolutional neural network, we use the following three methods: multi-scale segmentation, reduce the receptive of  convolutional neural network and conditional random field as post-processing.
4. The whole semantic segmentation process is divided into two parts: the first part is to use convolutional  neural network as a pre-processing to deal with the classification problem, so we can decrease the calculation time by reducing the receptive field and multi-scale prediction can  obtain more low-dimensional features to improve the border position accuracy.
5. The second part is to use the Gaussian conditional random field as a post-processing to solve the localization problem, so  we get the exact semantic segmentation result by recovering the missing edge information.
6. Next, this figure shows the main idea of our whole research process. The left side of the red dotted line is a convolutional neural network as pre-processing of semantic segmentation.  The right side of the red dotted line is the Gaussian conditional random field as post-processing of semantic segmentation. The database, The VOC PASCAL 2012, is used to train the entire network.
7. Next, I will explain the details of semantic segmentation pre-processing:
8. In order to decrease the calculation time, we need to reduce the receptive field.  Look at the red rectangle, spatially subsampling (by simply extracting) the first FC layer to a 4×4 (or 3×3) spatial size.
9. In order to obtain more low-dimensional features, the multi-scale method is used to improve the boundary position accuracy. The output and input image of each of the top four largest pooling layers are added to the two-layer multi-layer perceptron (MLP).
10. Then, we introduce Gaussian conditional random field as the  post-processing  of semantic segmentation.
11. In the post-processing of the whole semantic segmentation, we use the Gaussian conditional random field. We define the energy function of the conditional random field as the following form. To ensure that the global minimum condition is $ (A + \lambda I) x = B $ and the whole model convergence,  the gradient of A and B can be solved in terms of chaining rule.
12. Here we use another method to make the whole model converge faster. A Potts-type pairwise model is
    proposed to describe by the  following equation , the pairwise energy term is denoted for $ p_i$ pixel taking the $ l_i$
    label, and  $ p_j$ pixel taking the $ l_j$ label.To derive the inference and gradient equations, the inference equation$ (A + λI)x = B$ is rewritten as Matrix form.
13. And then through the chain derivation rule and a series of inferences, we can get  $ \hat {A} $  gradient.
14. After the gradient of every weight is obtained, the probability is given by Softmax, and the probability value is given for each pixel and label. These probability values are penalized by cross-entropy. Finally, with the increase of training times, the smaller the loss of cross entropy is, the more accurate the model will be.
15. The experimental part will be explained, the dataset is PASCAL VOC 2012, divided into training set, validation set and test set, including 20 objects classes and a background, image segmentation performance is measured in the terms of intersection-over-union(IOU) averaged across the 21 classes. During training processes, We used the pre-trained DCNN on IMAGENET, the cross-entropy loss function is conjugate gradient descent, and the final parameters as follow:  Mini-batch = 20, Momentum = 0.9, Weight decay = $ 5 * 10 ^ { -4} $, Learning rate = 0.001 and multiply 10 times per iteration of 2000 learning rates.
16. The following is the specific experimental data of our proposed method. Table 1 shows  the accuracy values under the validation set after model was trained in augmented training set, we see Basenet accuracy is 62.25%, each the data in red rectangle is approximately 1% higher than the previous one, the best results is up to  76.40% using multi-scale, secondary optimization and conditional random field. The difference between Table 2 and Table 1 is that Table 2 have the larger data set to achieve the highest accuracy rate of 78.3%.So it proves our method effective.
17. we can see visual results on the VOC PASCAL 2012 test set. The first column is the raw image, the second column shows the ground-truth of image segmentation, the third column shows the predicted segmentation of quadratic optimization model , the fourth column is  the predicted segmentation with combination of multi-scale and quadratic optimization, the last column shows outputs after the multi-scale, quadratic optimization and conditional random field are applied to our model, you can observe the combination of these three methods to capture more details and get a better segmentation results.
18. Finally, we summarize our research results.
19. Our conclusions are divided into three aspects. First, controlling the receptive field size can decrease computation time and memory capacity without sacrificing performance; Second,  a multi-scale prediction method can increase the boundary localization accuracy; Finally, Gaussian conditional random fields can obtain object boundaries at a level of detail and produce accurate predictions and detailed segmentation maps;
20. That's all, thank you for listening.


















