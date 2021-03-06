---
layout:     post                    # 使用的布局（不需要改）
title:      Identity Mappings in Deep Residual Networks           # 标题 
subtitle:   论文阅读笔记 #副标题
date:       2018-11-04              # 时间
author:     Kirito                      # 作者
header-img: img/archives-bg-mayday.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 卷积网络
---



# 论文笔记

**论文题目：**Identity Mappings in Deep Residual Networks

### 1、Introduction

- 一般残差单元过程可以公式化为 $$x_{l+1}=f(h(x_l)+{\cal F}(x_l,{\cal W}_l))$$ 。$$x_l$$ 和$x_{l+1}$ 为第$l$ 层的输入和输出。在ResNet V1中$$h(x_l)=x_l$$ 为恒等映射，$f$ 为**ReLU**函数。
- 采用恒等映射的收敛速度最快，训练的错误率下降最快
- 使用“**纯净**”的信息路径有助于更易于优化
- 与ResNet V1采用的“Post-activation”不同，ResNet V2采用的“Pre-activation”
  - Post-activation：weight $\rightarrow$ BN $\rightarrow $ ReLU $\rightarrow $ weight $\rightarrow $ BN $\rightarrow $ addition $\rightarrow $ ReLU
  - Pre-activation：BN $\rightarrow $ ReLU $\rightarrow $ weight $\rightarrow $ BN $\rightarrow $ ReLU $\rightarrow $ weight $\rightarrow $ addition
- ResNet V2 比ResNet V1更易于训练，且泛化能力强

### 2、Analysis of Deep Residual Network

- $x_l$ 为输入第$l$ 个残差单元的特征
- $${\cal W}_l=\{w_{l,k} \vert_{1 \leq k \leq K}\}$$ 为一组与第$l$ 个残差单元相关的权值（及偏差），$K$ 为残差单元的层数
- $\cal F$ 表示残差函数
- 在ResNet V1中$f$ 函数在叠加后进行，且此函数为ReLU
- $h$ 为恒等映射$$h(x_l)=x_l$$

如果将$f$ 设置为恒等映射，即$x_{l+1}\equiv y_l$ ，得$x_{l+1} = x_l +{\cal F}(x_l, {\cal W}_l)$
$$
\therefore \qquad x_L = x_l + \sum_{i=l}^{L-l}{\cal F}(x_i,{\cal W}_i)\qquad 
$$

对于任意深层单元$L$ 和任意浅层单元$l$ 之间的关系表达式

$$
x_L = x_0 + \sum_{i=0}^{L}{\cal F}(x_i, {\cal W}_i)
$$

在传统网络中，$x_L$ 是一系列矩阵与向量内积的结果$\, x_L= \prod_{i=0}^{L-1}{\cal W}_ix_0$  ，也就是说积累方式从累乘 转换成累加

令损失函数为$\mathcal{E}$ 为损失函数，根据链式法则有

$$
\cfrac {\partial \mathcal{E}}{\partial x_l}=\cfrac{\partial \mathcal{E}}{\partial x_L} \cfrac{\partial x_L}{\partial x_l}=\cfrac{\partial \mathcal{E}}{\partial x_L}\left (1+\cfrac{\partial}{\partial x_l}\sum_{i=l}^{L-1}{\cal F}(x_i,{\cal W}_i) \right)
$$

由此，梯度$\cfrac{\partial \mathcal{E}}{\partial x_l}$ 可以分成两个相加的部分：

- $\cfrac{\partial \cal{E}}{\partial x_L}$ ：不用考虑任何权值层，直接传播的信息，保证信息可直接反向传播到每一个浅层单元$l$
- $\cfrac{\partial \cal{E}}{\partial x_L}\left(\cfrac{\partial}{\partial x_l}\sum_{i=l}^{L-1}{\mathcal F}(x_i,{\cal W}_i)\right)$ ：传递经过每一个卷积层

由于在Mini Batch中不可能每一个样本的$\cfrac{\partial \cal{E}}{\partial x_L}\left(\cfrac{\partial}{\partial x_l}\sum_{i=l}^{L-1}{\mathcal F}(x_i,{\cal W}_i)\right)$ 值都为-1，因此这种形式的结构不容易产生梯度消失的问题。

### 3、On the Importance of Identity Skip Connections

 令$f$仍为恒等变换，假设
$$
x_{l+1}=\lambda_lx_l + \mathcal{F}(x_l,\mathcal{W}_l)
$$
迭代展开L层的特征值表达式
$$
x_L = (\prod_{i=l}^{L-1}\lambda_i)x_l+\sum_{i=l}^{L-1}\left(\prod_{j=i+1}^{L-1}\lambda_j\right)\mathcal{F}(x_i,\mathcal{W}_i)
$$
简化后
$$
x_L = (\prod_{i=l}^{L-1}\lambda_i)x_l+\sum_{i=l}^{L-1}\hat{\mathcal{F}}(x_i,\mathcal{W}_i)
$$
根据链式法则
$$
\cfrac {\partial \mathcal{E}}{\partial x_l}=\cfrac{\partial \mathcal{E}}{\partial x_L} \cfrac{\partial x_L}{\partial x_l}=\cfrac{\partial \mathcal{E}}{\partial x_L}\left((\prod_{i=l}^{L-1}\lambda_i)+\cfrac{\partial}{\partial x_l}\sum_{i=l}^{L-1}\hat{\mathcal{F}}(x_i,\mathcal{W}_i)\right)
$$
根据上式，若$\lambda_i > 1$时，第一项式会指数级增长，若$\lambda_i <1$，该项指数则会指数级减少无限趋近于零，导致梯度信息无法通过shortcut connection进行传播，同时逼迫梯度信息大多通过卷积层进行反向传播，这样与传统反向传播方式一致，仍然会存在梯度消失的问题。

### 4、On the Usage of Activation Function

**五组对比实验**

- original
- BN after addition
- ReLU before addition
- ReLU only pre-activation
- full pre-activation

**分析：**

- 由于$f$ 为恒等映射，易于优化
  - 当训练ResNet-1001时，这种影响尤为明显。使用ResNet V1模型结构，在训练初期训练错误率下降非常缓慢
- 在Pre-activation中使用BN，改善了模型的正规化，降低过拟合

### 5、Results

文章对残差网络的恒等映射做了公式化分析，提出两个恒等映射条件：（1）Identity Skip Connections. （2）Identity after-addition activation. 满足这两个条件的残差网络可以保证梯度反向传播的光滑性以及一定程度上缓解梯度消失问题。





















<html>

<head>
<title>MathJax TeX Test Page</title>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>
<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_CHTML">
</script>
</head>
<body>