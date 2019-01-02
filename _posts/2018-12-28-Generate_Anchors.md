---
layout:     post                    # 使用的布局（不需要改）
title:      Generate Anchors         # 标题
date:       2018-12-28             # 时间
author:     Kiri                      # 作者
header-img: img/kiri-bg-mayday.jpg   #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 笔记
---

# Generate Anchors

在Feature Map上选用用一个$3\times 3$的滑窗，每个滑窗生成9个anchors（具有相同中心点$(x_a,y_a）$，但具有3种不同的长宽比（aspect ratios）和3种不同的尺寸（scales），**计算的是相对于原始图片的尺寸**）

![pic1](https://github.com/caiwendi/caiwendi.github.io/raw/master/img/Anchors.png)

对每个anchor计算与ground-truth bbox的重叠比IOU

```
if IoU > 0.7 p = 1
	else if IoU < 0.3 p = -1
		else p = 0
```

从Feature Map中提取$3 \times 3$的空间特征，并将其输入到一个小网络中，该网络具有两个输出任务分支：

- 分类（classfication）：输出一个概率值，表示bounding-box中是否包含物体或者背景

  - 损失定义：

  判断是背景还是前景，进行损失计算

- 回归（regression）：预测边界框（bounding-box）

  - 损失定义：

  GroundTruth: $$(x^*,y^*,w^*,h^*)$$

  Anchors: $(x_a,y_a,w_a,h_a)​$

  prediction: $(x,y,w,h)$

  $$
  t=((x-x_a)/w_a, (y-y_a)/h_a, \log w/w_a, \log h/h_a)
  $$

  $$
  t^*=((x^*-x_a)/w_a,(y^*-y_a)/h_a,\log w^*/w_a,\log h^*/h_a)
  $$



  计算$t$和$t^*$的损失

  ![pic2](https://github.com/caiwendi/caiwendi.github.io/raw/master/img/RPN.png)


















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

