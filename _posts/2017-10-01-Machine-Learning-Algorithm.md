---
layout:     post                    # 使用的布局（不需要改）
title:      Machine Learning Algorithm           # 标题 
subtitle:   Gradient Descent, Logistic Regression, Naive Bayes,  SVM   #副标题
date:       2017-10-01             # 时间
author:     Brian                      # 作者
header-img: img/railway.jpg    #这篇文章标题背景图片
catalog:    true                       # 是否归档
tags:                                   #标签
          - Machine Learning Algorithm

---



## 1. Gradient Descent Algorithm



where $x_k$ ，denotes the vector of scores for all the pixels for the class $k\in\{1,\cdots,L\}$. The per-class unaries are denoted by $b_k$ , and the pairwise terms $\hat{A}$  are shared between each pair of classes. The equations that follow are derived by specializing the general inference (Eq. 2) and gradient equations (Eq.3, 4) to this particular setting. Following simple manipulations, the inference procedure becomes a two step process where we first compute the sum of our scores $\sum\nolimits_{i}^{}x_i$ , followed by $x_k$ . the scores for the class $k$  as ： 

$$(\lambda I+(L-1)\hat{A})\sum_{i}^{}x_i=\sum_{i}^{}b_i,​$$
$$(\lambda I-\hat{A})x_k=b_k-\hat{A}\sum_{i}^{}x_i.$$

Derivatives of the unary terms with respect to the loss are obtained by solving

$$(\lambda I+(L-1)\hat{A})\sum_{i}^{}\frac{\partial \zeta}{\partial b_i}=\sum_{i}^{}\frac{\partial \zeta}{\partial x_i},$$
$$(\lambda I-\hat{A})\frac{\partial \zeta}{\partial b_k}=\frac{\partial \zeta}{\partial x_k}-\hat{A}\sum_{i}^{}\frac{\partial \zeta}{\partial b_i}.$$

Finally, the gradients of $\hat{A}$  are computed as

$$\frac{\partial \zeta}{\partial \hat{A}}=\sum_{k}^{}\frac{\partial \zeta}{\partial b_k}\bigotimes \sum_{i\neq k}^{}x_i.$$




## 2. Logistic Regression Algorithm





## 3. Naive Bayes Algorithm





## 4. SVM Algorithm











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















