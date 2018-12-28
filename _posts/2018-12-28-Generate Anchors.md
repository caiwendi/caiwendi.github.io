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

在feature map上选用用一个$3\times 3$的滑窗，每个滑窗生成9个anchors（具有相同中心点$(x_a,y_a）$，但具有3种不同的长宽比（aspect ratios）和3种不同的尺寸（scales），**计算的是相对于原始图片的尺寸**）















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

