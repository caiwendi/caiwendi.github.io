---
layout:     post                    # 使用的布局（不需要改）
title:      基于候选区域的目标检测算法——Anchors         # 标题
date:       2018-12-28             # 时间
author:     Kiri                      # 作者
header-img: img/post-bg-img.jpg   #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 笔记
---

# Anchors

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

  Anchors: $(x_a,y_a,w_a,h_a)$

  prediction: $(x,y,w,h)$

  $$
  t=((x-x_a)/w_a, (y-y_a)/h_a, \log w/w_a, \log h/h_a)
  $$

  $$
  t^*=((x^*-x_a)/w_a,(y^*-y_a)/h_a,\log w^*/w_a,\log h^*/h_a)
  $$





  计算$t$和$t^*$的损失

  ![pic2](https://github.com/caiwendi/caiwendi.github.io/raw/master/img/RPN.png)

# Generate Anchors

具体实现代码如下：

```python
def generate_anchors(stride=16, size=(32, 64, 128), aspect_ratios=(0.5, 1, 2)):
    """
    生成anchors原理：
    （1）先对anchor生成不同aspect_ratios下的base_anchors
    （2）再根据实际设计的anchor的尺寸，将base_anchors进行扩大，扩大倍数就是size / stride
    :param stride: anchor的步长
    :param size: 候选框的尺寸
    :param aspect_ratios: 候选框的长宽比
    :return: anchors ((x1, y1, x2, y2),.....)
    """
    return _generate_anchors(
        stride,
        np.array(size, dtype=np.float) / stride,
        np.array(aspect_ratios, dtype=np.float),
    )

def _generate_anchors(base_size, scales, aspect_ratios):
    """
    生成anchors
    :param base_size: 基本anchor的尺寸
    :param scales: anchor的尺寸
    :param aspect_ratios: 长宽比
    :return: anchors detype=torch.float64
    """
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack([_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])])
    return torch.from_numpy(anchors)

def _whctrs(anchor):
    """
    返回anchor的宽度、高度以及中心点的坐标
    :param anchor: (x1, y1, x2, y2)
    :return: w, h, x_ctr, y_ctr
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
	"""
    给定中心点和长宽，返回对应的anchors
    :param ws: 设定的宽度
    :param hs: 设定的高度
    :param x_ctr: 中心点x坐标
    :param y_ctr: 中心点y坐标
    :return: anchors
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        )
    )
    return anchors

def _ratio_enum(anchor, ratios):
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

```


















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

