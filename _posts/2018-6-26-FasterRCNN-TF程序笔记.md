---
layout:     post                    # 使用的布局（不需要改）
title:      Faster RCNN TF        # 标题
date:       2018-6-26             # 时间
author:     Kiri                      # 作者
header-img: img/post-bg-2015.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 程序
---

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

# TF-Faster-RCNN
### Prerequisites
* CUDA 8.0
* CuDnn 5.0
* TensorFlow r1.2: sudo pip install tensorflow-gpu==1.2
* Cython: sudo pip install cython
* opencv-python: sudo pip install opencv-python
* easydist: sudo pip install easydist

### Installation
##### 1. Clone the repository
```
git clone https://github.com/endernewton/tf-faster-rcnn.git
```
##### 2. Update your-arch in setup script to match your GPU
```
cd tf-faster-rcnn/lib
# Change the GPU architecture (-arch) if necessary
vim setup.py
```
GPU model | Architecture
----|-------------------
TitanX | sm_52
GTX 960M | sm_50
GTX 1080 Ti | sm_61

##### 3. Build the Cython modules
```
make clean
make
cd ..
```
##### 4. Install the Python COCO API
```
cd data
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make
cd ../../..
```
###Setup data
#### PASCAL VOC 2007
##### 1. Download the training, validation, test data and VOCdevkit
```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```
##### 2. Extract all of these tars into one directory named ```VOCdevkit```
```
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```
##### 3. Create symlinks for the PASCAL VOC dataset
```
cd $FRCN_ROOT/data
ln -s $VOCdevkit VOCdevkit2007
```
#### MS COCO
Install the MS COCO dataset at /path/to/coco
```
cd $FRCN_ROOT/data
ln -s $COCO coco
```
### Demo and Test with pre-trained models
##### 1. Download pre-trained models
Downlod Link: [Google Driver](https://drive.google.com/drive/folders/0B1_fAEgxdnvJSmF3YUlZcHFqWTQ)
##### 2. Create a folder and a soft link to use the pre-trained model
```
NET=res101
TRAIN_IMDB=voc_2007_trainval+voc_2012_trainval
mkdir -p output/${NET}/${TRAIN_IMDB}
cd output/${NET}/${TRAIN_IMDB}
ln -s ../../../data/voc_2007_trainval+voc_2012_trainval ./default
cd ../../..
```
##### 3. Demo for testing on custom images
```
# at repository root
GPU_ID=0
CUDA_VISIBLE_DEVICES=${GPU_ID} ./tools/demo.py
```
**Note**: Resnet101 testing probably requires several gigabytes of memory, so if you encounter memory capacity issues, please install it with CPU support only. Refer to [Issue 25](https://github.com/endernewton/tf-faster-rcnn/issues/25)

##### 4. Test with pre-trained Resnet101 models
```
GPU_ID=0
./experiments/scripts/test_faster_rcnn.sh $GPU_ID pascal_voc_0712 res101
```
**Note**: If you cannot get the reported numbers (79.8 on my side), then probably the NMS function is compiled improperly, refer to [Issue 5](https://github.com/endernewton/tf-faster-rcnn/issues/5)
###Train model
###### 1. Download pre-trained models and weights.
Download [pre-trained models](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)
```
mkdir -p data/imagenet_weights
cd data/imagenet_weights
wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xzvf vgg_16_2016_08_28.tar.gz
mv vgg_16.ckpt vgg16.ckpt
cd ../..
```
##### 2. Train
```
./experiments/scripts/train_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
# GPU_ID is the GPU you want to test on
# NET in {vgg16, res50, res101, res152} is the network arch to use
# DATASET {pascal_voc, pascal_voc_0712, coco} is defined in train_faster_rcnn.sh
# Examples:
./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16
./experiments/scripts/train_faster_rcnn.sh 1 coco res101
```
##### 3. Visualization with Tensorboard
```
tensorboard --logdir=tensorboard/vgg16/voc_2007_trainval/ --port=7001 &
tensorboard --logdir=tensorboard/vgg16/coco_2014_train+coco_2014_valminusminival/ --port=7002 &
```
Chorme: IP:7001/7002
##### 4. Test and evaluate
```
./experiments/scripts/test_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
# GPU_ID is the GPU you want to test on
# NET in {vgg16, res50, res101, res152} is the network arch to use
# DATASET {pascal_voc, pascal_voc_0712, coco} is defined in test_faster_rcnn.sh
# Examples:
./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
./experiments/scripts/test_faster_rcnn.sh 1 coco res101
```

### VGG-16

* Conv 1_1: $3 \times 3 \times 64$
* Conv 1_2: $3 \times 3 \times 64$
* MaxPool 1: $2 \times 2$
* Conv 2_1: $3 \times 3 \times 128$
* Conv 2_2: $3 \times 3 \times 128$
* MaxPool 2: $2 \times 2$
* Conv 3_1: $3 \times 3 \times 256$
* Conv 3_2: $3 \times 3 \times 256$
* Conv 3_3: $3 \times 3 \times 256$
* MaxPool 3: $2 \times 2$
* Conv 4_1: $3 \times 3 \times 512$
* Conv 4_2: $3 \times 3 \times 512$
* Conv 4_3: $3 \times 3 \times 512$
* MaxPool 4: $2 \times 2$
* Conv 5_1: $3 \times 3 \times 512$
* Conv 5_2: $3 \times 3 \times 512$
* Conv 5_3: $3 \times 3 \times 512$
* FC 6: $4096$
* FC 6: $4096$


### Training Log (VGG-16)
```
+ echo Logging output to experiments/logs/vgg16_voc_2007_trainval__vgg16.txt.2018-06-25_15-39-57
Logging output to experiments/logs/vgg16_voc_2007_trainval__vgg16.txt.2018-06-25_15-39-57
+ set +x
+ '[' '!' -f output/vgg16/voc_2007_trainval/default/vgg16_faster_rcnn_iter_70000.ckpt.index ']'
+ [[ ! -z '' ]]
+ CUDA_VISIBLE_DEVICES=0
+ time python ./tools/trainval_net.py --weight data/imagenet_weights/vgg16.ckpt --imdb voc_2007_trainval --imdbval voc_2007_test --iters 70000 --cfg experiments/cfgs/vgg16.yml --net vgg16 --set ANCHOR_SCALES '[8,16,32]' ANCHOR_RATIOS '[0.5,1,2]' TRAIN.STEPSIZE '[50000]'
/usr/local/lib/python2.7/dist-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Called with args:
Namespace(cfg_file='experiments/cfgs/vgg16.yml', imdb_name='voc_2007_trainval', imdbval_name='voc_2007_test', max_iters=70000, net='vgg16', set_cfgs=['ANCHOR_SCALES', '[8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'TRAIN.STEPSIZE', '[50000]'], tag=None, weight='data/imagenet_weights/vgg16.ckpt')
Using config:
{'ANCHOR_RATIOS': [0.5, 1, 2],
 'ANCHOR_SCALES': [8, 16, 32],
 'DATA_DIR': '/home/lab/tf-faster-rcnn/data',
 'EXP_DIR': 'vgg16',
 'MATLAB': 'matlab',
 'MOBILENET': {'DEPTH_MULTIPLIER': 1.0,
               'FIXED_LAYERS': 5,
               'REGU_DEPTH': False,
               'WEIGHT_DECAY': 4e-05},
 'PIXEL_MEANS': array([[[102.9801, 115.9465, 122.7717]]]),
 'POOLING_MODE': 'crop',
 'POOLING_SIZE': 7,
 'RESNET': {'FIXED_BLOCKS': 1, 'MAX_POOL': False},
 'RNG_SEED': 3,
 'ROOT_DIR': '/home/lab/tf-faster-rcnn',
 'RPN_CHANNELS': 512,
 'TEST': {'BBOX_REG': True,
          'HAS_RPN': True,
          'MAX_SIZE': 1000,
          'MODE': 'nms',
          'NMS': 0.3,
          'PROPOSAL_METHOD': 'gt',
          'RPN_NMS_THRESH': 0.7,
          'RPN_POST_NMS_TOP_N': 300,
          'RPN_PRE_NMS_TOP_N': 6000,
          'RPN_TOP_N': 5000,
          'SCALES': [600],
          'SVM': False},
 'TRAIN': {'ASPECT_GROUPING': False,
           'BATCH_SIZE': 256,
           'BBOX_INSIDE_WEIGHTS': [1.0, 1.0, 1.0, 1.0],
           'BBOX_NORMALIZE_MEANS': [0.0, 0.0, 0.0, 0.0],
           'BBOX_NORMALIZE_STDS': [0.1, 0.1, 0.2, 0.2],
           'BBOX_NORMALIZE_TARGETS': True,
           'BBOX_NORMALIZE_TARGETS_PRECOMPUTED': True,
           'BBOX_REG': True,
           'BBOX_THRESH': 0.5,
           'BG_THRESH_HI': 0.5,
           'BG_THRESH_LO': 0.0,
           'BIAS_DECAY': False,
           'DISPLAY': 20,
           'DOUBLE_BIAS': True,
           'FG_FRACTION': 0.25,
           'FG_THRESH': 0.5,
           'GAMMA': 0.1,
           'HAS_RPN': True,
           'IMS_PER_BATCH': 1,
           'LEARNING_RATE': 0.001,
           'MAX_SIZE': 1000,
           'MOMENTUM': 0.9,
           'PROPOSAL_METHOD': 'gt',
           'RPN_BATCHSIZE': 256,
           'RPN_BBOX_INSIDE_WEIGHTS': [1.0, 1.0, 1.0, 1.0],
           'RPN_CLOBBER_POSITIVES': False,
           'RPN_FG_FRACTION': 0.5,
           'RPN_NEGATIVE_OVERLAP': 0.3,
           'RPN_NMS_THRESH': 0.7,
           'RPN_POSITIVE_OVERLAP': 0.7,
           'RPN_POSITIVE_WEIGHT': -1.0,
           'RPN_POST_NMS_TOP_N': 2000,
           'RPN_PRE_NMS_TOP_N': 12000,
           'SCALES': [600],
           'SNAPSHOT_ITERS': 5000,
           'SNAPSHOT_KEPT': 3,
           'SNAPSHOT_PREFIX': 'vgg16_faster_rcnn',
           'STEPSIZE': [50000],
           'SUMMARY_INTERVAL': 180,
           'TRUNCATED': False,
           'USE_ALL_GT': True,
           'USE_FLIPPED': True,
           'USE_GT': False,
           'WEIGHT_DECAY': 0.0001},
 'USE_E2E_TF': True,
 'USE_GPU_NMS': True}
Loaded dataset `voc_2007_trainval` for training
Set proposal method: gt
Appending horizontally-flipped training examples...
wrote gt roidb to /home/lab/tf-faster-rcnn/data/cache/voc_2007_trainval_gt_roidb.pkl
done
Preparing training data...
done
10022 roidb entries
Output will be saved to `/home/lab/tf-faster-rcnn/output/vgg16/voc_2007_trainval/default`
TensorFlow summaries will be saved to `/home/lab/tf-faster-rcnn/tensorboard/vgg16/voc_2007_trainval/default`
Loaded dataset `voc_2007_test` for training
Set proposal method: gt
Preparing training data...
wrote gt roidb to /home/lab/tf-faster-rcnn/data/cache/voc_2007_test_gt_roidb.pkl
done
4952 validation roidb entries
Filtered 0 roidb entries: 10022 -> 10022
Filtered 0 roidb entries: 4952 -> 4952
2018-06-25 15:40:06.125501: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2018-06-25 15:40:06.125520: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2018-06-25 15:40:06.125523: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2018-06-25 15:40:06.125526: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2018-06-25 15:40:06.125529: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX512F instructions, but these are available on your machine and could speed up CPU computations.
2018-06-25 15:40:06.125532: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2018-06-25 15:40:06.269930: I tensorflow/core/common_runtime/gpu/gpu_device.cc:940] Found device 0 with properties:
name: GeForce GTX 1080 Ti
major: 6 minor: 1 memoryClockRate (GHz) 1.6325
pciBusID 0000:65:00.0
Total memory: 10.91GiB
Free memory: 10.36GiB
2018-06-25 15:40:06.269950: I tensorflow/core/common_runtime/gpu/gpu_device.cc:961] DMA: 0
2018-06-25 15:40:06.269954: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 0:   Y
2018-06-25 15:40:06.269963: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:65:00.0)
Solving...
/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/gradients_impl.py:93: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
Loading initial model weights from data/imagenet_weights/vgg16.ckpt
Variables restored: vgg_16/conv1/conv1_1/biases:0
Variables restored: vgg_16/conv1/conv1_2/weights:0
Variables restored: vgg_16/conv1/conv1_2/biases:0
Variables restored: vgg_16/conv2/conv2_1/weights:0
Variables restored: vgg_16/conv2/conv2_1/biases:0
Variables restored: vgg_16/conv2/conv2_2/weights:0
Variables restored: vgg_16/conv2/conv2_2/biases:0
Variables restored: vgg_16/conv3/conv3_1/weights:0
Variables restored: vgg_16/conv3/conv3_1/biases:0
Variables restored: vgg_16/conv3/conv3_2/weights:0
Variables restored: vgg_16/conv3/conv3_2/biases:0
Variables restored: vgg_16/conv3/conv3_3/weights:0
Variables restored: vgg_16/conv3/conv3_3/biases:0
Variables restored: vgg_16/conv4/conv4_1/weights:0
Variables restored: vgg_16/conv4/conv4_1/biases:0
Variables restored: vgg_16/conv4/conv4_2/weights:0
Variables restored: vgg_16/conv4/conv4_2/biases:0
Variables restored: vgg_16/conv4/conv4_3/weights:0
Variables restored: vgg_16/conv4/conv4_3/biases:0
Variables restored: vgg_16/conv5/conv5_1/weights:0
Variables restored: vgg_16/conv5/conv5_1/biases:0
Variables restored: vgg_16/conv5/conv5_2/weights:0
Variables restored: vgg_16/conv5/conv5_2/biases:0
Variables restored: vgg_16/conv5/conv5_3/weights:0
Variables restored: vgg_16/conv5/conv5_3/biases:0
Variables restored: vgg_16/fc6/biases:0
Variables restored: vgg_16/fc7/biases:0
Loaded.
Fix VGG16 layers..
Fixed.
iter: 20 / 70000, total loss: 2.068716
 >>> rpn_loss_cls: 0.359152
 >>> rpn_loss_box: 0.081183
 >>> loss_cls: 0.882890
 >>> loss_box: 0.613642
 >>> lr: 0.001000
speed: 0.874s / iter
iter: 40 / 70000, total loss: 0.955767
 >>> rpn_loss_cls: 0.693823
 >>> rpn_loss_box: 0.119822
 >>> loss_cls: 0.010253
 >>> loss_box: 0.000000
 >>> lr: 0.001000
speed: 0.765s / iter
iter: 60 / 70000, total loss: 0.971641
 >>> rpn_loss_cls: 0.392060
 >>> rpn_loss_box: 0.063255
 >>> loss_cls: 0.312211
 >>> loss_box: 0.072253
 >>> lr: 0.001000
speed: 0.742s / iter
iter: 80 / 70000, total loss: 1.333313
 >>> rpn_loss_cls: 0.150017
 >>> rpn_loss_box: 0.044684
 >>> loss_cls: 0.660567
 >>> loss_box: 0.346180
 >>> lr: 0.001000
speed: 0.676s / iter
iter: 100 / 70000, total loss: 1.184405
 >>> rpn_loss_cls: 0.107303
 >>> rpn_loss_box: 0.102611
 >>> loss_cls: 0.450133
 >>> loss_box: 0.392485
 >>> lr: 0.001000
speed: 0.647s / iter
......
2018-06-25 15:58:34.140065: I tensorflow/core/common_runtime/gpu/pool_allocator.cc:247] PoolAllocator: After 156814 get requests, put_count=156814 evicted_count=1000 eviction_rate=0.00637698 and unsatisfied allocation rate=0.00701468
2018-06-25 15:58:34.140106: I tensorflow/core/common_runtime/gpu/pool_allocator.cc:259] Raising pool_size_limit_ from 100 to 110
......
Wrote snapshot to: /home/lab/tf-faster-rcnn/output/vgg16/voc_2007_trainval/default/vgg16_faster_rcnn_iter_5000.ckpt
......
Wrote snapshot to: /home/lab/tf-faster-rcnn/output/vgg16/voc_2007_trainval/default/vgg16_faster_rcnn_iter_10000.ckpt
......
Wrote snapshot to: /home/lab/tf-faster-rcnn/output/vgg16/voc_2007_trainval/default/vgg16_faster_rcnn_iter_15000.ckpt
 ......
iter: 70000 / 70000, total loss: 0.306778
 >>> rpn_loss_cls: 0.001028
 >>> rpn_loss_box: 0.099263
 >>> loss_cls: 0.036135
 >>> loss_box: 0.036002
 >>> lr: 0.000100
speed: 0.370s / iter
Wrote snapshot to: /home/lab/tf-faster-rcnn/output/vgg16/voc_2007_trainval/default/vgg16_faster_rcnn_iter_70000.ckpt
done solving
22206.86user 2868.33system 7:12:34elapsed 96%CPU (0avgtext+0avgdata 3774812maxresident)k
1928inputs+32316344outputs (16major+3045793minor)pagefaults 0swaps
+ ./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
+ set -e
+ export PYTHONUNBUFFERED=True
+ PYTHONUNBUFFERED=True
+ GPU_ID=0
+ DATASET=pascal_voc
+ NET=vgg16
+ array=($@)
+ len=3
+ EXTRA_ARGS=
+ EXTRA_ARGS_SLUG=
+ case ${DATASET} in
+ TRAIN_IMDB=voc_2007_trainval
+ TEST_IMDB=voc_2007_test
+ ITERS=70000
+ ANCHORS='[8,16,32]'
+ RATIOS='[0.5,1,2]'
++ date +%Y-%m-%d_%H-%M-%S
+ LOG=experiments/logs/test_vgg16_voc_2007_trainval_.txt.2018-06-25_22-52-31
+ exec
++ tee -a experiments/logs/test_vgg16_voc_2007_trainval_.txt.2018-06-25_22-52-31
+ echo Logging output to experiments/logs/test_vgg16_voc_2007_trainval_.txt.2018-06-25_22-52-31
Logging output to experiments/logs/test_vgg16_voc_2007_trainval_.txt.2018-06-25_22-52-31
+ set +x
+ [[ ! -z '' ]]
+ CUDA_VISIBLE_DEVICES=0
+ time python ./tools/test_net.py --imdb voc_2007_test --model output/vgg16/voc_2007_trainval/default/vgg16_faster_rcnn_iter_70000.ckpt --cfg experiments/cfgs/vgg16.yml --net vgg16 --set ANCHOR_SCALES '[8,16,32]' ANCHOR_RATIOS '[0.5,1,2]'
/usr/local/lib/python2.7/dist-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Called with args:
Namespace(cfg_file='experiments/cfgs/vgg16.yml', comp_mode=False, imdb_name='voc_2007_test', max_per_image=100, model='output/vgg16/voc_2007_trainval/default/vgg16_faster_rcnn_iter_70000.ckpt', net='vgg16', set_cfgs=['ANCHOR_SCALES', '[8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]'], tag='')
Using config:
{'ANCHOR_RATIOS': [0.5, 1, 2],
 'ANCHOR_SCALES': [8, 16, 32],
 'DATA_DIR': '/home/lab/tf-faster-rcnn/data',
 'EXP_DIR': 'vgg16',
 'MATLAB': 'matlab',
 'MOBILENET': {'DEPTH_MULTIPLIER': 1.0,
               'FIXED_LAYERS': 5,
               'REGU_DEPTH': False,
               'WEIGHT_DECAY': 4e-05},
 'PIXEL_MEANS': array([[[102.9801, 115.9465, 122.7717]]]),
 'POOLING_MODE': 'crop',
 'POOLING_SIZE': 7,
 'RESNET': {'FIXED_BLOCKS': 1, 'MAX_POOL': False},
 'RNG_SEED': 3,
 'ROOT_DIR': '/home/lab/tf-faster-rcnn',
 'RPN_CHANNELS': 512,
 'TEST': {'BBOX_REG': True,
          'HAS_RPN': True,
          'MAX_SIZE': 1000,
          'MODE': 'nms',
          'NMS': 0.3,
          'PROPOSAL_METHOD': 'gt',
          'RPN_NMS_THRESH': 0.7,
          'RPN_POST_NMS_TOP_N': 300,
          'RPN_PRE_NMS_TOP_N': 6000,
          'RPN_TOP_N': 5000,
          'SCALES': [600],
          'SVM': False},
 'TRAIN': {'ASPECT_GROUPING': False,
           'BATCH_SIZE': 256,
           'BBOX_INSIDE_WEIGHTS': [1.0, 1.0, 1.0, 1.0],
           'BBOX_NORMALIZE_MEANS': [0.0, 0.0, 0.0, 0.0],
           'BBOX_NORMALIZE_STDS': [0.1, 0.1, 0.2, 0.2],
           'BBOX_NORMALIZE_TARGETS': True,
           'BBOX_NORMALIZE_TARGETS_PRECOMPUTED': True,
           'BBOX_REG': True,
           'BBOX_THRESH': 0.5,
           'BG_THRESH_HI': 0.5,
           'BG_THRESH_LO': 0.0,
           'BIAS_DECAY': False,
           'DISPLAY': 20,
           'DOUBLE_BIAS': True,
           'FG_FRACTION': 0.25,
           'FG_THRESH': 0.5,
           'GAMMA': 0.1,
           'HAS_RPN': True,
           'IMS_PER_BATCH': 1,
           'LEARNING_RATE': 0.001,
           'MAX_SIZE': 1000,
           'MOMENTUM': 0.9,
           'PROPOSAL_METHOD': 'gt',
           'RPN_BATCHSIZE': 256,
           'RPN_BBOX_INSIDE_WEIGHTS': [1.0, 1.0, 1.0, 1.0],
           'RPN_CLOBBER_POSITIVES': False,
           'RPN_FG_FRACTION': 0.5,
           'RPN_NEGATIVE_OVERLAP': 0.3,
           'RPN_NMS_THRESH': 0.7,
           'RPN_POSITIVE_OVERLAP': 0.7,
           'RPN_POSITIVE_WEIGHT': -1.0,
           'RPN_POST_NMS_TOP_N': 2000,
           'RPN_PRE_NMS_TOP_N': 12000,
           'SCALES': [600],
           'SNAPSHOT_ITERS': 5000,
           'SNAPSHOT_KEPT': 3,
           'SNAPSHOT_PREFIX': 'vgg16_faster_rcnn',
           'STEPSIZE': [30000],
           'SUMMARY_INTERVAL': 180,
           'TRUNCATED': False,
           'USE_ALL_GT': True,
           'USE_FLIPPED': True,
           'USE_GT': False,
           'WEIGHT_DECAY': 0.0001},
 'USE_E2E_TF': True,
 'USE_GPU_NMS': True}
2018-06-25 22:52:32.542305: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2018-06-25 22:52:32.542322: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2018-06-25 22:52:32.542325: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2018-06-25 22:52:32.542328: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2018-06-25 22:52:32.542331: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX512F instructions, but these are available on your machine and could speed up CPU computations.
2018-06-25 22:52:32.542334: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2018-06-25 22:52:32.678648: I tensorflow/core/common_runtime/gpu/gpu_device.cc:940] Found device 0 with properties:
name: GeForce GTX 1080 Ti
major: 6 minor: 1 memoryClockRate (GHz) 1.6325
pciBusID 0000:65:00.0
Total memory: 10.91GiB
Free memory: 10.31GiB
2018-06-25 22:52:32.678669: I tensorflow/core/common_runtime/gpu/gpu_device.cc:961] DMA: 0
2018-06-25 22:52:32.678673: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 0:   Y
2018-06-25 22:52:32.678682: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:65:00.0)
Loading model check point from output/vgg16/voc_2007_trainval/default/vgg16_faster_rcnn_iter_70000.ckpt
Loaded.
im_detect: 1/4952 1.043s 0.024s
im_detect: 2/4952 0.883s 0.020s
im_detect: 3/4952 0.810s 0.021s
im_detect: 4/4952 0.769s 0.022s
im_detect: 5/4952 0.638s 0.022s
im_detect: 6/4952 0.550s 0.022s
......
im_detect: 4951/4952 0.152s 0.024s
im_detect: 4952/4952 0.152s 0.024s
Evaluating detections
Writing aeroplane VOC results file
Writing bicycle VOC results file
Writing bird VOC results file
Writing boat VOC results file
Writing bottle VOC results file
Writing bus VOC results file
Writing car VOC results file
Writing cat VOC results file
Writing chair VOC results file
Writing cow VOC results file
Writing diningtable VOC results file
Writing dog VOC results file
Writing horse VOC results file
Writing motorbike VOC results file
Writing person VOC results file
Writing pottedplant VOC results file
Writing sheep VOC results file
Writing sofa VOC results file
Writing train VOC results file
Writing tvmonitor VOC results file
VOC07 metric? Yes
AP for aeroplane = 0.7237
AP for bicycle = 0.7685
AP for bird = 0.6977
AP for boat = 0.5475
AP for bottle = 0.5412
AP for bus = 0.7771
AP for car = 0.8191
AP for cat = 0.8072
AP for chair = 0.5372
AP for cow = 0.8035
AP for diningtable = 0.6599
AP for dog = 0.7951
AP for horse = 0.8279
AP for motorbike = 0.7402
AP for person = 0.7776
AP for pottedplant = 0.4276
AP for sheep = 0.7101
AP for sofa = 0.6530
AP for train = 0.7269
AP for tvmonitor = 0.7189
Mean AP = 0.7030
~~~~~~~~
Results:
0.724
0.768
0.698
0.547
0.541
0.777
0.819
0.807
0.537
0.803
0.660
0.795
0.828
0.740
0.778
0.428
0.710
0.653
0.727
0.719
0.703
~~~~~~~~

--------------------------------------------------------------
Results computed with the **unofficial** Python eval code.
Results should be very close to the official MATLAB eval code.
Recompute with `./tools/reval.py --matlab ...` for your paper.
-- Thanks, The Management
--------------------------------------------------------------
767.79user 170.55system 15:01.74elapsed 104%CPU (0avgtext+0avgdata 1838380maxresident)k
240inputs+103984outputs (0major+499344minor)pagefaults 0swaps

```
