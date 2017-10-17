---
layout:     post                    # 使用的布局（不需要改）
title:      Convolutional Neural Network in MINIST Project           # 标题 
subtitle:   Convolutional Neural Network #副标题
date:       2017-09-20             # 时间
author:     Brian                      # 作者
header-img: img/post-bg-universe.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Kaggle Competition
---

# MINIST Project - Kaggle Competition
Author: Xie Zhong-zhao

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
```


```python
'''
设置变量的值
'''
Learning_rate = 1e-4
Training_iterations = 2500
Dropout = 0.5
Batch_size = 50

Validation_size = 2000
Image_to_display = 10
```


```python
'''
数据预处理
'''
data = pd.read_csv('train.csv')

print('data({0[0]},{0[1]})'.format(data.shape))
print(data.head())
```

    data(42000,785)
       label  pixel0  pixel1  pixel2  pixel3  pixel4  pixel5  pixel6  pixel7  \
    0      1       0       0       0       0       0       0       0       0   
    1      0       0       0       0       0       0       0       0       0   
    2      1       0       0       0       0       0       0       0       0   
    3      4       0       0       0       0       0       0       0       0   
    4      0       0       0       0       0       0       0       0       0   
    
       pixel8    ...     pixel774  pixel775  pixel776  pixel777  pixel778  \
    0       0    ...            0         0         0         0         0   
    1       0    ...            0         0         0         0         0   
    2       0    ...            0         0         0         0         0   
    3       0    ...            0         0         0         0         0   
    4       0    ...            0         0         0         0         0   
    
       pixel779  pixel780  pixel781  pixel782  pixel783  
    0         0         0         0         0         0  
    1         0         0         0         0         0  
    2         0         0         0         0         0  
    3         0         0         0         0         0  
    4         0         0         0         0         0  
    
    [5 rows x 785 columns]



```python
images = data.iloc[:,1:].values
images = images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)
print(images)
print('images({0[0]},{0[1]})'.format(images.shape))
```

    [[ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     ..., 
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]]
    images(42000,784)



```python
image_size = images.shape[1]
print('image_size => {0}'.format(image_size))

# in this case all images are square
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

print('image_width => {0}\nimage_height => {1}'.format(image_width,image_height))
```

    image_size => 784
    image_width => 28
    image_height => 28



```python
# display image
def display(img):
    # (784) => (28,28)
    one_image = img.reshape(image_width,image_height)

    plt.axis('off') #关闭坐标轴显示
    plt.imshow(one_image, cmap=cm.binary)
    plt.show()

# output image     
display(images[Image_to_display])
print(images[Image_to_display].shape)
```


![](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/MINIST%2BProject%2B-%2BKaggle%2BCompetition/output_6_0.png)


    (784,)
​    


```python
#选择第一列，value代表标签的值0-9，ravel变成列表[1 0 1,...,7 6 9]
labels_flat = data[[0]].values.ravel()
print(data[[0]].values.ravel())

print('labels_flat({0})'.format(len(labels_flat)))
print ('labels_flat[{0}] => {1}'.format(Image_to_display,labels_flat[Image_to_display]))
```

    [1 0 1 ..., 7 6 9]
    labels_flat(42000)
    labels_flat[10] => 8



```python
labels_count = np.unique(labels_flat).shape[0]
print(np.unique(labels_flat))
print(np.unique(labels_flat).shape[0])

print('labels_count => {0}'.format(labels_count))
```

    [0 1 2 3 4 5 6 7 8 9]
    10
    labels_count => 10



```python
# convert class labels from scalars to one-hot vectors
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
# ...
# 9 => [0 0 0 0 0 0 0 0 0 1]
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    print('num_labels = ',num_labels)
    index_offset = np.arange(num_labels) * num_classes
    print('np.arange(num_labels) = ',np.arange(num_labels))
    print('index_offset = ',index_offset)
    labels_one_hot = np.zeros((num_labels, num_classes))
    print('labels_one_hot = ',labels_one_hot)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1 #??????
    print('labels_one_hot.flat[] = ',labels_one_hot.flat[index_offset + labels_dense.ravel()])
    print('labels_dense.ravel() = ',labels_dense.ravel())
    return labels_one_hot

labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

print('labels({0[0]},{0[1]})'.format(labels.shape))
print ('labels[{0}] => {1}'.format(Image_to_display,labels[Image_to_display]))
```

    num_labels =  42000
    np.arange(num_labels) =  [    0     1     2 ..., 41997 41998 41999]
    index_offset =  [     0     10     20 ..., 419970 419980 419990]
    labels_one_hot =  [[ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     ..., 
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]]
    labels_one_hot.flat[] =  [ 1.  1.  1. ...,  1.  1.  1.]
    labels_dense.ravel() =  [1 0 1 ..., 7 6 9]
    labels(42000,10)
    labels[10] => [0 0 0 0 0 0 0 0 1 0]



```python
# split data into training & validation
validation_images = images[:Validation_size]
validation_labels = labels[:Validation_size]

train_images = images[Validation_size:]
train_labels = labels[Validation_size:]

print('train_images({0[0]},{0[1]})'.format(train_images.shape))
print('validation_images({0[0]},{0[1]})'.format(validation_images.shape))
```

    train_images(40000,784)
    validation_images(2000,784)



```python
#将数据分割成训练集和验证集
validation_images = images[:Validation_size]
validation_labels = labels[:Validation_size]

train_images = images[Validation_size:]
train_labels = labels[Validation_size:]

print('train_image({0[0]},{0[1]})'.format(train_images.shape))
print('validation_images({0[0]},{0[1]})'.format(validation_images.shape))
```

    train_image(40000,784)
    validation_images(2000,784)



```python
'''
创建卷积神经网络的权重和偏置
(1)我们需要给权重制造随机噪声来打破完全对称,比如截断的正态分布噪声,标准差设为0.1
(2)使用ReLU给偏置增加一些小的正值(0.1)用来避免死亡节点
'''
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

'''
创建卷积层和池化层
(1)参数x为输入,W为卷积参数,[5,5,1,32]:前面两个数字代表卷积尺寸,第三个数字代表有多少个channel,灰度单色为1,彩色RGB图片为3,
最后一个数字代表卷积数量.
(2)Strides代表模板卷积模板移动的步长,都是1代表会不遗漏地划过图片的每一个点.
(3)Padding代表边界的处理方式,这里的SAME代表给边界加上Padding让卷积的输出和输入保持同样的(SAME)的尺寸.
'''
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

'''
定义placeholder,x是特征,y_是真实的label
(1)卷积神经网络会利用空间信息,因此需要将1D的输入向量转化为2D图片结构,1*784转化原始的28*28,只有一个颜色通道,
最终尺寸[-1,28,28,1],前面的-1代表样本数量不固定,最后1代表颜色通道数量.
'''
x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
x_image = tf.reshape(x,[-1,28,28,1])

'''
定义第一个卷积层:
(1)参数初始化,包括weights和bias,[5,5,1,32]代表卷积核尺寸为5*5,1个颜色通道,32个不同卷积核.
(2)使用conv2d进行卷积,再加上偏置.
(3)接着使用ReLU激活函数进行非线性处理.
(4)最后用最大池化函数max_pool_2x2对卷积的结果进行池化.
'''
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
print(h_pool1.shape)

'''
定义第二个卷积层:
(1)参数初始化,包括weights和bias,[5,5,1,64]代表卷积核尺寸为5*5,1个颜色通道,64个不同卷积核.
(2)使用conv2d进行卷积,再加上偏置.
(3)接着使用ReLU激活函数进行非线性处理.
(4)最后用最大池化函数max_pool_2x2对卷积的结果进行池化.
'''
'''
经历两次步长为2*2的最大池化,图片尺寸28*28变成7*7,由于第二层卷积核数量为64,输出尺寸即为7*7*64
'''
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
print(h_pool2.shape)

'''
定义一个全连接层:
(1)将第二层卷积的输出tensor进行变形,将其转化为1D向量,然后连接一个全连接层,隐含节点为1024
(2)并使用ReLU激活函数
'''
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
print(h_fc1.shape)


'''
定义一个Dropout层:
(1)为了减轻过拟合,使用Dropout,通过一个placeholder传入keep_prob比率进行控制.
(2)在训练过程中,我们随机丢弃一部分节点的数据来减轻过拟合.
(3)预测时则保留全部的数据来追求最好的预测性能.
'''
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
print(h_fc1_drop.shape)

'''
最后将Dropout层的输出连接一个Softmax层,得到最后的概率输出
'''
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
print('y_conv.shape',y_conv.shape)
print('tf.argmax(y_conv,1)',tf.argmax(y_conv,1))
print('y_',y_)

'''
定义损失函数cross entropy, 优化器使用Adam, 并给予一个比较小的学习速率1e-4
'''
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
train_optimizer = tf.train.AdamOptimizer(Learning_rate).minimize(cross_entropy)

'''
定义评定准确率的操作:
'''
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# prediction function
#[0.1, 0.9, 0.2, 0.1, 0.1 0.3, 0.5, 0.1, 0.2, 0.3] => 1
predict = tf.argmax(y_conv,1)

'''
随机训练的样本提取
'''
epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

# serve data by batches
def next_batch(batch_size):
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

'''
开始训练过程:
(1)首先初始化所有参数
(2)设置训练时的Dropout的keep_prob比率为0.5
(3)然后使用大小为50的mini-batch,进行20000次的训练迭代,参与训练样本的数量总共为100万
(4)其中每训练100次,对准确率进行一次评测(评测时keep_prob设为1),用于实时监控模型性能
'''
init = tf.initialize_all_variables()  #启动该计算图。
sess = tf.InteractiveSession()
sess.run(init)
```

    (?, 14, 14, 32)
    (?, 7, 7, 64)
    (?, 1024)
    (?, 1024)
    y_conv.shape (?, 10)
    tf.argmax(y_conv,1) Tensor("ArgMax_3:0", shape=(?,), dtype=int64)
    y_ Tensor("Placeholder_4:0", shape=(?, 10), dtype=float32)
    WARNING:tensorflow:From <ipython-input-54-eaffccdcb979>:152: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
    Instructions for updating:
    Use `tf.global_variables_initializer` instead.



```python
#可视化变量
train_accuracies = []
validation_accuracies = []
x_range = []

display_step=1

for i in range(Training_iterations):
    # get new batch
    batch_xs, batch_ys = next_batch(Batch_size)
    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i%display_step == 0 or (i+1) == Training_iterations:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        if(Validation_size):
            validation_accuracy = accuracy.eval(feed_dict={x: validation_images[0:Batch_size], y_: validation_labels[0:Batch_size], keep_prob: 1.0})
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d' % (
            train_accuracy, validation_accuracy, i))
            validation_accuracies.append(validation_accuracy)
        else:
            print('training_accuracy => %.4f for step %d' % (train_accuracy, i))
        train_accuracies.append(train_accuracy)
        x_range.append(i)

        # increase display_step
        if i % (display_step * 10) == 0 and i:
            display_step *= 10
    # train on batch
    sess.run(train_optimizer, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: Dropout})

```

    training_accuracy / validation_accuracy => 0.08 / 0.06 for step 0
    training_accuracy / validation_accuracy => 0.14 / 0.10 for step 1
    training_accuracy / validation_accuracy => 0.12 / 0.14 for step 2
    training_accuracy / validation_accuracy => 0.14 / 0.24 for step 3
    training_accuracy / validation_accuracy => 0.28 / 0.28 for step 4
    training_accuracy / validation_accuracy => 0.24 / 0.20 for step 5
    training_accuracy / validation_accuracy => 0.18 / 0.22 for step 6
    training_accuracy / validation_accuracy => 0.24 / 0.20 for step 7
    training_accuracy / validation_accuracy => 0.28 / 0.24 for step 8
    training_accuracy / validation_accuracy => 0.46 / 0.28 for step 9
    training_accuracy / validation_accuracy => 0.30 / 0.34 for step 10
    training_accuracy / validation_accuracy => 0.44 / 0.46 for step 20
    training_accuracy / validation_accuracy => 0.66 / 0.68 for step 30
    training_accuracy / validation_accuracy => 0.74 / 0.80 for step 40
    training_accuracy / validation_accuracy => 0.64 / 0.70 for step 50
    training_accuracy / validation_accuracy => 0.82 / 0.88 for step 60
    training_accuracy / validation_accuracy => 0.84 / 0.90 for step 70
    training_accuracy / validation_accuracy => 0.76 / 0.90 for step 80
    training_accuracy / validation_accuracy => 1.00 / 0.90 for step 90
    training_accuracy / validation_accuracy => 0.84 / 0.92 for step 100
    training_accuracy / validation_accuracy => 0.88 / 0.90 for step 200
    training_accuracy / validation_accuracy => 0.92 / 0.92 for step 300
    training_accuracy / validation_accuracy => 0.88 / 0.92 for step 400
    training_accuracy / validation_accuracy => 0.96 / 0.90 for step 500
    training_accuracy / validation_accuracy => 0.98 / 0.94 for step 600
    training_accuracy / validation_accuracy => 0.96 / 0.94 for step 700
    training_accuracy / validation_accuracy => 1.00 / 0.92 for step 800
    training_accuracy / validation_accuracy => 1.00 / 0.94 for step 900
    training_accuracy / validation_accuracy => 0.88 / 0.92 for step 1000
    training_accuracy / validation_accuracy => 1.00 / 0.96 for step 2000
    training_accuracy / validation_accuracy => 1.00 / 0.98 for step 2499



```python
'''
检验在验证集上最后的准确率
'''
 # check final accuracy on validation set
if (Validation_size):
    validation_accuracy = accuracy.eval(feed_dict={x: validation_images,
                                                   y_: validation_labels,
                                                   keep_prob: 1.0})
    print('validation_accuracy => %.4f' % validation_accuracy)
    plt.plot(x_range, train_accuracies, '-b', label='Training')
    plt.plot(x_range, validation_accuracies, '-g', label='Validation')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymax=1.1, ymin=0.7)
    plt.ylabel('accuracy')
    plt.xlabel('step')
    plt.show()
```

    validation_accuracy => 0.9790
​    


![](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/MINIST%2BProject%2B-%2BKaggle%2BCompetition/output_14_1.png)



```python
# read test data from CSV file 
test_images = pd.read_csv('test.csv').values
test_images = test_images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
test_images = np.multiply(test_images, 1.0 / 255.0)

print('test_images({0[0]},{0[1]})'.format(test_images.shape))


# predict test set
#predicted_lables = predict.eval(feed_dict={x: test_images, keep_prob: 1.0})

# using batches is more resource efficient
predicted_lables = np.zeros(test_images.shape[0])
for i in range(0,test_images.shape[0]//Batch_size):
    predicted_lables[i*Batch_size : (i+1)*Batch_size] = predict.eval(
        feed_dict={x: test_images[i*Batch_size : (i+1)*Batch_size], keep_prob: 1.0})


print('predicted_lables({0})'.format(len(predicted_lables)))

# output test image and prediction
display(test_images[Image_to_display])
print ('predicted_lables[{0}] => {1}'.format(Image_to_display,predicted_lables[Image_to_display]))

# save results
np.savetxt('submission_softmax.csv', 
           np.c_[range(1,len(test_images)+1),predicted_lables], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')
```

    test_images(28000,784)
    predicted_lables(28000)



![](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/MINIST%2BProject%2B-%2BKaggle%2BCompetition/output_15_1.png)


    predicted_lables[10] => 5.0
​    
