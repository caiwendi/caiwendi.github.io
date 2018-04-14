---
layout:     post                    # 使用的布局（不需要改）
title:      【搬运】Softmax Regression,AutoEncoder Network # 标题 
subtitle:   Softmax regression,AutoEncoder #副标题
date:       2017-06-08              # 时间
author:     Brian                      # 作者
header-img: img/post-bg-os-metro.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Tensorflow
---

# 1. Tensorflow学习笔记

学习<Tensorflow实战>个人笔记,共大家学习交流,欢迎拍砖  作者: 谢中朝

```python
import tensorflow as tf
import numpy as np

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100)) # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 构造一个线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in xrange(0, 201):
        sess.run(train)
        if step % 20== 0:
                print (step, sess.run(W), sess.run(b))
sess.close()
# 得到最佳拟合结果 W: [[0.1000.200]], b: [0.300]
```

    WARNING:tensorflow:From <ipython-input-34-3188aa8b385d>:19: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
    Instructions for updating:
    Use `tf.global_variables_initializer` instead.
    0 [[ 0.06856041  0.56634241]] [ 0.2438574]
    20 [[ 0.09701308  0.28646886]] [ 0.25346363]
    40 [[ 0.10288773  0.22429344]] [ 0.28499389]
    60 [[ 0.10161395  0.20702811]] [ 0.29525524]
    80 [[ 0.10063458  0.2020756 ]] [ 0.29851687]
    100 [[ 0.10022143  0.20062158]] [ 0.29953957]
    120 [[ 0.10007308  0.20018786]] [ 0.29985765]
    140 [[ 0.10002342  0.20005712]] [ 0.29995611]
    160 [[ 0.10000737  0.20001741]] [ 0.29998651]
    180 [[ 0.1000023   0.20000532]] [ 0.29999584]
    200 [[ 0.10000069  0.20000161]] [ 0.29999873]


```python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print (sess.run(hello))

a = tf.constant(10)
b = tf.constant(32)
print (sess.run(a+b) )
```

    Hello, TensorFlow!
    42

# 2. Softmax Regression手写识别

1. 定义算法公式,也就是神经网络forward时计算  
2. 定义loss, 选定优化器,并指定优化器优化loss  
3. 迭代地对数据进行训练
4. 在测试集或者验证集上对准确率进行评测  


```python
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

    Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz


```python
print mnist.train.images.shape
print mnist.train.labels.shape
print mnist.validation.images.shape
print mnist.validation.labels.shape
print mnist.test.images.shape
print mnist.test.labels.shape
```

    (55000, 784)
    (55000, 10)
    (5000, 784)
    (5000, 10)
    (10000, 784)
    (10000, 10)


```python
import os
import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()  #之后的运算全部会跑在这个session里,相当运行空间
x = tf.placeholder(tf.float32, [None, 784])  #输入数据的地方
W = tf.Variable(tf.zeros([784, 10])) #Variable用来存储模型参数
b = tf.Variable(tf.zeros([10]))

#实现Softmax Regression算法
y  = tf.nn.softmax(tf.matmul(x, W) + b )

#定义cross-entropy
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
 
#SGD随机梯度下降,优化cross_entropy目标,学习速率为0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#全局参数初始化器
tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print (accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
    
```

    0.9204

# 3. 自编码器及多层感知机

## 3.1 自编码器


```python
#coding=utf-8
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#深度学习模型的权重初始化得太小,那么信号将在每层间传递时逐渐缩小而
#难以产生作用,如果权值初始化得过大,那么信号将在每层传递时逐渐放大导
#致发散和失效,Xavier就是让权值满足均值为0,方差2/(Nin+Nout),分布可以采
#用均匀分布或者高斯分布.

#根据某一层网络的输入,输出节点数量自动调整最合适的分布
def xavier_init(fan_in, fan_out, constant = 1):
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out) )
        high = constant * np.sqrt(6.0 / (fan_in + fan_out) )
        return tf.random_uniform( (fan_in, fan_out), 
                         minval = low, maxval = high, dtype = tf.float32 )
    
#去噪编码的class
#包括神经网络设计,权重初始化,以及常用的成员函数
class AdditiveGaussianNoiseAutoencoder(object):
    
        #定义__init__函数包括: n_inpt输入变量,n_hidden隐含层节点数,transfer_function隐含层激活函数(默认为softplus),
        #optimizer优化器(默认为Adam),scale为高斯噪声系数(默认为0.1),以下只用了一个隐藏层,也可以自行添加
        def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                    optimizer=tf.train.AdamOptimizer(), scale = 0.1):
                self.n_input = n_input
                self.n_hidden = n_hidden
                self.transfer = transfer_function
                self.scale = tf.placeholder(tf.float32)
                self.training_scale = scale
                network_weights = self._initialize_weights()
                self.weights = network_weights
                
                #定义网络结构: 
                #(1)输入x创建一个维度为n_input的placeholder,
                #(2)然后建立一个提取特征隐藏层:将输入x加上噪声,即self.x+scale*tf.random_normal((n_input,));然后用
                #tf.matmul将噪声的输入与隐藏层w1相乘,并使用tf.add加上隐藏层的偏置b1;最后self.tranfer对结果进行
                #激活函数处理,经过隐藏层后,进行数据复原和重建操作.
                self.x = tf.placeholder(tf.float32, [None, self.n_input])
                self.hidden = self.transfer(tf.add( tf.matmul(
                            self.x + scale * tf.random_normal((n_input,)), 
                            self.weights['w1']), self.weights['b1']) )
                self.reconstruction = tf.add(tf.matmul(self.hidden,
                            self.weights['w2']), self.weights['b2']) 
                  
                #定义自编码的损失函数:
                #(1)直接用平方误差(squared error)作为cost,用tf.substract计算输出self.reconstruction与self.x之差,
                #再使用tf.pow求差的平方,最后用tf.reduce_sum求和即可得到平方误差和
                #(2)训练操作作为优化器self.optimizer对损失self.cost进行优化
                #(3)最后创建Session,并初始化自编码的全部模型参数
                self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(
                            self.reconstruction, self.x), 2.0))
                self.optimizer = optimizer.minimize(self.cost)
                
                init = tf.global_variables_initializer()
                self.sess = tf.Session()
                self.sess.run(init)
                 
        #定义参数初始化函数_initialize_weights
        #(1)创建all_weights字典包括w1,b1,w2,b2,其中w1需要xavier_init函数初始化,传入输入节点数和隐藏层节点数,
        # 返回一个比较适合softplus等激活函数的权重初始分布,偏置b1使用tf.zeros全部置为0
        #(2)输出层self.reconstruction,因为没有激活函数,这里将w2,b2全部初始化为0
        def _initialize_weights(self):
                all_weights = dict()
                all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
                all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
                all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype = tf.float32))
                all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))
                return all_weights
            
        #定义计算损失cost及执行一步训练的函数partial_fit
        #(1)Session执行两个计算图的节点,分别是损失cost和训练过程optimizer;输入的feed_dict包括
        #数据x,以及噪声系数scale
        #(2)partial_fit用一个batch数据进行训练并返回当前的损失cost
        def partial_fit(self, X):
                cost, opt = self.sess.run((self.cost, self.optimizer),
                                         feed_dict = {self.x: X, self.scale: self.training_scale} )
                return cost
        
        #定义只求损失cost函数calc_total_cost
        #(1)在自编码训练完毕后,在测试集上对模型性能进行评测时会用到
        def calc_total_cost(self, X):
                return  self.sess.run(self.cost,  
                                      feed_dict = {self.x: X,  self.scale: self.training_scale} )
      
        #定义transform函数,
        #(1)它返回自编码器隐藏层的输出结果自编码的隐藏层主要功能就是学习出数据中的高阶特征
        def transform(self, X):
                return  self.sess.run(self.hidden, 
                                      feed_dict = {self.x: X, self.scale: self.training_scale} )
        
        #定义一个generate函数
        #(1)将隐藏层的输出结果作为输入,通过之后的重建层将提取到的 高阶特征恢复为原始数据
        def generate(self, hidden = None):
                if hidden is None:
                        hidden = np.random.normal(size = self.weights["b1"])
                return self.sess.run(self.reconstruction,
                                    feed_dict = {self.hidden: hidden})
            
        #定义reconstruct函数
        #(1)整体运行一遍复原过程,包括提取高阶特征和通过高阶特征恢复数据,包括tansform和generate两块
        #(2)输入数据为原数据,输出数据是复原后的数据
        def reconstruct(self, X):
                return self.sess.run(self.reconstruction, 
                                     feed_dict = {self.x: X, self.scale: self.training_scale})
            
        #定义getWeights函数: 获取隐藏层的权重w1
        def getWeights(self):
                return self.sess.run(self.weight['w1'])
            
        #定义getBiases函数: 获取隐藏层的权偏置系数b1
        def getBiases(self):
                return self.sess.run(self.weights['b1'])
```

    /usr/lib/python2.7/dist-packages/simplejson/encoder.py:262: DeprecationWarning: Interpreting naive datetime as local 2017-04-25 12:05:28.449682. Please add timezone info to timestamps.
      chunks = self.iterencode(o, _one_shot=True)


```python
#载入MNIST数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

'''
定义一个对训练,测试数据进行标准化处理的函数
(1)标准化就是让数据变成0均值,且标准差为1的分布
(2)必须保证训练,测试数据都使用完全相同的scaler
'''
def standard_scale(X_train, X_test):
        preprocessor = prep.StandardScaler().fit(X_train)
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)
        return X_train, X_test

'''
定义一个获取随机block数据函数
(1)取一个从0到len(data)-batch_size之间的随机整数,再以这个随机数作为block的起始位置,
然后顺序取到一个batch size的数据
(2)这属于不放回抽样,可以提高数据利用率
'''
def get_random_block_from_data(data, batchsize):
        start_index = np.random.randint(0, len(data) - batchsize)
        return data[start_index : (start_index + batchsize)]

'''
对训练集和测试集进行标准化变换
'''    
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

'''
常用参数: 总样本训练数, 最大训练的轮数(epoch)设为20, batch_size设为128, 
并设置每一轮(epoch)就显示一次cost
'''
n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1

'''
创建一个AGN自编码的实例:
(1)定义模型输入节点数n_input为784
(2)自编码的隐藏层节点数n_hidden为200
(3)隐藏层的激活函数transfer_function为softplus
(4)优化器optimizer为Adam且学习速率为0.001,同时噪声系数scale设置0.01
'''
autoencoder = AdditiveGaussianNoiseAutoencoder(
                    n_input = 784, n_hidden = 200,
                    transfer_function = tf.nn.softplus,
                    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001), scale = 0.01 )
'''
下面开始训练过程:
(1)在每一轮(epoch)循环开始,我们将平均损失avg_cost设为0,并计算总共需要的batch数目
(2)每个batch循环中,先使用get_random_block_from_data函数随机抽取一个block数据,然后
使用成员函数partial_fit训练这个batch的数据并计算当前的cost,最后将当前的cost整合到avg_cost中
(3)每一次迭代后,显示当前的迭代数和每一轮迭代的平均cost
'''
for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(n_samples / batch_size)
        for i  in range(total_batch):
                batch_xs = get_random_block_from_data(X_train, batch_size)
                cost = autoencoder.partial_fit(batch_xs)
                avg_cost += cost / n_samples * batch_size
                
        if epoch % display_step == 0:
                print("epoch:", '%04d' %(epoch + 1), "cost=",
                "{: .9f}".format(avg_cost) )
#print ("Total cost: " + str(autoencoder.calc_total_cost(X_test)))

```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
    ('epoch:', '0001', 'cost=', ' 19536.334405682')
    ('epoch:', '0002', 'cost=', ' 12002.419304545')
    ('epoch:', '0003', 'cost=', ' 11700.977278977')
    ('epoch:', '0004', 'cost=', ' 9684.620906250')
    ('epoch:', '0005', 'cost=', ' 9680.535675568')
    ('epoch:', '0006', 'cost=', ' 9375.237253409')
    ('epoch:', '0007', 'cost=', ' 9421.320490909')
    ('epoch:', '0008', 'cost=', ' 8612.968441477')
    ('epoch:', '0009', 'cost=', ' 9112.435678409')
    ('epoch:', '0010', 'cost=', ' 8507.437596591')
    ('epoch:', '0011', 'cost=', ' 7996.247656250')
    ('epoch:', '0012', 'cost=', ' 8450.895822727')
    ('epoch:', '0013', 'cost=', ' 8452.416527273')
    ('epoch:', '0014', 'cost=', ' 8680.766208523')
    ('epoch:', '0015', 'cost=', ' 7925.744233523')
    ('epoch:', '0016', 'cost=', ' 8090.308035795')
    ('epoch:', '0017', 'cost=', ' 7730.277733523')
    ('epoch:', '0018', 'cost=', ' 8050.471460795')
    ('epoch:', '0019', 'cost=', ' 8168.497669318')
    ('epoch:', '0020', 'cost=', ' 8041.845478977')

    /usr/lib/python2.7/dist-packages/simplejson/encoder.py:262: DeprecationWarning: Interpreting naive datetime as local 2017-04-25 12:05:34.097629. Please add timezone info to timestamps.
      chunks = self.iterencode(o, _one_shot=True)

## 3.2 多层感知机

1. 定义算法公式,也就是神经网络forward时计算  
2. 定义loss, 选定优化器,并指定优化器优化loss  
3. 迭代地对数据进行训练
4. 在测试集或者验证集上对准确率进行评测  


```python
###(1)定义算法公式(即神经网络forward时计算)
'''
载入Tensorflow并加载MNIST数据集,创建一个Tensorflow默认Interactive Session,
这样后面执行各项操作就无需指定Session
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
sess = tf.InteractiveSession()

'''
给隐藏层的参数设置Variable并进行初始化
'''
in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev = 0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

'''
定义输入x的placeholder:
(1)Dropout的比率keep_prob(保留节点的概率)是不一样的
(2)通常训练时小于1,而预测时则等于1
'''
x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)

'''
定义模型结构:
(1)需要一个隐藏层,命名为hidden1,可以通过tf.nn.relu(tf.matmul(x, W1)+b1)实现一个激活函数为ReLU的隐含层(y = relu(W1x+b))
(2)调用tf.nn.dropout实现Dropout功能,训练时小于1,制造随机性,防止过拟合,预测时等于1.用全部特征来预测样本的类别
(3)softmax分类
'''
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

###(2)定义损失函数和选择优化器来优化loss,这里损失函数继续使用交叉熵,优化器选择自适应的优化器Adagrad
'''
定义损失函数和选择优化器来优化loss
(1)损失函数: 交叉熵
(2)优化器:Adagrad(也可以使用Adadelta,Adam等优化器,当然学习速率需要调整)
'''
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

###(3)训练过程
'''
训练过程: 加入了keep_prob作为计算图的输入,并且在训练时设为0.75,即保留75%的节点,其余的25%置为0
一般来说,对于越复杂越大的规模的神经网络,Dropout的效果尤其明显;因为加入了隐含层,我们需要更多次的
迭代来优化模型参数以达到比较好的效果
一共采用3000个batch,每个batch包括100条样本,一共30万的样本,相当于对全部数据集进行了5轮(epoch)迭代
'''
tf.global_variables_initializer().run()
for i in range(3000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

###(4)对模型进行准确率的评测
'''
预测部分,直接令keep_prob等于1即可,这样可以达到模型最好的预测效果
'''
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1) )
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
    0.9782

    /usr/lib/python2.7/dist-packages/simplejson/encoder.py:262: DeprecationWarning: Interpreting naive datetime as local 2017-04-25 13:21:33.208661. Please add timezone info to timestamps.
      chunks = self.iterencode(o, _one_shot=True)
