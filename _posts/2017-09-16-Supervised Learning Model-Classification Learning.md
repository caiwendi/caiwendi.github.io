---
layout:     post                    # 使用的布局（不需要改）
title:      Supervised Learning Model           # 标题 
subtitle:   Classification Learning #副标题
date:       2017-02-06              # 时间
author:     Brian                      # 作者
header-img: img/post-bg-2015.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Supervised Learning
---

# Supervised Learning Model-Classification Learning
Author: Xie Zhong-zhao

## 1. Logisitic Regression


```python
#导入pandas与numpy工具包
import pandas as pd
import numpy as np

#创建特征列表
column_names = ['sample code number','clump thickness','uniformity of cell size','uniformity of cell shape',
                'marginal adhesion','single epithelial cell size','bare nuclei','bland chromatin','normal nucleoli',
                'mitoses','class']

'''
#使用pandas.read_csv函数从互联网读取指定数据
'''
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
                   , names = column_names)

#打印完整数据的行列数
print(data.shape)

#将？替换为标准缺失值表示
data = data.replace(to_replace= '?', value=np.nan)
#丢弃带有缺失值的数据（只要有一个维度有缺失）
data = data.dropna(how = 'any')
#输出data的数据量和维度
print(data.shape)

'''
#准备训练数据和测试数据，通常情况下，25%的数据会做测试集，其余75%的数据用作训练集
'''
#使用sklearn.cross_validation里面的train_test_split模块用于数据分割
from sklearn.cross_validation import train_test_split
#随机采样25%的数据用于测试，剩下75%用于构建训练集合
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)

#查验训练样本的数量和类别分布
print(y_train.value_counts())
#查验测试样本的数量和类别分布
print(y_test.value_counts())


#从sklearn.preprocessing里导入standarscaler
from sklearn.preprocessing import StandardScaler
#从sklearn.linear_model里导入LogisticRegression与SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

'''
#标准化数据，保证每一个维度的特征数据的方差为1，均值为0，使得预测结果不会被某些维度过大的特征值而主导
'''
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

'''
初始化LogisticRegression与SGDClassifier
'''
lr = LogisticRegression()
sgdc = SGDClassifier()
#调用LogisticRegression中的fit函数/模块用来训练参数模型参数
lr.fit(X_train,y_train)
#使用训练好的模型lr对X_test进行预测，结果存储在变量lr_y_predict中
lr_y_predict = lr.predict(X_test)
#调用SGDClassifier中的fit函数/模块来训练参数模型参数
sgdc.fit(X_train,y_train)
#使用训练好的模型sgdc对X_test进行预测，结果储存在变量sgdc_y_predict中
sgdc_y_predict = sgdc.predict(X_test)

'''
使用线性分类模型对预测任务进行性能分析
'''
#从sklearn.metrics里导入classification_report模块
from sklearn.metrics import classification_report

#使用logisticRegression模型自带的评分函数score来获得模型在测试集上准确性结果
print('Accuracy of LR Classifier:', lr.score(X_test,y_test))
#使用classification_report模块来获得LogisticRegression其他三个指标的结果
print(classification_report(y_test,lr_y_predict,target_names=['benign','malignant']))

#使用SGD模型自带的评分函数score来获得模型在测试集上准确性结果
print('Accuracy of SGD Classifier:', sgdc.score(X_test,y_test))
#使用classification_report模块来获得SGD Classifier其他三个指标的结果
print(classification_report(y_test,sgdc_y_predict,target_names=['benign','malignant']))

```

    (699, 11)
    (683, 11)
    2    344
    4    168
    Name: class, dtype: int64
    2    100
    4     71
    Name: class, dtype: int64
    Accuracy of LR Classifier: 0.970760233918
                 precision    recall  f1-score   support
    
         benign       0.96      0.99      0.98       100
      malignant       0.99      0.94      0.96        71
    
    avg / total       0.97      0.97      0.97       171
    
    Accuracy of SGD Classifier: 0.976608187135
                 precision    recall  f1-score   support
    
         benign       0.97      0.99      0.98       100
      malignant       0.99      0.96      0.97        71
    
    avg / total       0.98      0.98      0.98       171
    
    

## 2. Support Vector Classification


```python
#从sklearn.datasets里面导入手写体数字加载器
from sklearn.datasets import load_digits
#将加载的图像数据存储在digits上
digit = load_digits()
#检查数据的规模和特征维度
print(digit.data.shape)

'''
手写体数据分割代码样例
'''
#从sklearn.cross_validation中导入train_test_split用于数据分割
from sklearn.cross_validation import train_test_split
#随机选取75%数据作为训练样本，其余的25%作为测试样本
X_train,X_test,y_train,y_test = train_test_split(digit.data,digit.target,test_size=0.25,random_state=33)
#分别检验训练和测试数据的规模
print(y_train.shape)
print(y_test.shape)

'''
使用支持向量机（分类）对手写数字图像进行识别
'''
#从sklearn.precessing里导入数据标准化模块
from sklearn.preprocessing import StandardScaler
#从sklearn.svm里面导入基于线性假设的支持向量分类器LinearSVC
from sklearn.svm import LinearSVC

'''
对训练和测试的特征数据进行标准化
'''
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

'''
支持向量机分类器LinearSVC初始化和模型训练
'''
#初始化线性假设支持向量机分类器LinearSVC
lsvc = LinearSVC()
#进行模型训练
lsvc.fit(X_train,y_train)
#利用训练好的模型对测试样本的数字类别进行预测，预测结果存储在变量y_predict中
y_predict = lsvc.predict(X_test)

'''
使用模型自带的评估函数进行准确性测评
'''
print('The Accuracy of Linear SVC is',lsvc.score(X_test,y_test))
#依然使用sklearn.metrics里面的classification_report模块对预测结果做更详细的分析
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict,target_names=digit.target_names.astype(str)))

```

    (1797, 64)
    (1347,)
    (450,)
    The Accuracy of Linear SVC is 0.953333333333
                 precision    recall  f1-score   support
    
              0       0.92      1.00      0.96        35
              1       0.96      0.98      0.97        54
              2       0.98      1.00      0.99        44
              3       0.93      0.93      0.93        46
              4       0.97      1.00      0.99        35
              5       0.94      0.94      0.94        48
              6       0.96      0.98      0.97        51
              7       0.92      1.00      0.96        35
              8       0.98      0.84      0.91        58
              9       0.95      0.91      0.93        44
    
    avg / total       0.95      0.95      0.95       450
    
    

## 3. Naive Bayes


```python
#从sklearn.datasets里面导入新闻数据抓取器fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups
#与之前预存的数据不同，fetch_20newsgroups需要即时从互联网下载数据
news = fetch_20newsgroups(subset='all')

#查验数据的规模和细节
print(len((news.data)))
print(news.data[1])

'''
20类新闻文本数据分割
'''
#从sklearn.cross_validation导入train_test_split
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25,random_state=33)

'''
用朴素贝叶斯分类器对新闻文本数据进行类别预测
'''
#从sklearn.feature_extraction.text里导入用于文本特征向量转化模块
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

#从sklearn.naive_bayes里导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
#使用默认初始化朴素贝叶斯模型
mnb = MultinomialNB()
#利用训练数据对模型的参数进行估计
mnb.fit(X_train,y_train)
#对测试的样本进行类别预测，结果存储在变量y_predict
y_predict = mnb.predict(X_test)


'''
对朴素贝叶斯分类器在新闻文本数据上的表现性能进行评估
'''
#从sklearn.metrics里面导入classification_report用来详细的分类性能报告
from sklearn.metrics import classification_report
print('The accuracy of naive bayes classifier is',mnb.score(X_test,y_test))
print(classification_report(y_test,y_predict,target_names=news.target_names))

```

    18846
    From: mblawson@midway.ecn.uoknor.edu (Matthew B Lawson)
    Subject: Which high-performance VLB video card?
    Summary: Seek recommendations for VLB video card
    Nntp-Posting-Host: midway.ecn.uoknor.edu
    Organization: Engineering Computer Network, University of Oklahoma, Norman, OK, USA
    Keywords: orchid, stealth, vlb
    Lines: 21
    
      My brother is in the market for a high-performance video card that supports
    VESA local bus with 1-2MB RAM.  Does anyone have suggestions/ideas on:
    
      - Diamond Stealth Pro Local Bus
    
      - Orchid Farenheit 1280
    
      - ATI Graphics Ultra Pro
    
      - Any other high-performance VLB card
    
    
    Please post or email.  Thank you!
    
      - Matt
    
    -- 
        |  Matthew B. Lawson <------------> (mblawson@essex.ecn.uoknor.edu)  |   
      --+-- "Now I, Nebuchadnezzar, praise and exalt and glorify the King  --+-- 
        |   of heaven, because everything he does is right and all his ways  |   
        |   are just." - Nebuchadnezzar, king of Babylon, 562 B.C.           |   
    
    The accuracy of naive bayes classifier is 0.839770797963
                              precision    recall  f1-score   support
    
                 alt.atheism       0.86      0.86      0.86       201
               comp.graphics       0.59      0.86      0.70       250
     comp.os.ms-windows.misc       0.89      0.10      0.17       248
    comp.sys.ibm.pc.hardware       0.60      0.88      0.72       240
       comp.sys.mac.hardware       0.93      0.78      0.85       242
              comp.windows.x       0.82      0.84      0.83       263
                misc.forsale       0.91      0.70      0.79       257
                   rec.autos       0.89      0.89      0.89       238
             rec.motorcycles       0.98      0.92      0.95       276
          rec.sport.baseball       0.98      0.91      0.95       251
            rec.sport.hockey       0.93      0.99      0.96       233
                   sci.crypt       0.86      0.98      0.91       238
             sci.electronics       0.85      0.88      0.86       249
                     sci.med       0.92      0.94      0.93       245
                   sci.space       0.89      0.96      0.92       221
      soc.religion.christian       0.78      0.96      0.86       232
          talk.politics.guns       0.88      0.96      0.92       251
       talk.politics.mideast       0.90      0.98      0.94       231
          talk.politics.misc       0.79      0.89      0.84       188
          talk.religion.misc       0.93      0.44      0.60       158
    
                 avg / total       0.86      0.84      0.82      4712
    
    

## 4. KNN


```python
#读取Iris数据集细节资料
#从sklearn.datasets导入iris数据加载器
from sklearn.datasets import load_iris

#s使用加载器读取数据并且存入变量iris
iris = load_iris()
#查看数据规模
print(iris.data.shape)

#查看数据说明，对于一个机器学习实践者来讲，这是一个好习惯
print(iris.DESCR)

'''
对iris数据集进行分割
'''
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.25,random_state=33)

'''
用KNN对iris数据进行类别预测
'''
#从sklearn.precessing里导入数据标准化模块
from sklearn.preprocessing import StandardScaler
#从sklearn.neighbors里选择导入KNeighborsClassifier,K近领分类器
from sklearn.neighbors import KNeighborsClassifier

#对训练和测试集进行标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#使用K近邻分类器对测试的数据进行类别预测，预测结果存储在y_predict中
knc = KNeighborsClassifier()
knc.fit(X_train,y_train)
y_predict = knc.predict(X_test)

'''
对KNN在iris数据上的预测性能进行评估
'''
#使用模型自带的评估函数进行准确性测评
print('the accuracy of K-Nearest Neighbor Classifier is',knc.score(X_train,y_train))
#依然sklearn.metrics里面的classification_report模块对预测结果进行详细的分析
#依然使用sklearn.metrics里面的classification_report模块对预测的结果做出更加详细分析
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict,target_names=iris.target_names))
```

    (150, 4)
    Iris Plants Database
    
    Notes
    -----
    Data Set Characteristics:
        :Number of Instances: 150 (50 in each of three classes)
        :Number of Attributes: 4 numeric, predictive attributes and the class
        :Attribute Information:
            - sepal length in cm
            - sepal width in cm
            - petal length in cm
            - petal width in cm
            - class:
                    - Iris-Setosa
                    - Iris-Versicolour
                    - Iris-Virginica
        :Summary Statistics:
    
        ============== ==== ==== ======= ===== ====================
                        Min  Max   Mean    SD   Class Correlation
        ============== ==== ==== ======= ===== ====================
        sepal length:   4.3  7.9   5.84   0.83    0.7826
        sepal width:    2.0  4.4   3.05   0.43   -0.4194
        petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
        petal width:    0.1  2.5   1.20  0.76     0.9565  (high!)
        ============== ==== ==== ======= ===== ====================
    
        :Missing Attribute Values: None
        :Class Distribution: 33.3% for each of 3 classes.
        :Creator: R.A. Fisher
        :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
        :Date: July, 1988
    
    This is a copy of UCI ML iris datasets.
    http://archive.ics.uci.edu/ml/datasets/Iris
    
    The famous Iris database, first used by Sir R.A Fisher
    
    This is perhaps the best known database to be found in the
    pattern recognition literature.  Fisher's paper is a classic in the field and
    is referenced frequently to this day.  (See Duda & Hart, for example.)  The
    data set contains 3 classes of 50 instances each, where each class refers to a
    type of iris plant.  One class is linearly separable from the other 2; the
    latter are NOT linearly separable from each other.
    
    References
    ----------
       - Fisher,R.A. "The use of multiple measurements in taxonomic problems"
         Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
         Mathematical Statistics" (John Wiley, NY, 1950).
       - Duda,R.O., & Hart,P.E. (1973) Pattern Classification and Scene Analysis.
         (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
       - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
         Structure and Classification Rule for Recognition in Partially Exposed
         Environments".  IEEE Transactions on Pattern Analysis and Machine
         Intelligence, Vol. PAMI-2, No. 1, 67-71.
       - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
         on Information Theory, May 1972, 431-433.
       - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
         conceptual clustering system finds 3 classes in the data.
       - Many, many more ...
    
    the accuracy of K-Nearest Neighbor Classifier is 0.973214285714
                 precision    recall  f1-score   support
    
         setosa       1.00      1.00      1.00         8
     versicolor       0.73      1.00      0.85        11
      virginica       1.00      0.79      0.88        19
    
    avg / total       0.92      0.89      0.90        38
    
    

## 5. Decision Tree


```python
#泰坦尼克号乘客数据查验
import pandas as pd
#利用pandas的read_csv模块直接从互联网收集泰坦尼克号乘客数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
#观察前几行数据，可以发现数据种类各异，数值型，类别型，甚至还有缺失数据
print(titanic.head())
#使用pandas,数据都转入pandas独有的dataframe格式（二维数据表格），直接使用info,查看数据的统计特性
print(titanic.info())

#相当重要的环节- 特征选择，基于一些背景知识，根据对这场故事的了解，sex,age,pclass这些特征都很有可能决定幸免的关键因素
X = titanic[['pclass','age','sex']]
y = titanic['survived']
#对当前选择的特征进行探查
print(X.info())

'''
借助上面的输出，我们设计如下几个数据处理的任务
(1)age这个数据列，只有633个，需要补充完整-用均值替代
(2)sex与pclass两个数据列的值都是类别型的，需要转化为数值特征，用0/1替代
'''
#首先我们补充age里的数据，使用平均数或者中位数都是对模型偏离造成最小影响的策略
print(X['age'].fillna(X['age'].mean(),inplace=True))
#对补完的数据重新探查
print(X.info())
print(X['age'])
print(X['sex'])
print(X['pclass'])

'''
进行数据分割和特征抽取
'''
#数据分割
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)

#s使用scikit-learn.feature_extraction中特征转化器
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
#转换特征后，我们发现凡是类别型的特征都单独剥离出来，独成一列特征，数值型的则保持不变
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
print(vec.feature_names_)
#同样需要对测试数据的特征进行转换
X_test = vec.transform(X_test.to_dict(orient='record'))
print(X_train)
print('training samples is',X_train.shape)
print('training samples is',X_test.shape)
'''
导入决策树分类器，使用分割到的训练数据进行模型学习
用训练好的决策树模型对测试特征数据进行预测
'''
#从sklearn.tree中导入决策树分类器
from sklearn.tree import DecisionTreeClassifier
#使用默认配置初始化决策树分类器
dtc = DecisionTreeClassifier()
#使用分割的训练数据进行模型学习
dtc.fit(X_train,y_train)
#用训练好的决策树模型对测试特征数据进行预测
y_predict = dtc.predict(X_test)
print(y_predict)
print(y_test)

'''
决策树模型对泰坦尼克号乘客是否生还的预测功能
'''
#从sklearn.metrics导入classification_report
from sklearn.metrics import classification_report
#输出预测准确性
print('the accuracy of Desion Tree is',dtc.score(X_test,y_test))
#输出更加详细的分类性能
print(classification_report(y_predict,y_test,target_names=['died','survived']))

```

       row.names pclass  survived  \
    0          1    1st         1   
    1          2    1st         0   
    2          3    1st         0   
    3          4    1st         0   
    4          5    1st         1   
    
                                                  name      age     embarked  \
    0                     Allen, Miss Elisabeth Walton  29.0000  Southampton   
    1                      Allison, Miss Helen Loraine   2.0000  Southampton   
    2              Allison, Mr Hudson Joshua Creighton  30.0000  Southampton   
    3  Allison, Mrs Hudson J.C. (Bessie Waldo Daniels)  25.0000  Southampton   
    4                    Allison, Master Hudson Trevor   0.9167  Southampton   
    
                             home.dest room      ticket   boat     sex  
    0                     St Louis, MO  B-5  24160 L221      2  female  
    1  Montreal, PQ / Chesterville, ON  C26         NaN    NaN  female  
    2  Montreal, PQ / Chesterville, ON  C26         NaN  (135)    male  
    3  Montreal, PQ / Chesterville, ON  C26         NaN    NaN  female  
    4  Montreal, PQ / Chesterville, ON  C22         NaN     11    male  
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1313 entries, 0 to 1312
    Data columns (total 11 columns):
    row.names    1313 non-null int64
    pclass       1313 non-null object
    survived     1313 non-null int64
    name         1313 non-null object
    age          633 non-null float64
    embarked     821 non-null object
    home.dest    754 non-null object
    room         77 non-null object
    ticket       69 non-null object
    boat         347 non-null object
    sex          1313 non-null object
    dtypes: float64(1), int64(2), object(8)
    memory usage: 112.9+ KB
    None
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1313 entries, 0 to 1312
    Data columns (total 3 columns):
    pclass    1313 non-null object
    age       633 non-null float64
    sex       1313 non-null object
    dtypes: float64(1), object(2)
    memory usage: 30.9+ KB
    None
    

    C:\Users\xxz\Anaconda3\lib\site-packages\pandas\core\generic.py:3191: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self._update_inplace(new_data)
    

    None
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1313 entries, 0 to 1312
    Data columns (total 3 columns):
    pclass    1313 non-null object
    age       1313 non-null float64
    sex       1313 non-null object
    dtypes: float64(1), object(2)
    memory usage: 30.9+ KB
    None
    0       29.000000
    1        2.000000
    2       30.000000
    3       25.000000
    4        0.916700
    5       47.000000
    6       63.000000
    7       39.000000
    8       58.000000
    9       71.000000
    10      47.000000
    11      19.000000
    12      31.194181
    13      31.194181
    14      31.194181
    15      50.000000
    16      24.000000
    17      36.000000
    18      37.000000
    19      47.000000
    20      26.000000
    21      25.000000
    22      25.000000
    23      19.000000
    24      28.000000
    25      45.000000
    26      39.000000
    27      30.000000
    28      58.000000
    29      31.194181
              ...    
    1283    31.194181
    1284    31.194181
    1285    31.194181
    1286    31.194181
    1287    31.194181
    1288    31.194181
    1289    31.194181
    1290    31.194181
    1291    31.194181
    1292    31.194181
    1293    31.194181
    1294    31.194181
    1295    31.194181
    1296    31.194181
    1297    31.194181
    1298    31.194181
    1299    31.194181
    1300    31.194181
    1301    31.194181
    1302    31.194181
    1303    31.194181
    1304    31.194181
    1305    31.194181
    1306    31.194181
    1307    31.194181
    1308    31.194181
    1309    31.194181
    1310    31.194181
    1311    31.194181
    1312    31.194181
    Name: age, dtype: float64
    0       female
    1       female
    2         male
    3       female
    4         male
    5         male
    6       female
    7         male
    8       female
    9         male
    10        male
    11      female
    12      female
    13        male
    14        male
    15      female
    16        male
    17        male
    18        male
    19      female
    20        male
    21        male
    22        male
    23      female
    24        male
    25        male
    26        male
    27      female
    28      female
    29        male
             ...  
    1283    female
    1284      male
    1285      male
    1286      male
    1287      male
    1288      male
    1289      male
    1290      male
    1291      male
    1292      male
    1293    female
    1294      male
    1295      male
    1296      male
    1297      male
    1298      male
    1299      male
    1300      male
    1301      male
    1302      male
    1303      male
    1304    female
    1305      male
    1306    female
    1307    female
    1308      male
    1309      male
    1310      male
    1311    female
    1312      male
    Name: sex, dtype: object
    0       1st
    1       1st
    2       1st
    3       1st
    4       1st
    5       1st
    6       1st
    7       1st
    8       1st
    9       1st
    10      1st
    11      1st
    12      1st
    13      1st
    14      1st
    15      1st
    16      1st
    17      1st
    18      1st
    19      1st
    20      1st
    21      1st
    22      1st
    23      1st
    24      1st
    25      1st
    26      1st
    27      1st
    28      1st
    29      1st
           ... 
    1283    3rd
    1284    3rd
    1285    3rd
    1286    3rd
    1287    3rd
    1288    3rd
    1289    3rd
    1290    3rd
    1291    3rd
    1292    3rd
    1293    3rd
    1294    3rd
    1295    3rd
    1296    3rd
    1297    3rd
    1298    3rd
    1299    3rd
    1300    3rd
    1301    3rd
    1302    3rd
    1303    3rd
    1304    3rd
    1305    3rd
    1306    3rd
    1307    3rd
    1308    3rd
    1309    3rd
    1310    3rd
    1311    3rd
    1312    3rd
    Name: pclass, dtype: object
    ['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', 'sex=male']
    [[ 31.19418104   0.           0.           1.           0.           1.        ]
     [ 31.19418104   1.           0.           0.           1.           0.        ]
     [ 31.19418104   0.           0.           1.           0.           1.        ]
     ..., 
     [ 12.           0.           1.           0.           1.           0.        ]
     [ 18.           0.           1.           0.           0.           1.        ]
     [ 31.19418104   0.           0.           1.           1.           0.        ]]
    training samples is (984, 6)
    training samples is (329, 6)
    [0 1 0 0 0 1 1 0 0 1 0 1 1 0 0 1 0 1 0 1 1 1 0 0 0 0 1 1 0 0 1 0 0 0 1 1 1
     0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 1 1 0 0 1 0 0 1 0 0 0 0 0 0 0 1 0
     0 1 0 0 0 1 1 1 0 0 0 0 0 0 1 0 0 1 0 0 0 1 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0
     1 1 0 0 0 0 0 0 0 0 0 1 1 0 0 1 1 0 0 1 0 0 1 0 0 1 0 1 1 0 0 0 0 0 0 0 1
     1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0
     0 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 1 0 1 1 1 0
     0 0 0 0 0 0 0 0 1 1 0 0 0 1 0 0 0 0 0 1 0 1 1 0 0 0 0 1 1 1 0 0 0 1 1 0 0
     1 1 0 0 0 1 0 0 0 1 0 0 0 1 1 0 0 0 0 1 0 1 0 0 0 1 0 0 1 1 0 0 0 1 0 0 0
     1 0 0 0 1 1 0 0 0 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 1 0]
    386     0
    89      1
    183     1
    746     0
    1211    0
    384     1
    729     1
    1076    0
    1290    0
    385     1
    722     1
    169     1
    275     0
    485     0
    790     0
    159     1
    302     0
    296     1
    1208    0
    428     1
    745     0
    109     1
    1300    0
    906     1
    1291    0
    32      1
    584     1
    585     1
    1250    0
    44      0
           ..
    414     0
    241     1
    31      1
    764     0
    932     0
    1226    1
    657     1
    256     0
    803     0
    677     0
    1279    1
    396     1
    462     1
    693     0
    927     0
    628     1
    506     0
    371     0
    1198    0
    232     1
    709     0
    508     0
    282     1
    698     0
    1049    0
    1048    0
    106     1
    618     0
    175     0
    937     0
    Name: survived, dtype: int64
    the accuracy of Desion Tree is 0.781155015198
                 precision    recall  f1-score   support
    
           died       0.91      0.78      0.84       236
       survived       0.58      0.80      0.67        93
    
    avg / total       0.81      0.78      0.79       329
    
    

## 6. Ensemble Classification


```python
import pandas as pd

'''
通过互联网读取泰坦尼克号乘客文档，并存储在变量titanic中
'''
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

'''
人工选取pclass,age以及sex作为乘客生还的特征
'''
X = titanic[['pclass','age','sex']]
y = titanic['survived']

'''
对于缺失的年龄信息，我们使用全体乘客的平均年龄替代，这样可以在保证顺序训练模型的同时，尽可能不影响预测任务
'''
X['age'].fillna(X['age'].mean(),inplace=True)

'''
对原始数据进行分割，25%的乘客数据用于测试
'''
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)

'''
对类别型特征进行转化，成为特征向量
'''
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

'''
使用单一决策树进行模型训练以及预测分析
'''
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc_y_pred = dtc.predict(X_test)

'''
使用随机森林分类器进行集成模型的训练以及预测分析
'''
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_pred = rfc.predict(X_test)

'''
使用梯度提升决策树进行集成模型训练和预测
'''
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_pred = gbc.predict(X_test)

'''
集成模型对泰坦尼克号乘客是否生还的预测性能
'''
#从sklearn.metrics导入classification_report
from sklearn.metrics import classification_report

#输出单一决策在测试集上的准确率，以及详细的精确率，召回率，F1指标
print('the accuracy of decision tree is',dtc.score(X_test,y_test))
print(classification_report(dtc_y_pred,y_test,target_names=['died','survived']))

#输出随机森林分类器在测试集上的准确率，以及详细的精确率，召回率，F1指标
print('the accuracy of random forest classifier is',rfc.score(X_test,y_test))
print(classification_report(rfc_y_pred,y_test,target_names=['died','survived']))

#输出梯度提升决策树在测试集上的准确率，以及详细的精确率，召回率，F1指标
print('the accuracy of decision tree is',gbc.score(X_test,y_test))
print(classification_report(gbc_y_pred,y_test,target_names=['died','survived']))


```

    C:\Users\xxz\Anaconda3\lib\site-packages\pandas\core\generic.py:3191: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self._update_inplace(new_data)
    

    the accuracy of decision tree is 0.781155015198
                 precision    recall  f1-score   support
    
           died       0.91      0.78      0.84       236
       survived       0.58      0.80      0.67        93
    
    avg / total       0.81      0.78      0.79       329
    
    the accuracy of random forest classifier is 0.775075987842
                 precision    recall  f1-score   support
    
           died       0.90      0.77      0.83       236
       survived       0.57      0.78      0.66        93
    
    avg / total       0.81      0.78      0.78       329
    
    the accuracy of decision tree is 0.790273556231
                 precision    recall  f1-score   support
    
           died       0.92      0.78      0.84       239
       survived       0.58      0.82      0.68        90
    
    avg / total       0.83      0.79      0.80       329
    
    
