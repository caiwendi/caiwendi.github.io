---
layout:     post                    # 使用的布局（不需要改）
title:      【转】Skills of debugging model           # 标题 
subtitle:   特征提升，模型正则化，模型检验，超参数搜索 #副标题
date:       2017-09-18              # 时间
author:     Brian                      # 作者
header-img: img/post-bg-universe.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Kaggle Competition
---

# Skills of debugging model

Author: Xie Zhong-zhao

## 1. 特征提升

###  1.1 Feature Extraction


```python
from sklearn.feature_extraction import  DictVectorizer
'''
举个栗子，使用DictVectorizer对特征进行抽取和向量化
'''
measurements = [{'city':'Dubai','temperature':33.},{'city':'London','temperature':12.},{'city':'San Fransisco','temperature':18.}]

#初始化DictVectorizer特征抽取器
vec = DictVectorizer()
#输出转化之后的矩阵
print(vec.fit_transform(measurements).toarray())
#输出各个维度的特征含义
print(vec.get_feature_names())

'''
**************************************************************************
**************************************************************************
'''

'''
特征数值常见的计算方式有两种，分别是：CountVectorizer和TfidfVectorizer
(1)对每条训练文本，CountVectorizer只考虑每种词汇在该条训练文本中出现的频率
(2)TfidfVectorizer除了考量某个词汇在当前文本中出现的频率之外，同时关注包含这个词汇的文本条数的倒数。
'''

'''
使用CountVectorizer并且不去掉停用词的情况下，对文本特征进行量化的朴素贝叶斯分类性能测试
'''
#从sklearn.datasets里导入20类新闻文本数据抓取器
from sklearn.datasets import fetch_20newsgroups

#从互联网上及时下载新闻样本，subset='all'参数代表下载全部近2万条文本存储在变量news中
news = fetch_20newsgroups(subset='all')

#从sklearn.cross_validation导入train_test_split模块用于分割数据
from sklearn.cross_validation import train_test_split
#对news中数据data进行分割数据，25%的文本用作测试集；75%作为训练集
X_train,X_test,y_train,y_test = train_test_split(news.data, news.target,test_size=0.25,random_state=33)

#从sklearn.feature_extraction.text里导入CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#采用默认设置对CountVectorizer进行初始化(默认配置不去除英文停用词)，并且赋值给变量count_vec
count_vec = CountVectorizer()

#只使用词频统计方式将原始训练和测试文本转化为特征向量
X_count_train = count_vec.fit_transform(X_train)
X_count_test = count_vec.transform(X_test)

#从sklearn.naive_bayes里导入朴素贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB
#使用默认配置对分类器进行初始化
mnb_count = MultinomialNB()
#使用朴素贝叶斯分类器，对CountVectorizer(不去停用词)训练样本进行参数学习
mnb_count.fit(X_count_train,y_train)

#输出模型准确性结果
print('the accuracy of classifying 20newgroups using Naive Bayes(CountVectorizer without filtering stopwords):',
      mnb_count.score(X_count_test,y_test))
#将分类预测结果存储在变量y_count_predict中
y_count_predict = mnb_count.predict(X_count_test)
#输出更加详细的其他评价的分类性能指标
from sklearn.metrics import classification_report
print(classification_report(y_test,y_count_predict,target_names=news.target_names))


'''
**************************************************************************
**************************************************************************
'''

'''
使用TfidfVectorizer并且不去掉停用词的情况下，对文本特征进行量化的朴素贝叶斯分类性能测试
'''
#从sklearn.feature_extraction.text里面分别导入TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#采用默认配置对TfidfVectorizer进行初始化，并且赋值给变量tfidf_vec
tfidf_vec = TfidfVectorizer()

#使用tfidf的方式，将原始训练和测试的文本转化为特征向量
X_tfidf_train = tfidf_vec.fit_transform(X_train)
X_tfidf_test = tfidf_vec.transform(X_test)

#依然使用默认配置的朴素贝叶斯分类器，在相同的训练和测试数据上，对新的特征量化方式进行性能评估
mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(X_tfidf_train,y_train)
#输出模型准确性结果
print('the accuracy of classifying 20newgroups using Naive Bayes(TfidfVectorizer without filtering stopwords):',
      mnb_tfidf.score(X_tfidf_test,y_test))
y_tfidf_predict = mnb_tfidf.predict(X_tfidf_test)
#输出更加详细的其他评价的分类性能指标
print(classification_report(y_test,y_tfidf_predict,target_names=news.target_names))

'''
**************************************************************************
**************************************************************************
'''

'''
分别使用CountVectorizer与TfidfVectorizer,并且去掉停用词的条件下，对文本特征进行量化的朴素贝叶斯分类性能测试
'''
count_filter_vec,tfidf_filter_vec = CountVectorizer(analyzer='word',stop_words='english'),TfidfVectorizer(analyzer='word',stop_words='english')

#使用带有停用词过滤的CountVectorizer对训练和测试文本分别进行量化处理
X_count_filter_train = count_filter_vec.fit_transform(X_train)
X_count_filter_test = count_filter_vec.transform(X_test)

#使用带有停用词过滤的TfidfVectorizer对训练和测试样本分别进行量化处理
X_tfidf_filter_train = tfidf_filter_vec.fit_transform(X_train)
X_tfidf_filter_test = tfidf_filter_vec.transform(X_test)

#初始化默认配置朴素贝叶斯分类器，并对CountVectorizer后的数据进行预测与准确的评估
mnb_count_filter = MultinomialNB()
mnb_count_filter.fit(X_count_filter_train,y_train)
print('the accuracy of classifying 20newgroups using Naive Bayes(CountVectorizer by filtering stopwords):',
      mnb_count_filter.score(X_count_filter_test,y_test))
y_count_filter_predict = mnb_count_filter.predict(X_count_filter_test)
print(classification_report(y_test,y_count_filter_predict,target_names=news.target_names))

#初始化默认配置朴素贝叶斯分类器，并对TfidfVectorizer后的数据进行预测与准确的评估
mnb_tfidf_filter = MultinomialNB()
mnb_tfidf_filter.fit(X_tfidf_filter_train,y_train)
print('the accuracy of classifying 20newgroups using Naive Bayes(tfidfVectorizer by filtering stopwords):',
      mnb_tfidf_filter.score(X_tfidf_filter_test,y_test))
y_tfidf_filter_predict = mnb_tfidf_filter.predict(X_tfidf_filter_test)
print(classification_report(y_test,y_tfidf_filter_predict,target_names=news.target_names))
```

    [[  1.   0.   0.  33.]
     [  0.   1.   0.  12.]
     [  0.   0.   1.  18.]]
    ['city=Dubai', 'city=London', 'city=San Fransisco', 'temperature']
    the accuracy of classifying 20newgroups using Naive Bayes(CountVectorizer without filtering stopwords): 0.839770797963
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
    
    the accuracy of classifying 20newgroups using Naive Bayes(TfidfVectorizer without filtering stopwords): 0.846349745331
                              precision    recall  f1-score   support
    
                 alt.atheism       0.84      0.67      0.75       201
               comp.graphics       0.85      0.74      0.79       250
     comp.os.ms-windows.misc       0.82      0.85      0.83       248
    comp.sys.ibm.pc.hardware       0.76      0.88      0.82       240
       comp.sys.mac.hardware       0.94      0.84      0.89       242
              comp.windows.x       0.96      0.84      0.89       263
                misc.forsale       0.93      0.69      0.79       257
                   rec.autos       0.84      0.92      0.88       238
             rec.motorcycles       0.98      0.92      0.95       276
          rec.sport.baseball       0.96      0.91      0.94       251
            rec.sport.hockey       0.88      0.99      0.93       233
                   sci.crypt       0.73      0.98      0.83       238
             sci.electronics       0.91      0.83      0.87       249
                     sci.med       0.97      0.92      0.95       245
                   sci.space       0.89      0.96      0.93       221
      soc.religion.christian       0.51      0.97      0.67       232
          talk.politics.guns       0.83      0.96      0.89       251
       talk.politics.mideast       0.92      0.97      0.95       231
          talk.politics.misc       0.98      0.62      0.76       188
          talk.religion.misc       0.93      0.16      0.28       158
    
                 avg / total       0.87      0.85      0.84      4712
    
    the accuracy of classifying 20newgroups using Naive Bayes(CountVectorizer by filtering stopwords): 0.863752122241
                              precision    recall  f1-score   support
    
                 alt.atheism       0.85      0.89      0.87       201
               comp.graphics       0.62      0.88      0.73       250
     comp.os.ms-windows.misc       0.93      0.22      0.36       248
    comp.sys.ibm.pc.hardware       0.62      0.88      0.73       240
       comp.sys.mac.hardware       0.93      0.85      0.89       242
              comp.windows.x       0.82      0.85      0.84       263
                misc.forsale       0.90      0.79      0.84       257
                   rec.autos       0.91      0.91      0.91       238
             rec.motorcycles       0.98      0.94      0.96       276
          rec.sport.baseball       0.98      0.92      0.95       251
            rec.sport.hockey       0.92      0.99      0.95       233
                   sci.crypt       0.91      0.97      0.93       238
             sci.electronics       0.87      0.89      0.88       249
                     sci.med       0.94      0.95      0.95       245
                   sci.space       0.91      0.96      0.93       221
      soc.religion.christian       0.87      0.94      0.90       232
          talk.politics.guns       0.89      0.96      0.93       251
       talk.politics.mideast       0.95      0.98      0.97       231
          talk.politics.misc       0.84      0.90      0.87       188
          talk.religion.misc       0.91      0.53      0.67       158
    
                 avg / total       0.88      0.86      0.85      4712
    
    the accuracy of classifying 20newgroups using Naive Bayes(tfidfVectorizer by filtering stopwords): 0.882640067912
                              precision    recall  f1-score   support
    
                 alt.atheism       0.86      0.81      0.83       201
               comp.graphics       0.85      0.81      0.83       250
     comp.os.ms-windows.misc       0.84      0.87      0.86       248
    comp.sys.ibm.pc.hardware       0.78      0.88      0.83       240
       comp.sys.mac.hardware       0.92      0.90      0.91       242
              comp.windows.x       0.95      0.88      0.91       263
                misc.forsale       0.90      0.80      0.85       257
                   rec.autos       0.89      0.92      0.90       238
             rec.motorcycles       0.98      0.94      0.96       276
          rec.sport.baseball       0.97      0.93      0.95       251
            rec.sport.hockey       0.88      0.99      0.93       233
                   sci.crypt       0.85      0.98      0.91       238
             sci.electronics       0.93      0.86      0.89       249
                     sci.med       0.96      0.93      0.95       245
                   sci.space       0.90      0.97      0.93       221
      soc.religion.christian       0.70      0.96      0.81       232
          talk.politics.guns       0.84      0.98      0.90       251
       talk.politics.mideast       0.92      0.99      0.95       231
          talk.politics.misc       0.97      0.74      0.84       188
          talk.religion.misc       0.96      0.29      0.45       158
    
                 avg / total       0.89      0.88      0.88      4712

​    

### 1.2 Feature Selection 

```python
import pandas as pd

'''
使用Titanic数据集，通过特征筛选的方式一步步提升决策树的预测性能
'''
#从互联网读取titanic数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
#分离数据特征与预测目标
y = titanic['survived']
X = titanic.drop(['row.names','name','survived'],axis=1)

#对缺失的数据进行填充
X['age'].fillna(X['age'].mean(),inplace=True)
X.fillna('UNKNOW',inplace=True)
print(X.shape)
print(y.shape)

#分割数据，依然采用25%用于测试
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)

print(X_train)
print(X_train.shape)
#类别型特征向量化
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient ='record'))
print(X_train)
print(X_train.shape)
print(len(vec.feature_names_))
# print(X_test)
# #输出处理后特征向量的维度
# print(len(vec.feature_names_))
#
#使用决策树模型依靠所有特征进行预测，并作性能评估
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train,y_train)
print('the accuracy of Decision Tree',dt.score(X_test,y_test))
#
#从sklearn导入特征筛选器
from sklearn import feature_selection
#筛选前20%的特征，使用相同的配置的决策树模型进行预测，并且评估性能
fs = feature_selection.SelectPercentile(feature_selection.chi2,percentile=20)
X_train_fs = fs.fit_transform(X_train,y_train)
dt.fit(X_train_fs,y_train)
X_test_fs = fs.transform(X_test)
print('the accuracy of selecting 20% features in model',dt.score(X_test_fs,y_test))
#
#通过交叉验证的方法，按照固定的百分比筛选特征，并作图展示性能随特征筛选比例的变化
from sklearn.cross_validation import cross_val_score
import numpy as np

percentiles = range(1,100,2)
results = []
for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2,percentile=i)
    X_train_fs = fs.fit_transform(X_train,y_train)
    scores = cross_val_score(dt,X_train_fs,y_train,cv=5)
    results = np.append(results,scores.mean()) #列表拼接
print(results)
#找到提现最佳性能的特征筛选的百分比
# opt = np.where(results == results.max())[0]
# print('Optimal number of features %d'%percentiles[opt])
#
import pylab as pl
fig = pl.figure()
pl.plot(percentiles,results)
pl.xlabel('percentiles of features')
pl.ylabel('accuracy')
fig.savefig("feature selecting.png", dpi=300)  #save graph
pl.show()

#使用最佳筛选后的特征，利用相同配置在测试集上进行性能和评估
from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.chi2,percentile=7)
X_train_fs = fs.fit_transform(X_train,y_train)
dt.fit(X_train_fs,y_train)
X_test_fs = fs.transform(X_test)
print(dt.score(X_test_fs,y_test))
```

    (1313, 8)
    (1313,)
         pclass        age     embarked                          home.dest  \
    1086    3rd  31.194181       UNKNOW                             UNKNOW   
    12      1st  31.194181    Cherbourg                      Paris, France   
    1036    3rd  31.194181       UNKNOW                             UNKNOW   
    833     3rd  32.000000  Southampton      Foresvik, Norway Portland, ND   
    1108    3rd  31.194181       UNKNOW                             UNKNOW   
    562     2nd  41.000000    Cherbourg                       New York, NY   
    437     2nd  48.000000  Southampton       Somerset / Bernardsville, NJ   
    663     3rd  26.000000  Southampton                             UNKNOW   
    669     3rd  19.000000  Southampton                            England   
    507     2nd  31.194181  Southampton                   Petworth, Sussex   
    1167    3rd  31.194181       UNKNOW                             UNKNOW   
    821     3rd   9.000000  Southampton  Strood, Kent, England Detroit, MI   
    327     2nd  32.000000  Southampton                   Warwick, England   
    715     3rd  21.000000  Southampton               Ireland New York, NY   
    308     1st  31.194181       UNKNOW                             UNKNOW   
    1274    3rd  31.194181       UNKNOW                             UNKNOW   
    640     3rd  40.000000  Southampton              Sweden  Worcester, MA   
    72      1st  70.000000  Southampton                      Milwaukee, WI   
    1268    3rd  31.194181       UNKNOW                             UNKNOW   
    1024    3rd  31.194181       UNKNOW                             UNKNOW   
    1047    3rd  31.194181       UNKNOW                             UNKNOW   
    940     3rd  31.194181       UNKNOW                             UNKNOW   
    350     2nd  20.000000  Southampton       Skara, Sweden / Rockford, IL   
    892     3rd  31.194181       UNKNOW                             UNKNOW   
    555     2nd  30.000000  Southampton           Finland / Washington, DC   
    176     1st  36.000000  Southampton                   Philadelphia, PA   
    107     1st  31.194181    Cherbourg                       New York, NY   
    475     2nd  34.000000  Southampton                        Chicago, IL   
    330     2nd  23.000000  Southampton                           Guernsey   
    533     2nd  34.000000  Southampton             Denmark / New York, NY   
    ...     ...        ...          ...                                ...   
    235     1st  24.000000  Southampton                     Huntington, WV   
    465     2nd  22.000000  Southampton             India / Pittsburgh, PA   
    210     1st  31.194181  Southampton                       New York, NY   
    579     2nd  40.000000  Southampton            Aberdeen / Portland, OR   
    650     3rd  23.000000  Southampton                             UNKNOW   
    1031    3rd  31.194181       UNKNOW                             UNKNOW   
    99      1st  24.000000  Southampton                       Winnipeg, MB   
    969     3rd  31.194181       UNKNOW                             UNKNOW   
    535     2nd  31.194181    Cherbourg                              Paris   
    403     2nd  31.194181  Southampton         England / Philadelphia, PA   
    744     3rd  45.000000  Southampton               Australia Fingal, ND   
    344     2nd  26.000000  Southampton            Elmira, NY / Orange, NJ   
    84      1st  31.194181  Southampton                  San Francisco, CA   
    528     2nd  20.000000  Southampton    Gunnislake, England / Butte, MT   
    1270    3rd  31.194181       UNKNOW                             UNKNOW   
    662     3rd  40.000000    Cherbourg                             UNKNOW   
    395     2nd  42.000000  Southampton                      Greenport, NY   
    1196    3rd  31.194181       UNKNOW                             UNKNOW   
    543     2nd  23.000000    Cherbourg               Paris / Montreal, PQ   
    845     3rd  31.194181       UNKNOW                             UNKNOW   
    813     3rd  25.000000   Queenstown                       New York, NY   
    61      1st  31.194181  Southampton   St Leonards-on-Sea, England Ohio   
    102     1st  23.000000  Southampton                       Winnipeg, MB   
    195     1st  28.000000       UNKNOW                      ?Havana, Cuba   
    57      1st  27.000000  Southampton          New York, NY / Ithaca, NY   
    1225    3rd  31.194181       UNKNOW                             UNKNOW   
    658     3rd  31.194181    Cherbourg                 Syria New York, NY   
    578     2nd  12.000000  Southampton            Aberdeen / Portland, OR   
    391     2nd  18.000000  Southampton                New Forest, England   
    1044    3rd  31.194181       UNKNOW                             UNKNOW   
    
            room         ticket    boat     sex  
    1086  UNKNOW         UNKNOW  UNKNOW    male  
    12      B-35   17477 L69 6s       9  female  
    1036  UNKNOW         UNKNOW  UNKNOW    male  
    833   UNKNOW         UNKNOW  UNKNOW    male  
    1108  UNKNOW         UNKNOW  UNKNOW    male  
    562   UNKNOW         UNKNOW  UNKNOW    male  
    437   UNKNOW         UNKNOW       9  female  
    663   UNKNOW         UNKNOW  UNKNOW    male  
    669   UNKNOW         UNKNOW  UNKNOW    male  
    507   UNKNOW         UNKNOW  UNKNOW    male  
    1167  UNKNOW         UNKNOW  UNKNOW    male  
    821   UNKNOW         UNKNOW       C    male  
    327   UNKNOW         UNKNOW  UNKNOW  female  
    715   UNKNOW         UNKNOW  UNKNOW    male  
    308   UNKNOW         UNKNOW  UNKNOW  female  
    1274  UNKNOW         UNKNOW  UNKNOW    male  
    640   UNKNOW         UNKNOW   (142)    male  
    72    UNKNOW         UNKNOW   (269)    male  
    1268  UNKNOW         UNKNOW  UNKNOW    male  
    1024  UNKNOW         UNKNOW  UNKNOW    male  
    1047  UNKNOW         UNKNOW  UNKNOW  female  
    940   UNKNOW         UNKNOW  UNKNOW    male  
    350   UNKNOW         UNKNOW      12  female  
    892   UNKNOW         UNKNOW  UNKNOW    male  
    555   UNKNOW         UNKNOW  UNKNOW  female  
    176   UNKNOW         UNKNOW       7    male  
    107   UNKNOW         UNKNOW       5  female  
    475     F-33         UNKNOW    14/D  female  
    330   UNKNOW         UNKNOW  UNKNOW    male  
    533   UNKNOW         250647  UNKNOW    male  
    ...      ...            ...     ...     ...  
    235   UNKNOW         UNKNOW  UNKNOW    male  
    465   UNKNOW         UNKNOW  UNKNOW  female  
    210   UNKNOW         UNKNOW      15    male  
    579   UNKNOW         UNKNOW       9  female  
    650   UNKNOW         UNKNOW  UNKNOW    male  
    1031  UNKNOW         UNKNOW  UNKNOW    male  
    99    UNKNOW         UNKNOW      10  female  
    969   UNKNOW         UNKNOW  UNKNOW    male  
    535   UNKNOW         UNKNOW  UNKNOW    male  
    403   UNKNOW         UNKNOW   (286)    male  
    744   UNKNOW         UNKNOW      15    male  
    344   UNKNOW         UNKNOW  UNKNOW    male  
    84    UNKNOW         UNKNOW      13    male  
    528   UNKNOW         UNKNOW  UNKNOW    male  
    1270  UNKNOW         UNKNOW  UNKNOW    male  
    662   UNKNOW         UNKNOW  UNKNOW    male  
    395   UNKNOW  28220 L32 10s  UNKNOW    male  
    1196  UNKNOW         UNKNOW  UNKNOW    male  
    543   UNKNOW         UNKNOW  UNKNOW    male  
    845   UNKNOW         UNKNOW  UNKNOW    male  
    813   UNKNOW         UNKNOW  UNKNOW    male  
    61    UNKNOW         UNKNOW       6  female  
    102   UNKNOW         UNKNOW      10  female  
    195   UNKNOW         UNKNOW   (189)    male  
    57    UNKNOW         UNKNOW       5    male  
    1225  UNKNOW         UNKNOW  UNKNOW    male  
    658   UNKNOW         UNKNOW  UNKNOW  female  
    578   UNKNOW         UNKNOW       9  female  
    391   UNKNOW         UNKNOW  UNKNOW    male  
    1044  UNKNOW         UNKNOW  UNKNOW  female  
    
    [984 rows x 8 columns]
    (984, 8)
      (0, 0)	31.1941810427
      (0, 78)	1.0
      (0, 82)	1.0
      (0, 366)	1.0
      (0, 391)	1.0
      (0, 435)	1.0
      (0, 437)	1.0
      (0, 473)	1.0
      (1, 0)	31.1941810427
      (1, 73)	1.0
      (1, 79)	1.0
      (1, 296)	1.0
      (1, 389)	1.0
      (1, 397)	1.0
      (1, 436)	1.0
      (1, 446)	1.0
      (2, 0)	31.1941810427
      (2, 78)	1.0
      (2, 82)	1.0
      (2, 366)	1.0
      (2, 391)	1.0
      (2, 435)	1.0
      (2, 437)	1.0
      (2, 473)	1.0
      (3, 0)	32.0
      :	:
      (980, 473)	1.0
      (981, 0)	12.0
      (981, 73)	1.0
      (981, 81)	1.0
      (981, 84)	1.0
      (981, 390)	1.0
      (981, 435)	1.0
      (981, 436)	1.0      (981, 473)	1.0
      (982, 0)	18.0
      (982, 78)	1.0
      (982, 81)	1.0
      (982, 277)	1.0
      (982, 390)	1.0
      (982, 435)	1.0
      (982, 437)	1.0
      (982, 473)	1.0
      (983, 0)	31.1941810427
      (983, 78)	1.0
      (983, 82)	1.0
      (983, 366)	1.0
      (983, 391)	1.0
      (983, 435)	1.0
      (983, 436)	1.0
      (983, 473)	1.0
    (984, 474)
    474
    the accuracy of Decision Tree 0.80547112462
    the accuracy of selecting 20% features in model 0.826747720365
    [ 0.85063904  0.85673057  0.87602556  0.88622964  0.86082251  0.87302618
      0.87100598  0.87302618  0.86693465  0.86894455  0.86895485  0.86491445
      0.86387343  0.86793445  0.86082251  0.86486291  0.86283241  0.8628221
      0.86283241  0.86083282  0.86690373  0.86997526  0.86693465  0.86793445
      0.86792414  0.86892393  0.87200577  0.87097506  0.86691404  0.86284271
      0.86794475  0.86997526  0.86998557  0.86996496  0.87098536  0.86487322
      0.86692435  0.86794475  0.86895485  0.86588332  0.86588332  0.86791383
      0.86692435  0.85776129  0.86692435  0.8618223   0.86286333  0.86082251
      0.85982272  0.86386312]


![](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/Skills%20of%20debugging%20model/output_6_1.png)


    0.857142857143
​    

## 2. 模型正则化

### 2.1 Underfitting and Overfitting


```python
#使用线性回归模型在比萨训练样本上进行拟合
from sklearn.linear_model import LinearRegression
'''
使用线性回归模型在比萨训练样本上进行拟合
'''

#输入训练样本的特征及其目标值，分别存储在变量X_train与y_train
X_train = [[6],[8],[10],[14],[18]]
y_train = [[7],[9],[13],[17.5],[18]]

#使用默认配置初始化线性回归模型
regressor = LinearRegression()
#直接以比萨的半径作为特征训练模型
regressor.fit(X_train,y_train)

import numpy as np
#在x轴上从0到25均匀采样100个数据点
xx = np.linspace(0,26,100)
# print(xx.shape[0])
xx = xx.reshape(xx.shape[0],1)
print(xx)
#以上述100个数据点作为基准，预测回归直线
yy = regressor.predict(xx)

#对回归预测到直线进行作图
import matplotlib.pyplot as plt
plt.scatter(X_train,y_train) #画出样本散点图

plt1,=plt.plot(xx,yy,label="Degree=1") #画出回归预测直线

plt.axis([0,25,0,25]) #设置坐标轴的范围
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of pizza')
plt.legend(handles=[plt1])
plt.show()

'''
输出在线性回归模型在训练样本上的R-squared值
'''
print('the R-squared value of Linear Regressor performing on the training data is:', regressor.score(X_train,y_train))

'''
*****************************************************************************************************
*****************************************************************************************************
*****************************************************************************************************
'''

'''
使用2次多项式回归模型在比萨训练样本上进行拟合
'''
#从sklearn.preprocessing中导入多项式特征产生器
from sklearn.preprocessing import PolynomialFeatures
#使用PolynomialFeatures(degree=2)映射出2次多项式特征，存储变量X_train_poly2中
poly2 = PolynomialFeatures(degree=2)
X_train_poly2 = poly2.fit_transform(X_train)

#以线性回归器为基础，初始化回归模型，尽管特征的维度有所提升，但是模型基础仍然是线性模型
regressor_poly2 = LinearRegression()

#对2次多项式回归模型进行训练
regressor_poly2.fit(X_train_poly2,y_train)

#从新映射绘图用x轴采样数据
xx_poly2 = poly2.transform(xx)

#使用2次多项式回归模型对应x轴采样数据进行回归预测
yy_poly2 = regressor_poly2.predict(xx_poly2)

'''
分别对训练数据点、线性回归直线、2次多项式回归曲线进行作图
'''
plt.scatter(X_train,y_train)

plt1,=plt.plot(xx,yy,label='Degree=1')
plt2,=plt.plot(xx,yy_poly2,label='Degree=2')

plt.axis([0,25,0,25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1,plt2])
plt.show()

'''
输出2次多项式回归模型在训练样本上的R-squared值
'''
print('the R-squared value of Polynomial Regressor(Degree=2) performing on training data is:',
      regressor_poly2.score(X_train_poly2,y_train))

'''
*****************************************************************************************************
*****************************************************************************************************
*****************************************************************************************************
'''

'''
使用4次多项式回归模型在披萨训练样本上进行拟合
'''
from sklearn.preprocessing import PolynomialFeatures
#使用PolynomialFratures将x特征映射到四次多项式，存储在X_train_poly4
poly4 = PolynomialFeatures(degree=4)
X_train_poly4 = poly4.fit_transform(X_train)

#以线性回归器为基础，初始化回归模型，尽管特征的维度有所提升，但是模型基础仍然是线性模型
regressor_poly4 = LinearRegression()

#对4次多项式回归模型进行训练
regressor_poly4.fit(X_train_poly4,y_train)

#从新映射绘图用x轴采样数据
xx_poly4 = poly4.transform(xx)

#使用4次多项式回归模型对应x轴采样数据进行回归预测
yy_poly4 = regressor_poly4.predict(xx_poly4)

'''
分别对训练数据点、线性回归直线、4次多项式回归曲线进行作图
'''
plt.scatter(X_train,y_train)

plt1,=plt.plot(xx,yy,label='Degree=1')
plt2,=plt.plot(xx,yy_poly2,label='Degree=2')
plt4,=plt.plot(xx,yy_poly4,label='Degree=4')

plt.axis([0,25,0,25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1,plt2,plt4])
plt.show()

'''
输出4次多项式回归模型在训练样本上的R-squared值
'''
print('the R-squared value of Polynomial Regressor(Degree=4) performing on training data is:',
      regressor_poly4.score(X_train_poly4,y_train))
```

    [[  0.        ]
     [  0.26262626]
     [  0.52525253]
     [  0.78787879]
     [  1.05050505]
     [  1.31313131]
     [  1.57575758]
     [  1.83838384]
     [  2.1010101 ]
     [  2.36363636]
     [  2.62626263]
     [  2.88888889]
     [  3.15151515]
     [  3.41414141]
     [  3.67676768]
     [  3.93939394]
     [  4.2020202 ]
     [  4.46464646]
     [  4.72727273]
     [  4.98989899]
     [  5.25252525]
     [  5.51515152]
     [  5.77777778]
     [  6.04040404]
     [  6.3030303 ]
     [  6.56565657]
     [  6.82828283]
     [  7.09090909]
     [  7.35353535]
     [  7.61616162]
     [  7.87878788]
     [  8.14141414]
     [  8.4040404 ]
     [  8.66666667]
     [  8.92929293]
     [  9.19191919]
     [  9.45454545]
     [  9.71717172]
     [  9.97979798]
     [ 10.24242424]
     [ 10.50505051]
     [ 10.76767677]
     [ 11.03030303]
     [ 11.29292929]
     [ 11.55555556]
     [ 11.81818182]
     [ 12.08080808]
     [ 12.34343434]
     [ 12.60606061]
     [ 12.86868687]
     [ 13.13131313]
     [ 13.39393939]
     [ 13.65656566]
     [ 13.91919192]
     [ 14.18181818]
     [ 14.44444444]
     [ 14.70707071]
     [ 14.96969697]
     [ 15.23232323]
     [ 15.49494949]
     [ 15.75757576]
     [ 16.02020202]
     [ 16.28282828]
     [ 16.54545455]
     [ 16.80808081]
     [ 17.07070707]
     [ 17.33333333]
     [ 17.5959596 ]
     [ 17.85858586]
     [ 18.12121212]
     [ 18.38383838]
     [ 18.64646465]
     [ 18.90909091]     [ 19.17171717]
     [ 19.43434343]
     [ 19.6969697 ]
     [ 19.95959596]
     [ 20.22222222]
     [ 20.48484848]
     [ 20.74747475]
     [ 21.01010101]
     [ 21.27272727]
     [ 21.53535354]
     [ 21.7979798 ]
     [ 22.06060606]
     [ 22.32323232]
     [ 22.58585859]
     [ 22.84848485]
     [ 23.11111111]
     [ 23.37373737]
     [ 23.63636364]
     [ 23.8989899 ]
     [ 24.16161616]
     [ 24.42424242]
     [ 24.68686869]
     [ 24.94949495]
     [ 25.21212121]
     [ 25.47474747]
     [ 25.73737374]
     [ 26.        ]]


![](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/Skills%20of%20debugging%20model/output_9_1.png)


    the R-squared value of Linear Regressor performing on the training data is: 0.910001596424
​    


![](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/Skills%20of%20debugging%20model/output_9_3.png)


    the R-squared value of Polynomial Regressor(Degree=2) performing on training data is: 0.98164216396
​    


![](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/Skills%20of%20debugging%20model/output_9_5.png)


    the R-squared value of Polynomial Regressor(Degree=4) performing on training data is: 1.0
​    

### 2.2 L1 Regularization


```python
from sklearn.linear_model import LinearRegression
'''
使用线性回归模型在比萨训练样本上进行拟合
'''

#输入训练样本的特征及其目标值，分别存储在变量X_train与y_train
X_train = [[6],[8],[10],[14],[18]]
y_train = [[7],[9],[13],[17.5],[18]]

X_test = [[6],[8],[11],[16]]
y_test = [[8],[12],[15],[18]]

#使用默认配置初始化线性回归模型
regressor = LinearRegression()
#直接以比萨的半径作为特征训练模型
regressor.fit(X_train,y_train)

import numpy as np
#在x轴上从0到25均匀采样100个数据点
xx = np.linspace(0,26,100)
# print(xx.shape[0])
xx = xx.reshape(xx.shape[0],1)
print(xx)
#以上述100个数据点作为基准，预测回归直线
yy = regressor.predict(xx)

#对回归预测到直线进行作图
import matplotlib.pyplot as plt
plt.scatter(X_train,y_train) #画出样本散点图

plt1,=plt.plot(xx,yy,label="Degree=1") #画出回归预测直线

plt.axis([0,25,0,25]) #设置坐标轴的范围
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of pizza')
plt.legend(handles=[plt1])
plt.show()

'''
Lasso模型在4次多项式特征上拟合的表现,等同于在线性回归目标加入L1范数正则化，避免参数过拟合
'''
'''
使用4次多项式回归模型在披萨训练样本上进行拟合
'''

from sklearn.preprocessing import PolynomialFeatures
#使用PolynomialFratures将x特征映射到四次多项式，存储在X_train_poly4
poly4 = PolynomialFeatures(degree=4)
X_train_poly4 = poly4.fit_transform(X_train)
#以线性回归器为基础，初始化回归模型，尽管特征的维度有所提升，但是模型基础仍然是线性模型
regressor_poly4 = LinearRegression()
#对4次多项式回归模型进行训练
regressor_poly4.fit(X_train_poly4,y_train)

X_test_poly4 = poly4.transform(X_test)
print('the accuracy is:',regressor_poly4.score(X_test_poly4,y_test))


#从sklearn.linear_model中导入Lasso
from sklearn.linear_model import Lasso
#从使用配置初始化Lasso
lasso_poly4 = Lasso()
#从使用Lasso对4次多项式特征进行拟合
lasso_poly4.fit(X_train_poly4,y_train)
X_test_poly4_lasso = poly4.transform(X_test)
print('the accuracy is:',lasso_poly4.score(X_test_poly4_lasso,y_test))
```

    [[  0.        ]
     [  0.26262626]
     [  0.52525253]
     [  0.78787879]
     [  1.05050505]
     [  1.31313131]
     [  1.57575758]
     [  1.83838384]
     [  2.1010101 ]
     [  2.36363636]
     [  2.62626263]
     [  2.88888889]
     [  3.15151515]
     [  3.41414141]
     [  3.67676768]
     [  3.93939394]
     [  4.2020202 ]
     [  4.46464646]
     [  4.72727273]
     [  4.98989899]
     [  5.25252525]
     [  5.51515152]
     [  5.77777778]
     [  6.04040404]
     [  6.3030303 ]
     [  6.56565657]
     [  6.82828283]
     [  7.09090909]
     [  7.35353535]
     [  7.61616162]
     [  7.87878788]
     [  8.14141414]
     [  8.4040404 ]
     [  8.66666667]
     [  8.92929293]
     [  9.19191919]
     [  9.45454545]
     [  9.71717172]
     [  9.97979798]
     [ 10.24242424]
     [ 10.50505051]
     [ 10.76767677]
     [ 11.03030303]
     [ 11.29292929]
     [ 11.55555556]
     [ 11.81818182]
     [ 12.08080808]
     [ 12.34343434]
     [ 12.60606061]
     [ 12.86868687]
     [ 13.13131313]
     [ 13.39393939]
     [ 13.65656566]
     [ 13.91919192]
     [ 14.18181818]
     [ 14.44444444]
     [ 14.70707071]
     [ 14.96969697]
     [ 15.23232323]
     [ 15.49494949]
     [ 15.75757576]
     [ 16.02020202]
     [ 16.28282828]
     [ 16.54545455]
     [ 16.80808081]
     [ 17.07070707]
     [ 17.33333333]
     [ 17.5959596 ]
     [ 17.85858586]
     [ 18.12121212]
     [ 18.38383838]
     [ 18.64646465]
     [ 18.90909091]
     [ 19.17171717]
     [ 19.43434343]
     [ 19.6969697 ]
     [ 19.95959596]
     [ 20.22222222]
     [ 20.48484848]
     [ 20.74747475]
     [ 21.01010101]
     [ 21.27272727]
     [ 21.53535354]
     [ 21.7979798 ]
     [ 22.06060606]     [ 22.32323232]
     [ 22.58585859]
     [ 22.84848485]
     [ 23.11111111]
     [ 23.37373737]
     [ 23.63636364]
     [ 23.8989899 ]
     [ 24.16161616]
     [ 24.42424242]
     [ 24.68686869]
     [ 24.94949495]
     [ 25.21212121]
     [ 25.47474747]
     [ 25.73737374]
     [ 26.        ]]


![](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/Skills%20of%20debugging%20model/output_11_1.png)


    the accuracy is: 0.809588079577
    the accuracy is: 0.83889268736

    C:\Users\xxz\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:466: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations
      ConvergenceWarning)

### 2.3 L2 Regularization


```python
from sklearn.linear_model import LinearRegression
'''
使用线性回归模型在比萨训练样本上进行拟合
'''

#输入训练样本的特征及其目标值，分别存储在变量X_train与y_train
X_train = [[6],[8],[10],[14],[18]]
y_train = [[7],[9],[13],[17.5],[18]]

X_test = [[6],[8],[11],[16]]
y_test = [[8],[12],[15],[18]]

#使用默认配置初始化线性回归模型
regressor = LinearRegression()
#直接以比萨的半径作为特征训练模型
regressor.fit(X_train,y_train)

import numpy as np
#在x轴上从0到25均匀采样100个数据点
xx = np.linspace(0,26,100)
# print(xx.shape[0])
xx = xx.reshape(xx.shape[0],1)
print(xx)
#以上述100个数据点作为基准，预测回归直线
yy = regressor.predict(xx)

#对回归预测到直线进行作图
import matplotlib.pyplot as plt
plt.scatter(X_train,y_train) #画出样本散点图

plt1,=plt.plot(xx,yy,label="Degree=1") #画出回归预测直线

plt.axis([0,25,0,25]) #设置坐标轴的范围
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of pizza')
plt.legend(handles=[plt1])
plt.show()

'''
Ridge模型在4次多项式特征上拟合的表现,等同于在线性回归目标加入L2范数正则化，避免参数过拟合
'''
'''
使用4次多项式回归模型在披萨训练样本上进行拟合
'''

from sklearn.preprocessing import PolynomialFeatures
#使用PolynomialFratures将x特征映射到四次多项式，存储在X_train_poly4
poly4 = PolynomialFeatures(degree=4)
X_train_poly4 = poly4.fit_transform(X_train)
#以线性回归器为基础，初始化回归模型，尽管特征的维度有所提升，但是模型基础仍然是线性模型
regressor_poly4 = LinearRegression()
#对4次多项式回归模型进行训练
regressor_poly4.fit(X_train_poly4,y_train)

X_test_poly4 = poly4.transform(X_test)
print('the accuracy is:',regressor_poly4.score(X_test_poly4,y_test))
#输出上述参数的平方和，验证参数之前的巨大差异
print(regressor_poly4.coef_)
print(np.sum(regressor_poly4.coef_ ** 2))


#从sklearn.linear_model中导入Lasso
from sklearn.linear_model import Ridge
#从使用配置初始化Lasso
Ridge_poly4 = Ridge()
#从使用Lasso对4次多项式特征进行拟合
Ridge_poly4.fit(X_train_poly4,y_train)
X_test_poly4_Ridge = poly4.transform(X_test)
print('the accuracy is:',Ridge_poly4.score(X_test_poly4_Ridge,y_test))
#输出上述参数的平方和，验证参数之前的巨大差异
print(Ridge_poly4.coef_)
print(np.sum(Ridge_poly4.coef_ ** 2))
```

    [[  0.        ]
     [  0.26262626]
     [  0.52525253]
     [  0.78787879]
     [  1.05050505]
     [  1.31313131]
     [  1.57575758]
     [  1.83838384]
     [  2.1010101 ]
     [  2.36363636]
     [  2.62626263]
     [  2.88888889]
     [  3.15151515]
     [  3.41414141]
     [  3.67676768]
     [  3.93939394]
     [  4.2020202 ]
     [  4.46464646]
     [  4.72727273]
     [  4.98989899]
     [  5.25252525]
     [  5.51515152]
     [  5.77777778]
     [  6.04040404]
     [  6.3030303 ]
     [  6.56565657]
     [  6.82828283]
     [  7.09090909]
     [  7.35353535]
     [  7.61616162]
     [  7.87878788]
     [  8.14141414]
     [  8.4040404 ]
     [  8.66666667]
     [  8.92929293]
     [  9.19191919]
     [  9.45454545]
     [  9.71717172]
     [  9.97979798]
     [ 10.24242424]
     [ 10.50505051]
     [ 10.76767677]
     [ 11.03030303]
     [ 11.29292929]
     [ 11.55555556]
     [ 11.81818182]
     [ 12.08080808]
     [ 12.34343434]
     [ 12.60606061]
     [ 12.86868687]
     [ 13.13131313]
     [ 13.39393939]
     [ 13.65656566]
     [ 13.91919192]
     [ 14.18181818]
     [ 14.44444444]
     [ 14.70707071]
     [ 14.96969697]
     [ 15.23232323]
     [ 15.49494949]
     [ 15.75757576]
     [ 16.02020202]
     [ 16.28282828]
     [ 16.54545455]
     [ 16.80808081]
     [ 17.07070707]
     [ 17.33333333]
     [ 17.5959596 ]
     [ 17.85858586]
     [ 18.12121212]
     [ 18.38383838]
     [ 18.64646465]
     [ 18.90909091]
     [ 19.17171717]
     [ 19.43434343]
     [ 19.6969697 ]
     [ 19.95959596]
     [ 20.22222222]
     [ 20.48484848]
     [ 20.74747475]
     [ 21.01010101]
     [ 21.27272727]
     [ 21.53535354]
     [ 21.7979798 ]
     [ 22.06060606]
     [ 22.32323232]
     [ 22.58585859]
     [ 22.84848485]
     [ 23.11111111]     [ 23.37373737]
     [ 23.63636364]
     [ 23.8989899 ]
     [ 24.16161616]
     [ 24.42424242]
     [ 24.68686869]
     [ 24.94949495]
     [ 25.21212121]
     [ 25.47474747]
     [ 25.73737374]
     [ 26.        ]]


![](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/Skills%20of%20debugging%20model/output_13_1.png)


    the accuracy is: 0.809588079577
    [[  0.00000000e+00  -2.51739583e+01   3.68906250e+00  -2.12760417e-01
        4.29687500e-03]]
    647.382645737
    the accuracy is: 0.837420175937
    [[ 0.         -0.00492536  0.12439632 -0.00046471 -0.00021205]]
    0.0154989652036

## 3. 模型检验 


```python
'''
模型检验主要有两种方法:
(1) 留一验证
(2) 交叉验证
'''
```




    '\n模型检验主要有两种方法:\n(1) 留一验证\n(2) 交叉验证\n'


## 4. 超参数搜索 

### 4.1 Grid Searching 


```python
import numpy as np

'''
网格搜索(Grid Searching)对多种超参数组合的空间进行暴力搜索，每一套超参数组合被代入到学习函数中作为新的模型，并且比较新模型之间的性能，
每个模型都会采用交叉验证的方法在多组相同的训练和开发数据集下进行评估。
'''
'''
使用单线程对文本分类的朴素贝叶斯模型的超参数组合执行网络搜索
'''
#从sklearn.datasets中导入20类新闻文本抓取器
from sklearn.datasets import fetch_20newsgroups

#使用新闻抓取器从互联网上下载所有数据，并且存储在变量news中
news = fetch_20newsgroups(subset='all')

#从sklearn.cross_validation导入train_test_split用来分割数据
from sklearn.cross_validation import train_test_split

#对前3000条新闻文本进行数据分割，25%文本用于未来测试
X_train,X_test,y_train,y_test = train_test_split(news.data[:3000],news.target[:3000],test_size=0.25,random_state=33)

#导入支持向量机（分类）模型
from sklearn.svm import SVC
#导入TfidfVectorizer文本抽取器
from sklearn.feature_extraction.text import TfidfVectorizer

#导入Pipeline
from sklearn.pipeline import Pipeline

#使用Pipeline简化系统搭建流程，将文本抽取与分类器模型串联起来
clf = Pipeline([('vect',TfidfVectorizer(stop_words='english',analyzer='word')),('svc',SVC())])

#这里需要试验的2个超参数的个数分别是4、3，svc__gamma的参数共有10^-2,10^-1……，这样我们一共有12种的超参数组合，12个不同参数下的模型
parameters = {'svc__gamma':np.logspace(-2,1,4),'svc__C':np.logspace(-1,1,3)}

#从sklearn.grid_search中导入网络搜索模块GridSearchCV
from sklearn.grid_search import GridSearchCV

#将12组参数组合以及初始化的Pipline包括3折交叉验证的要求全部告知GridSearchCV
gs = GridSearchCV(clf,parameters,verbose=2,refit=True,cv=3)

#执行单线程网络搜索
time_ = gs.fit(X_train,y_train)
print(gs.best_params_, gs.best_score_)

#输出最佳模型在测试集上的准确性
print(gs.score(X_test,y_test))
```

    Fitting 3 folds for each of 12 candidates, totalling 36 fits
    [CV] svc__C=0.1, svc__gamma=0.01 .....................................
    [CV] ............................ svc__C=0.1, svc__gamma=0.01 -   7.9s
    [CV] svc__C=0.1, svc__gamma=0.01 .....................................
    [CV] ............................ svc__C=0.1, svc__gamma=0.01 -   6.0s
    [CV] svc__C=0.1, svc__gamma=0.01 .....................................
    [CV] ............................ svc__C=0.1, svc__gamma=0.01 -   6.3s
    [CV] svc__C=0.1, svc__gamma=0.1 ......................................
    [CV] ............................. svc__C=0.1, svc__gamma=0.1 -   5.9s
    [CV] svc__C=0.1, svc__gamma=0.1 ......................................
    [CV] ............................. svc__C=0.1, svc__gamma=0.1 -   5.9s
    [CV] svc__C=0.1, svc__gamma=0.1 ......................................
    [CV] ............................. svc__C=0.1, svc__gamma=0.1 -   6.2s
    [CV] svc__C=0.1, svc__gamma=1.0 ......................................
    [CV] ............................. svc__C=0.1, svc__gamma=1.0 -   6.1s
    [CV] svc__C=0.1, svc__gamma=1.0 ......................................
    [CV] ............................. svc__C=0.1, svc__gamma=1.0 -   6.0s
    [CV] svc__C=0.1, svc__gamma=1.0 ......................................
    [CV] ............................. svc__C=0.1, svc__gamma=1.0 -   6.1s
    [CV] svc__C=0.1, svc__gamma=10.0 .....................................
    [CV] ............................ svc__C=0.1, svc__gamma=10.0 -   6.3s
    [CV] svc__C=0.1, svc__gamma=10.0 .....................................
    [CV] ............................ svc__C=0.1, svc__gamma=10.0 -   6.3s
    [CV] svc__C=0.1, svc__gamma=10.0 .....................................
    [CV] ............................ svc__C=0.1, svc__gamma=10.0 -   6.7s
    [CV] svc__C=1.0, svc__gamma=0.01 .....................................
    [CV] ............................ svc__C=1.0, svc__gamma=0.01 -   6.3s
    [CV] svc__C=1.0, svc__gamma=0.01 .....................................
    [CV] ............................ svc__C=1.0, svc__gamma=0.01 -   6.1s
    [CV] svc__C=1.0, svc__gamma=0.01 .....................................
    [CV] ............................ svc__C=1.0, svc__gamma=0.01 -   6.2s
    [CV] svc__C=1.0, svc__gamma=0.1 ......................................
    [CV] ............................. svc__C=1.0, svc__gamma=0.1 -   5.8s
    [CV] svc__C=1.0, svc__gamma=0.1 ......................................
    [CV] ............................. svc__C=1.0, svc__gamma=0.1 -   5.9s
    [CV] svc__C=1.0, svc__gamma=0.1 ......................................
    [CV] ............................. svc__C=1.0, svc__gamma=0.1 -   6.2s
    [CV] svc__C=1.0, svc__gamma=1.0 ......................................
    [CV] ............................. svc__C=1.0, svc__gamma=1.0 -   6.0s
    [CV] svc__C=1.0, svc__gamma=1.0 ......................................
    [CV] ............................. svc__C=1.0, svc__gamma=1.0 -   6.1s
    [CV] svc__C=1.0, svc__gamma=1.0 ......................................
    [CV] ............................. svc__C=1.0, svc__gamma=1.0 -   6.3s
    [CV] svc__C=1.0, svc__gamma=10.0 .....................................
    [CV] ............................ svc__C=1.0, svc__gamma=10.0 -   6.4s
    [CV] svc__C=1.0, svc__gamma=10.0 .....................................
    [CV] ............................ svc__C=1.0, svc__gamma=10.0 -   6.4s
    [CV] svc__C=1.0, svc__gamma=10.0 .....................................
    [CV] ............................ svc__C=1.0, svc__gamma=10.0 -   6.4s
    [CV] svc__C=10.0, svc__gamma=0.01 ....................................
    [CV] ........................... svc__C=10.0, svc__gamma=0.01 -   6.3s
    [CV] svc__C=10.0, svc__gamma=0.01 ....................................
    [CV] ........................... svc__C=10.0, svc__gamma=0.01 -   6.1s
    [CV] svc__C=10.0, svc__gamma=0.01 ....................................
    [CV] ........................... svc__C=10.0, svc__gamma=0.01 -   6.1s
    [CV] svc__C=10.0, svc__gamma=0.1 .....................................
    [CV] ............................ svc__C=10.0, svc__gamma=0.1 -   6.0s
    [CV] svc__C=10.0, svc__gamma=0.1 .....................................
    [CV] ............................ svc__C=10.0, svc__gamma=0.1 -   6.0s
    [CV] svc__C=10.0, svc__gamma=0.1 .....................................
    [CV] ............................ svc__C=10.0, svc__gamma=0.1 -   6.2s
    [CV] svc__C=10.0, svc__gamma=1.0 .....................................
    [CV] ............................ svc__C=10.0, svc__gamma=1.0 -   6.0s
    [CV] svc__C=10.0, svc__gamma=1.0 .....................................
    [CV] ............................ svc__C=10.0, svc__gamma=1.0 -   6.1s
    [CV] svc__C=10.0, svc__gamma=1.0 .....................................
    [CV] ............................ svc__C=10.0, svc__gamma=1.0 -   6.2s
    [CV] svc__C=10.0, svc__gamma=10.0 ....................................
    [CV] ........................... svc__C=10.0, svc__gamma=10.0 -   6.2s
    [CV] svc__C=10.0, svc__gamma=10.0 ....................................
    [CV] ........................... svc__C=10.0, svc__gamma=10.0 -   6.3s
    [CV] svc__C=10.0, svc__gamma=10.0 ....................................
    [CV] ........................... svc__C=10.0, svc__gamma=10.0 -   6.3s

    [Parallel(n_jobs=1)]: Done  36 out of  36 | elapsed:  3.8min finished
​    

    {'svc__C': 10.0, 'svc__gamma': 0.10000000000000001} 0.790666666667
    0.822666666667

### 4.2 Parallel Grid Search 


```python
import numpy as np

'''
网格搜索(Grid Searching)对多种超参数组合的空间进行暴力搜索，每一套超参数组合被代入到学习函数中作为新的模型，并且比较新模型之间的性能，
每个模型都会采用交叉验证的方法在多组相同的训练和开发数据集下进行评估。
'''
'''
使用多线程对文本分类的朴素贝叶斯模型的超参数组合执行网络搜索
'''
#从sklearn.datasets中导入20类新闻文本抓取器
from sklearn.datasets import fetch_20newsgroups
import numpy as np
#使用新闻抓取器从互联网上下载所有数据，并且存储在变量news中
news = fetch_20newsgroups(subset='all')

#从sklearn.cross_validation导入train_test_split用来分割数据
from sklearn.cross_validation import train_test_split

#对前3000条新闻文本进行数据分割，25%文本用于未来测试
X_train,X_test,y_train,y_test = train_test_split(news.data[:3000],news.target[:3000],test_size=0.25,random_state=33)

#导入支持向量机（分类）模型
from sklearn.svm import SVC
#导入TfidfVectorizer文本抽取器
from sklearn.feature_extraction.text import TfidfVectorizer
#导入Pipeline
from sklearn.pipeline import Pipeline

#使用Pipeline简化系统搭建流程，将文本抽取与分类器模型串联起来
clf = Pipeline([('vect',TfidfVectorizer(stop_words='english',analyzer='word')),('svc',SVC())])

#这里需要试验的2个超参数的个数分别是4、3，svc__gamma的参数共有10^-2,10^-1……，这样我们一共有12种的超参数组合，12个不同参数下的模型
parameters = {'svc__gamma':np.logspace(-2,1,4),'svc__C':np.logspace(-1,1,3)}

#从sklearn.grid_search中导入网络搜索模块GridSearchCV
from sklearn.grid_search import GridSearchCV


if __name__ == '__main__':
    #将12组参数组合以及初始化的Pipline包括3折交叉验证的要求全部告知GridSearchCV
    #初始化配置并行网络搜索，n_jobs=-1代表使用该计算机全部的CPU
    gs = GridSearchCV(clf,parameters,verbose=2,refit=True,cv=3,n_jobs=-1)

    #执行多线程并行网络搜索
    time_ = gs.fit(X_train,y_train)
    print(gs.best_params_, gs.best_score_)

    #输出最佳模型在测试集上的准确性
    print(gs.score(X_test,y_test))
```

    Fitting 3 folds for each of 12 candidates, totalling 36 fits
​    

    [Parallel(n_jobs=-1)]: Done  36 out of  36 | elapsed:   59.9s finished
​    

    {'svc__C': 10.0, 'svc__gamma': 0.10000000000000001} 0.790666666667
    0.822666666667
