---
layout:     post                    # 使用的布局（不需要改）
title:      【转】Supervised Learning Model  # 标题 
subtitle:   Regression Prediction #副标题
date:       2017-09-16              # 时间
author:     Brian                      # 作者
header-img: img/post-bg-2015.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 转载
---

# Supervised Learning Model-Regression Prediction

Author: Xie Zhong-zhao

## 1. Linear Regression


```python
#从sklearn.datasets导入波士顿房价数据读取器
from sklearn.datasets import load_boston
#读取房价数据到存储变量boston中
boston = load_boston()
#输出数据描述
print(boston.DESCR)

'''
线性回归和随机梯度下降两种算法对比，MAE, MSE, R-square
'''

'''
从sklearn.cross_validation导入数据分割
'''
from sklearn.cross_validation import train_test_split
import numpy as np
X = boston.data
y = boston.target
#随机采样25%的数据构建测试样本，其余作为训练样本
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=33,test_size=0.25)
#分析回归目标值的差异
print('the max target value is',np.max(boston.target))
print('the min target value is',np.min(boston.target))
print('the average target value is',np.mean(boston.target))

'''
训练与测试标准化处理
'''
from sklearn.preprocessing import StandardScaler

#分别初始化对特征和目标值的标准器
ss_X = StandardScaler()
ss_y = StandardScaler()

#分别对训练和测试数据的特征以及目标值进行标准化处理
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

'''
使用线性回归模型LinearRegression和SGDRegressor对美国波士顿地区房价进行预测
'''
#从sklearn.linear_model导入LinearRegression
from sklearn.linear_model import LinearRegression

#使用默认的配置初始化线性回归器LinearRegression
lr = LinearRegression()
#使用训练数据进行参数估计
lr.fit(X_train,y_train)
#对测试数据进行回归预测
lr_y_predict = lr.predict(X_test)

#从sklearn.linear_model导入SGDRegresion
from sklearn.linear_model import SGDRegressor
#使用默认配置初始化线性回归SGDRegressor
sgdr = SGDRegressor()
#使用训练数据进行参数估计
sgdr.fit(X_train,y_train)
#对测试数据进行回归预测
sgdr_y_predict = sgdr.predict(X_test)


'''
LinearRegression使用三种回归评价机制以及两种调用R-square评价模块的方法，对本节模型的回归性能做出评价
'''
#使用LinearRegression模型自带的评估模块，并输出评估结果
print('the value of default measurement of LinearRegression is',lr.score(X_test,y_test))
#使用sklearn.metrics依次导入r2_score,mean_squared_error以及mean_absolute_error用于回归性能评估
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
#使用r2_score模块，并输出评估结果
print('the value of R-squared of LinearRegression is',r2_score(y_test,lr_y_predict))
#使用mean_squared_error模块，并输出评估结果
print('the mean squared error of LinearRegression is',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))
#使用mean_absolute_error模块，并输出评估结果
print('the mean absolute error of LinearRegression is',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))


'''
SGDRegressor使用三种回归评价机制以及两种调用R-square评价模块的方法，对本节模型的回归性能做出评价
'''
#使用SGDRegressor模型自带的评估模块，并输出评估结果
print('the value of default measurement of SGDRegressor is',sgdr.score(X_test,y_test))
#使用sklearn.metrics依次导入r2_score,mean_squared_error以及mean_absolute_error用于回归性能评估
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
#使用r2_score模块，并输出评估结果
print('the value of R-squared of SGDRegressor is',r2_score(y_test,sgdr_y_predict))
#使用mean_squared_error模块，并输出评估结果
print('the mean squared error of SGDRegressor is',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_predict)))
#使用mean_absolute_error模块，并输出评估结果
print('the mean absolute error of SGDRegressor is',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_predict)))

```

    Boston House Prices dataset
    
    Notes
    ------
    Data Set Characteristics:  
    
        :Number of Instances: 506 
    
        :Number of Attributes: 13 numeric/categorical predictive
        
        :Median Value (attribute 14) is usually the target
    
        :Attribute Information (in order):
            - CRIM     per capita crime rate by town
            - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
            - INDUS    proportion of non-retail business acres per town
            - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
            - NOX      nitric oxides concentration (parts per 10 million)
            - RM       average number of rooms per dwelling
            - AGE      proportion of owner-occupied units built prior to 1940
            - DIS      weighted distances to five Boston employment centres
            - RAD      index of accessibility to radial highways
            - TAX      full-value property-tax rate per $10,000
            - PTRATIO  pupil-teacher ratio by town
            - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
            - LSTAT    % lower status of the population
            - MEDV     Median value of owner-occupied homes in $1000's
    
        :Missing Attribute Values: None
    
        :Creator: Harrison, D. and Rubinfeld, D.L.
    
    This is a copy of UCI ML housing dataset.
    http://archive.ics.uci.edu/ml/datasets/Housing
    
    This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
    
    The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
    prices and the demand for clean air', J. Environ. Economics & Management,
    vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
    ...', Wiley, 1980.   N.B. Various transformations are used in the table on
    pages 244-261 of the latter.
    
    The Boston house-price data has been used in many machine learning papers that address regression
    problems.   
         
    **References**
    
       - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
       - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
       - many more! (see http://archive.ics.uci.edu/ml/datasets/Housing)
    
    the max target value is 50.0
    the min target value is 5.0
    the average target value is 22.5328063241
    
    C:\Users\xxz\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:583: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)
    C:\Users\xxz\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:646: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)
    C:\Users\xxz\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:646: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)
    
    the value of default measurement of LinearRegression is 0.6763403831
    the value of R-squared of LinearRegression is 0.6763403831
    the mean squared error of LinearRegression is 25.0969856921
    the mean absolute error of LinearRegression is 3.5261239964
    the value of default measurement of SGDRegressor is 0.658690771038
    the value of R-squared of SGDRegressor is 0.658690771038
    the mean squared error of SGDRegressor is 26.4655594599
    the mean absolute error of SGDRegressor is 3.50136123888

## 2. SVM Regression


```python
#从sklearn.datasets导入波士顿房价数据读取器
from sklearn.datasets import load_boston
#读取房价数据到存储变量boston中
boston = load_boston()
#输出数据描述
print(boston.DESCR)

'''
支持向量机的三种核函数配置下的对比，MAE, MSE, R-square
'''

'''
从sklearn.cross_validation导入数据分割
'''
from sklearn.cross_validation import train_test_split
import numpy as np
X = boston.data
y = boston.target
#随机采样25%的数据构建测试样本，其余作为训练样本
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=33,test_size=0.25)
#分析回归目标值的差异
print('the max target value is',np.max(boston.target))
print('the min target value is',np.min(boston.target))
print('the average target value is',np.mean(boston.target))

'''
训练与测试标准化处理
'''
from sklearn.preprocessing import StandardScaler

#分别初始化对特征和目标值的标准器
ss_X = StandardScaler()
ss_y = StandardScaler()

#分别对训练和测试数据的特征以及目标值进行标准化处理
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)


#从sklearn.svm中导入支持向量机（回归）模型
from sklearn.svm import SVR

#使用线性核函数的配置的支持向量机进行回归训练，并且对测试样本进行预测
linear_svr = SVR(kernel = 'linear')
linear_svr.fit(X_train,y_train)
linear_svr_y_predict = linear_svr.predict(X_test)

#使用多项式核函数的配置的支持向量机进行回归训练，并且对测试样本进行预测
poly_svr = SVR(kernel = 'poly')
poly_svr.fit(X_train,y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

#使用径向基核函数的配置的支持向量机进行回归训练，并且对测试样本进行预测
rbf_svr = SVR(kernel= 'rbf')
rbf_svr.fit(X_train,y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)


'''
对三种核函数的配置下的支持向量机回归模型在相同的测试集上进行性能评估
'''
#使用R-square,MSE和MAE指标对三种配置的支持向量机（回归）模型在测试集上进行性能评估
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
'''
线性核函数配置对测试样本进行预测
'''
print('the accuracy of Linear SVR is',linear_svr.score(X_test,y_test))
print('R-square value of Linear SVR is', linear_svr.score(X_test,y_test))
print('the mean square error of Linear SVR is',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_y_predict)))
print('the mean absoluate error of linear SVR is',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_y_predict)))
print("----------------------------------------------------")
'''
多项式核函数对测试样本进行预测
'''
print('the accuracy of poly SVR is',poly_svr.score(X_test,y_test))
print('R-square value of poly SVR is', poly_svr.score(X_test,y_test))
print('the mean square error of poly SVR is',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(poly_svr_y_predict)))
print('the mean absoluate error of poly SVR is',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(poly_svr_y_predict)))
print("----------------------------------------------------")
'''
径向基核函数对测试样本进行预测
'''
print('the accuracy of rbf SVR is',rbf_svr.score(X_test,y_test))
print('R-square value of rbf SVR is', rbf_svr.score(X_test,y_test))
print('the mean square error of rbf SVR is',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rbf_svr_y_predict)))
print('the mean absoluate error of rbf SVR is',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rbf_svr_y_predict)))
print("----------------------------------------------------")

```

    Boston House Prices dataset
    
    Notes
    ------
    Data Set Characteristics:  
    
        :Number of Instances: 506 
    
        :Number of Attributes: 13 numeric/categorical predictive
        
        :Median Value (attribute 14) is usually the target
    
        :Attribute Information (in order):
            - CRIM     per capita crime rate by town
            - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
            - INDUS    proportion of non-retail business acres per town
            - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
            - NOX      nitric oxides concentration (parts per 10 million)
            - RM       average number of rooms per dwelling
            - AGE      proportion of owner-occupied units built prior to 1940
            - DIS      weighted distances to five Boston employment centres
            - RAD      index of accessibility to radial highways
            - TAX      full-value property-tax rate per $10,000
            - PTRATIO  pupil-teacher ratio by town
            - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
            - LSTAT    % lower status of the population
            - MEDV     Median value of owner-occupied homes in $1000's
    
        :Missing Attribute Values: None
    
        :Creator: Harrison, D. and Rubinfeld, D.L.
    
    This is a copy of UCI ML housing dataset.
    http://archive.ics.uci.edu/ml/datasets/Housing
    
    This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
    
    The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
    prices and the demand for clean air', J. Environ. Economics & Management,
    vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
    ...', Wiley, 1980.   N.B. Various transformations are used in the table on
    pages 244-261 of the latter.
    
    The Boston house-price data has been used in many machine learning papers that address regression
    problems.   
         
    **References**
    
       - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
       - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
       - many more! (see http://archive.ics.uci.edu/ml/datasets/Housing)
    
    the max target value is 50.0
    the min target value is 5.0
    the average target value is 22.5328063241
    the accuracy of Linear SVR is 0.65171709743
    R-square value of Linear SVR is 0.65171709743
    the mean square error of Linear SVR is 27.0063071393
    the mean absoluate error of linear SVR is 3.42667291687
    ----------------------------------------------------
    the accuracy of poly SVR is 0.404454058003
    R-square value of poly SVR is 0.404454058003
    the mean square error of poly SVR is 46.179403314
    the mean absoluate error of poly SVR is 3.75205926674
    ----------------------------------------------------
    the accuracy of rbf SVR is 0.756406891227
    R-square value of rbf SVR is 0.756406891227
    the mean square error of rbf SVR is 18.8885250008
    the mean absoluate error of rbf SVR is 2.60756329798
    ----------------------------------------------------
    
    C:\Users\xxz\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:583: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)
    C:\Users\xxz\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:646: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)
    C:\Users\xxz\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:646: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)

## 3. KNN Regression


```python
#从sklearn.datasets导入波士顿房价数据读取器
from sklearn.datasets import load_boston
#读取房价数据到存储变量boston中
boston = load_boston()
#输出数据描述
print(boston.DESCR)

'''
KNN算法对K个近邻目标数值使用普通的算术平均算法同时考虑距离的差距进行加权平均，MAE, MSE, R-square指标比较回归性能的差异
'''

'''
从sklearn.cross_validation导入数据分割
'''
from sklearn.cross_validation import train_test_split
import numpy as np
X = boston.data
y = boston.target
#随机采样25%的数据构建测试样本，其余作为训练样本
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=33,test_size=0.25)
#分析回归目标值的差异
print('the max target value is',np.max(boston.target))
print('the min target value is',np.min(boston.target))
print('the average target value is',np.mean(boston.target))

'''
训练与测试标准化处理
'''
from sklearn.preprocessing import StandardScaler

#分别初始化对特征和目标值的标准器
ss_X = StandardScaler()
ss_y = StandardScaler()

#分别对训练和测试数据的特征以及目标值进行标准化处理
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

'''
使用两种不同的配置的K近邻回归模型对波士顿房价数据进行预测回归
'''
#从sklearn.neighbors导入KNeighborRegressior(K近邻回归器)
from sklearn.neighbors import KNeighborsRegressor

#初始化K近邻回归器，并且调整配置，使得预测的方式为平均回归: weights = 'uniform'
uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(X_train,y_train)
uni_knr_y_predict = uni_knr.predict(X_test)

#初始化K近邻回归器，使得预测方式为根据距离加权回归: weights = 'distance'
dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(X_train,y_train)
dis_knr_y_predict = dis_knr.predict(X_test)

'''
对两种不同的配置的K近邻回归模型在波士顿房价数据上进行预测性能的评估
'''
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#使用R-squared,MSE以及MAE三种指标对‘平均回归’配置的K近邻模型在测试集上进行性能评估
print('the accuracy of KNN uniform is',uni_knr.score(X_test,y_test))
print('R-square value of uniform-weighted KNeighborRegression:',uni_knr.score(X_test,y_test))
print('the mean squared error of uniform-weighted KNeighborRegression:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(uni_knr_y_predict)))
print('the mean absolute error of uniform-weighted KNeighborRegression',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(uni_knr_y_predict)))

#使用R-squared,MSE以及MAE三种指标对‘距离加权回归’配置的K近邻模型在测试集上进行性能评估
print('the accuracy of KNN distance is',dis_knr.score(X_test,y_test))
print('R-square value of distance-weighted KNeighborRegression:',dis_knr.score(X_test,y_test))
print('the mean squared error of distance-weighted KNeighborRegression:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dis_knr_y_predict)))
print('the mean absolute error of distance-weighted KNeighborRegression',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dis_knr_y_predict)))
```

    Boston House Prices dataset
    
    Notes
    ------
    Data Set Characteristics:  
    
        :Number of Instances: 506 
    
        :Number of Attributes: 13 numeric/categorical predictive
        
        :Median Value (attribute 14) is usually the target
    
        :Attribute Information (in order):
            - CRIM     per capita crime rate by town
            - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
            - INDUS    proportion of non-retail business acres per town
            - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
            - NOX      nitric oxides concentration (parts per 10 million)
            - RM       average number of rooms per dwelling
            - AGE      proportion of owner-occupied units built prior to 1940
            - DIS      weighted distances to five Boston employment centres
            - RAD      index of accessibility to radial highways
            - TAX      full-value property-tax rate per $10,000
            - PTRATIO  pupil-teacher ratio by town
            - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
            - LSTAT    % lower status of the population
            - MEDV     Median value of owner-occupied homes in $1000's
    
        :Missing Attribute Values: None
    
        :Creator: Harrison, D. and Rubinfeld, D.L.
    
    This is a copy of UCI ML housing dataset.
    http://archive.ics.uci.edu/ml/datasets/Housing
    
    This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
    
    The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
    prices and the demand for clean air', J. Environ. Economics & Management,
    vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
    ...', Wiley, 1980.   N.B. Various transformations are used in the table on
    pages 244-261 of the latter.
    
    The Boston house-price data has been used in many machine learning papers that address regression
    problems.   
         
    **References**
    
       - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
       - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
       - many more! (see http://archive.ics.uci.edu/ml/datasets/Housing)
    
    the max target value is 50.0
    the min target value is 5.0
    the average target value is 22.5328063241
    the accuracy of KNN uniform is 0.690345456461
    R-square value of uniform-weighted KNeighborRegression: 0.690345456461
    the mean squared error of uniform-weighted KNeighborRegression: 24.0110141732
    the mean absolute error of uniform-weighted KNeighborRegression 2.96803149606
    the accuracy of KNN distance is 0.719758997016
    R-square value of distance-weighted KNeighborRegression: 0.719758997016
    the mean squared error of distance-weighted KNeighborRegression: 21.7302501609
    the mean absolute error of distance-weighted KNeighborRegression 2.80505687851
    
    C:\Users\xxz\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:583: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)
    C:\Users\xxz\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:646: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)
    C:\Users\xxz\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:646: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)

## 4. Decision Tree Regressor


```python
#从sklearn.datasets导入波士顿房价数据读取器
from sklearn.datasets import load_boston
#读取房价数据到存储变量boston中
boston = load_boston()
#输出数据描述
print(boston.DESCR)

'''
支持向量机的三种核函数配置下的对比，MAE, MSE, R-square
'''

'''
从sklearn.cross_validation导入数据分割
'''
from sklearn.cross_validation import train_test_split
import numpy as np
X = boston.data
y = boston.target
#随机采样25%的数据构建测试样本，其余作为训练样本
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=33,test_size=0.25)
#分析回归目标值的差异
print('the max target value is',np.max(boston.target))
print('the min target value is',np.min(boston.target))
print('the average target value is',np.mean(boston.target))

'''
训练与测试标准化处理
'''
from sklearn.preprocessing import StandardScaler

#分别初始化对特征和目标值的标准器
ss_X = StandardScaler()
ss_y = StandardScaler()

#分别对训练和测试数据的特征以及目标值进行标准化处理
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

'''
使用回归树对波士顿房价训练数据进行学习，并对测试数据进行预测
'''
#从sklearn.tree中导入DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
#使用默认配置初始化DecisionTreeRegressor
dtr = DecisionTreeRegressor()
#用波士顿的房价训练数据构建树
dtr.fit(X_train,y_train)
#使用默认配置单一回归树对测试数据进行预测，并存储在dtr_y_predict
dtr_y_predict = dtr.predict(X_test)

"""
对单一的回归树模型在美国波士顿房价测试数据上预测性能进行评估
"""
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print('R-squared value of DecisionTreeRegressor:',dtr.score(X_test,y_test))
print('the mean squared error of DecisionTreeRegressor:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dtr_y_predict)))
print('the mean absoluate error of DecisionTreeRegressor:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dtr_y_predict)))

```

    Boston House Prices dataset
    
    Notes
    ------
    Data Set Characteristics:  
    
        :Number of Instances: 506 
    
        :Number of Attributes: 13 numeric/categorical predictive
        
        :Median Value (attribute 14) is usually the target
    
        :Attribute Information (in order):
            - CRIM     per capita crime rate by town
            - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
            - INDUS    proportion of non-retail business acres per town
            - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
            - NOX      nitric oxides concentration (parts per 10 million)
            - RM       average number of rooms per dwelling
            - AGE      proportion of owner-occupied units built prior to 1940
            - DIS      weighted distances to five Boston employment centres
            - RAD      index of accessibility to radial highways
            - TAX      full-value property-tax rate per $10,000
            - PTRATIO  pupil-teacher ratio by town
            - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
            - LSTAT    % lower status of the population
            - MEDV     Median value of owner-occupied homes in $1000's
    
        :Missing Attribute Values: None
    
        :Creator: Harrison, D. and Rubinfeld, D.L.
    
    This is a copy of UCI ML housing dataset.
    http://archive.ics.uci.edu/ml/datasets/Housing
    
    This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
    
    The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
    prices and the demand for clean air', J. Environ. Economics & Management,
    vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
    ...', Wiley, 1980.   N.B. Various transformations are used in the table on
    pages 244-261 of the latter.
    
    The Boston house-price data has been used in many machine learning papers that address regression
    problems.   
         
    **References**
    
       - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
       - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
       - many more! (see http://archive.ics.uci.edu/ml/datasets/Housing)
    
    the max target value is 50.0
    the min target value is 5.0
    the average target value is 22.5328063241
    R-squared value of DecisionTreeRegressor: 0.639605796776
    the mean squared error of DecisionTreeRegressor: 27.9454330709
    the mean absoluate error of DecisionTreeRegressor: 3.34881889764
    
    C:\Users\xxz\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:583: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)
    C:\Users\xxz\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:646: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)
    C:\Users\xxz\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:646: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)

## 5. Ensemble Regression 


```python
#从sklearn.datasets导入波士顿房价数据读取器
from sklearn.datasets import load_boston
#读取房价数据到存储变量boston中
boston = load_boston()
#输出数据描述
print(boston.DESCR)

'''
支持向量机的三种核函数配置下的对比，MAE, MSE, R-square
'''

'''
从sklearn.cross_validation导入数据分割
'''
from sklearn.cross_validation import train_test_split
import numpy as np
X = boston.data
y = boston.target
#随机采样25%的数据构建测试样本，其余作为训练样本
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=33,test_size=0.25)
#分析回归目标值的差异
print('the max target value is',np.max(boston.target))
print('the min target value is',np.min(boston.target))
print('the average target value is',np.mean(boston.target))

'''
训练与测试标准化处理
'''
from sklearn.preprocessing import StandardScaler

#分别初始化对特征和目标值的标准器
ss_X = StandardScaler()
ss_y = StandardScaler()

#分别对训练和测试数据的特征以及目标值进行标准化处理
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

'''
使用三种集成回归模型对美国波士顿房价训练数据进行学习，并对测试数据进行预测
'''
#从sklearn.ensemble中导入RandomForestRegressor,ExtraTreesGressor以及GradientBoostingRegeressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

#使用RandomForestRegressor训练模型，对测试数据进行预测，结果存储在变量rfr_y_predict中
rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)
rfr_y_predict = rfr.predict(X_test)

#使用EtraTreeRegressor训练模型，并对测试数据做出预测，结果储存在变量etr_y_predict中
etr = ExtraTreesRegressor()
etr.fit(X_train,y_train)
etr_y_predict = etr.predict(X_test)

#使用GradientBoostingRegressor训练模型，并对测试数据做出预测，结果存储在变量gbr_y_predict中
gbr = GradientBoostingRegressor()
gbr.fit(X_train,y_train)
gbr_y_predict = gbr.predict(X_test)

'''
对三种集成回归模型在美国波士顿房价测试数据的回归预测性能
'''
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#使用R-squared,MSE以及MAE指标对默认配置的随机回归森林在测试集上进行性能评估
print('R-squared value of ExtraTreesRegressor:',rfr.score(X_test,y_test))
print('the mean error of ExtraTreesRegressor:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rfr_y_predict)))
print('the mean absoluate error of ExtraTreesRegressor:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rfr_y_predict)))
print("------------------------------------------------------")

#使用R-squared,MSE以及MAE指标对默认配置的极端回归森林在测试集上进行性能评估
print('R-squared value of GradientBoostingRegressor:',etr.score(X_test,y_test))
print('the mean error of GradientBoostingRegressor:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(etr_y_predict)))
print('the mean absoluate error of GradientBoostingRegressor:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(etr_y_predict)))
print("------------------------------------------------------")

#使用R-squared,MSE以及MAE指标对默认配置的梯度提升回归树在测试集上进行性能评估
print('R-squared value of GradientBoostingRegressor:',gbr.score(X_test,y_test))
print('the mean error of GradientBoostingRegressor:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(gbr_y_predict)))
print('the mean absoluate error of GradientBoostingRegressor:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(gbr_y_predict)))
```

    Boston House Prices dataset
    
    Notes
    ------
    Data Set Characteristics:  
    
        :Number of Instances: 506 
    
        :Number of Attributes: 13 numeric/categorical predictive
        
        :Median Value (attribute 14) is usually the target
    
        :Attribute Information (in order):
            - CRIM     per capita crime rate by town
            - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
            - INDUS    proportion of non-retail business acres per town
            - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
            - NOX      nitric oxides concentration (parts per 10 million)
            - RM       average number of rooms per dwelling
            - AGE      proportion of owner-occupied units built prior to 1940
            - DIS      weighted distances to five Boston employment centres
            - RAD      index of accessibility to radial highways
            - TAX      full-value property-tax rate per $10,000
            - PTRATIO  pupil-teacher ratio by town
            - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
            - LSTAT    % lower status of the population
            - MEDV     Median value of owner-occupied homes in $1000's
    
        :Missing Attribute Values: None
    
        :Creator: Harrison, D. and Rubinfeld, D.L.
    
    This is a copy of UCI ML housing dataset.
    http://archive.ics.uci.edu/ml/datasets/Housing
    
    This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
    
    The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
    prices and the demand for clean air', J. Environ. Economics & Management,
    vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
    ...', Wiley, 1980.   N.B. Various transformations are used in the table on
    pages 244-261 of the latter.
    
    The Boston house-price data has been used in many machine learning papers that address regression
    problems.   
         
    **References**
    
       - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
       - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
       - many more! (see http://archive.ics.uci.edu/ml/datasets/Housing)
    
    the max target value is 50.0
    the min target value is 5.0
    the average target value is 22.5328063241
    R-squared value of ExtraTreesRegressor: 0.796553675979
    the mean error of ExtraTreesRegressor: 15.7754913386
    the mean absoluate error of ExtraTreesRegressor: 2.51464566929
    ------------------------------------------------------
    R-squared value of GradientBoostingRegressor: 0.779116088968
    the mean error of GradientBoostingRegressor: 17.1276244094
    the mean absoluate error of GradientBoostingRegressor: 2.38527559055
    ------------------------------------------------------
    R-squared value of GradientBoostingRegressor: 0.843173295919
    the mean error of GradientBoostingRegressor: 12.1605456565
    the mean absoluate error of GradientBoostingRegressor: 2.27306754066
    
    C:\Users\xxz\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:583: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)
    C:\Users\xxz\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:646: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)
    C:\Users\xxz\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:646: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)
