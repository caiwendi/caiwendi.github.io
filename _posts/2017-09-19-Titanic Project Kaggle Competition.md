
# Titanic Project - Kaggle Competition

Author: Xie Zhong-zhao


```python
import pandas as pd

'''
分别对训练和测试的数据从本地进行读取
'''
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

'''
先分别输出训练和测试数据的基本信息，这是一个好习惯，可以对数据的规模、各个特征的数据类型以及是否有缺失等，有一个总体的了解
'''
print(train.info())
print(test.info())

'''
按照我们之前对Titanic事件经验，人工选取预测有效特征
'''
selected_features = ['Pclass','Sex','Age','Embarked','SibSp','Parch','Fare']

X_train = train[selected_features]
X_test = test[selected_features]

y_train = train['Survived']

'''
通过我们之前对数据的总体观察，得知Embarked特征存在缺失值，需要补充
'''
print(X_train['Embarked'].value_counts())
print(X_test['Embarked'].value_counts())

'''
对于Embarked这种类别型的特征，我们使用出现频率最高的特征值来填充，这也是相对可以减少引入误差的一种填充方法
'''
X_train['Embarked'].fillna('S',inplace=True)
X_test['Embarked'].fillna('S',inplace=True)

'''
而对于Age这种数值类型的特征，我们习惯求平均值或者中位数来填充缺失值，也是相对可以减少引入误差的填充方式
'''
X_train['Age'].fillna(X_train['Age'].mean(),inplace=True)
X_test['Age'].fillna(X_train['Age'].mean(),inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(),inplace=True)

'''
重新对处理后的训练和测试数据进行查验，发现一切就绪
'''
print(X_train.info())
print(X_test.info())

'''
接下来便是采用DictVectorizer对特征向量化
'''
from sklearn.feature_extraction import DictVectorizer
dict_vec = DictVectorizer(sparse=False)
X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
print(dict_vec.feature_names_)
X_test = dict_vec.transform(X_test.to_dict(orient='record'))

'''
从决策树sklearn.tree中导入决策树分类器
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
#使用默认的配置初始化决策树分类器
dtc = DecisionTreeClassifier()
print('the accuracy of DecisionTreeClassifier on training set',cross_val_score(dtc,X_train,y_train,cv=5).mean())
dtc.fit(X_train,y_train)
dtc_y_predict = dtc.predict(X_test)
dtc_submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':dtc_y_predict})
print(dtc_submission)
'''
将默认配置的DecisionTreeClassifier对预测数据的结果存储在文件dtc_submission.csv中
'''
dtc_submission.to_csv('dtc_submission.csv',index=False)

'''
**********************************************************************************************************
**********************************************************************************************************
'''

'''
使用梯度提升决策树进行集成模型训练已经预测分析
'''
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
#使用默认的配置初始化决策树分类器
gbc = GradientBoostingClassifier()
print('the accuracy of GradientBoostingClassifier on training set',cross_val_score(gbc,X_train,y_train,cv=5).mean())
gbc.fit(X_train,y_train)
gbc_y_predict = gbc.predict(X_test)
gbc_submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':gbc_y_predict})
#将默认配置的GradientBoostingClassifier对预测数据的结果存储在文件dtc_submission.csv中
dtc_submission.to_csv('gbc_submission.csv',index=False)
'''
**********************************************************************************************************
**********************************************************************************************************
'''
'''
从sklearn.ensemble中导入RandomForestClassifier
'''
from sklearn.ensemble import RandomForestClassifier
'''
使用默认初始化RandomForestClassifier
'''
rfc = RandomForestClassifier()
'''
使用5折交叉验证的方法在训练集上分别对默认配置的RandomForestClassifier以及XGBClassifier进行性能评估，并获得平均分类准确性的得分
'''
from sklearn.cross_validation import cross_val_score
print('the accuracy of RandonForestClassifier on training set',cross_val_score(rfc,X_train,y_train,cv=5).mean())
'''
使用默认配置的RandomForestClassifier进行预操作
'''
rfc.fit(X_train,y_train)
rfc_y_predict = rfc.predict(X_test)
rfc_submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':rfc_y_predict})
print(rfc_submission)
'''
将默认配置的RandomForestClassifier对预测数据的结果存储在文件rfc_submission.csv中
'''
rfc_submission.to_csv('rfc_submission.csv',index=False)
'''
**********************************************************************************************************
**********************************************************************************************************
'''


'''
从流行工具包xgboost导入XGBClassifier用于处理分类预测问题
'''
from xgboost import XGBClassifier
'''
也使用默认初始化XGBClassifier
'''
xgbc = XGBClassifier()
print('the accuracy of XGBClassifier on training set',cross_val_score(xgbc,X_train,y_train,cv=5).mean())
'''
使用默认配置的XGBClassifier进行预测操作
'''
xgbc.fit(X_train,y_train)
xgbc_y_predict = xgbc.predict(X_test)
xgbc_submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':xgbc_y_predict})
xgbc_submission.to_csv('xgbc_submission.csv',index=False)
print(xgbc_submission)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.6+ KB
    None
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 11 columns):
    PassengerId    418 non-null int64
    Pclass         418 non-null int64
    Name           418 non-null object
    Sex            418 non-null object
    Age            332 non-null float64
    SibSp          418 non-null int64
    Parch          418 non-null int64
    Ticket         418 non-null object
    Fare           417 non-null float64
    Cabin          91 non-null object
    Embarked       418 non-null object
    dtypes: float64(2), int64(4), object(5)
    memory usage: 36.0+ KB
    None
    S    644
    C    168
    Q     77
    Name: Embarked, dtype: int64
    S    270
    C    102
    Q     46
    Name: Embarked, dtype: int64
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 7 columns):
    Pclass      891 non-null int64
    Sex         891 non-null object
    Age         891 non-null float64
    Embarked    891 non-null object
    SibSp       891 non-null int64
    Parch       891 non-null int64
    Fare        891 non-null float64
    dtypes: float64(2), int64(3), object(2)
    memory usage: 48.8+ KB
    None
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 7 columns):
    Pclass      418 non-null int64
    Sex         418 non-null object
    Age         418 non-null float64
    Embarked    418 non-null object
    SibSp       418 non-null int64
    Parch       418 non-null int64
    Fare        418 non-null float64
    dtypes: float64(2), int64(3), object(2)
    memory usage: 22.9+ KB
    None
    

    C:\Users\xxz\Anaconda3\lib\site-packages\pandas\core\generic.py:3191: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self._update_inplace(new_data)
    

    ['Age', 'Embarked=C', 'Embarked=Q', 'Embarked=S', 'Fare', 'Parch', 'Pclass', 'Sex=female', 'Sex=male', 'SibSp']
    the accuracy of DecisionTreeClassifier on training set 0.772234108463
         PassengerId  Survived
    0            892         0
    1            893         0
    2            894         1
    3            895         1
    4            896         1
    5            897         0
    6            898         0
    7            899         0
    8            900         1
    9            901         0
    10           902         0
    11           903         0
    12           904         1
    13           905         0
    14           906         1
    15           907         1
    16           908         0
    17           909         1
    18           910         1
    19           911         0
    20           912         1
    21           913         1
    22           914         1
    23           915         0
    24           916         1
    25           917         0
    26           918         1
    27           919         1
    28           920         1
    29           921         0
    ..           ...       ...
    388         1280         0
    389         1281         0
    390         1282         0
    391         1283         1
    392         1284         0
    393         1285         0
    394         1286         0
    395         1287         1
    396         1288         0
    397         1289         1
    398         1290         0
    399         1291         0
    400         1292         1
    401         1293         0
    402         1294         1
    403         1295         1
    404         1296         0
    405         1297         1
    406         1298         0
    407         1299         0
    408         1300         1
    409         1301         1
    410         1302         1
    411         1303         1
    412         1304         0
    413         1305         0
    414         1306         1
    415         1307         0
    416         1308         0
    417         1309         0
    
    [418 rows x 2 columns]
    the accuracy of GradientBoostingClassifier on training set 0.821610036503
    the accuracy of RandonForestClassifier on training set 0.793525929441
         PassengerId  Survived
    0            892         0
    1            893         0
    2            894         0
    3            895         0
    4            896         0
    5            897         0
    6            898         0
    7            899         0
    8            900         1
    9            901         0
    10           902         0
    11           903         0
    12           904         1
    13           905         0
    14           906         1
    15           907         1
    16           908         0
    17           909         1
    18           910         0
    19           911         1
    20           912         1
    21           913         1
    22           914         1
    23           915         0
    24           916         1
    25           917         0
    26           918         1
    27           919         1
    28           920         1
    29           921         0
    ..           ...       ...
    388         1280         1
    389         1281         0
    390         1282         0
    391         1283         1
    392         1284         0
    393         1285         0
    394         1286         0
    395         1287         1
    396         1288         0
    397         1289         1
    398         1290         0
    399         1291         0
    400         1292         1
    401         1293         0
    402         1294         1
    403         1295         0
    404         1296         0
    405         1297         0
    406         1298         0
    407         1299         0
    408         1300         1
    409         1301         0
    410         1302         1
    411         1303         1
    412         1304         0
    413         1305         0
    414         1306         1
    415         1307         0
    416         1308         0
    417         1309         1
    
    [418 rows x 2 columns]
    the accuracy of XGBClassifier on training set 0.818245597983
         PassengerId  Survived
    0            892         0
    1            893         0
    2            894         0
    3            895         0
    4            896         0
    5            897         0
    6            898         1
    7            899         0
    8            900         1
    9            901         0
    10           902         0
    11           903         0
    12           904         1
    13           905         0
    14           906         1
    15           907         1
    16           908         0
    17           909         0
    18           910         1
    19           911         0
    20           912         0
    21           913         1
    22           914         1
    23           915         1
    24           916         1
    25           917         0
    26           918         1
    27           919         0
    28           920         1
    29           921         0
    ..           ...       ...
    388         1280         0
    389         1281         0
    390         1282         0
    391         1283         1
    392         1284         0
    393         1285         0
    394         1286         0
    395         1287         1
    396         1288         0
    397         1289         1
    398         1290         0
    399         1291         0
    400         1292         1
    401         1293         0
    402         1294         1
    403         1295         0
    404         1296         0
    405         1297         0
    406         1298         0
    407         1299         0
    408         1300         1
    409         1301         1
    410         1302         1
    411         1303         1
    412         1304         1
    413         1305         0
    414         1306         1
    415         1307         0
    416         1308         0
    417         1309         0
    
    [418 rows x 2 columns]
    
