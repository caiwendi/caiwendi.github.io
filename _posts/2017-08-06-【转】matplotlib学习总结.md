---
layout:     post                    # 使用的布局（不需要改）
title:      【搬运】Matplotlib           # 标题 
subtitle:   Matplotlib operations #副标题
date:       2017-08-06              # 时间
author:     Brian                      # 作者
header-img: img/Taylor1.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Matplotlib
---


# Python科学计算-Matplotlib

Author：Xie Zhong-zhao  
Date： 2017/8/3  
Running Environment: Python3.5
____

## 1、Introduction to Matplotlib and basic line


```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10,6))

ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(212)
x = [1, 2, 3]
y = [5, 7, 4]

ax1.plot(x,x,'r')
ax1.grid(True, color = 'g')

ax2.plot(x,y,'b')
ax2.grid(True, color = 'g')

ax3.plot(x,y,'y')
ax3.grid(True, color = 'g')

plt.show()
```


![](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/matplotlib%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/output_3_0.png)  



## 2、Legends, Titles, and Labels with Matplotlib


```python
import matplotlib.pyplot as plt

x = [1,2,3]
y = [5,7,4]

x2 = [1,2,3]
y2 = [10,14,12]

fig = plt.figure(figsize = (8,5) ) 

plt.plot(x, y, label='First Line')
plt.plot(x2, y2, label='Second Line')

plt.xlabel('Plot Number')
plt.ylabel('Important var')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.legend(loc = 'upper left')
plt.show()
```


![png](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/matplotlib%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/output_5_0.png)   



## 3、Bar Charts and Histograms with Matplotlib


```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10,6) )

plt.bar([1,3,5,7,9],[5,2,7,8,2], label="Example one")
plt.bar([2,4,6,8,10],[8,6,2,5,6], label="Example two", color='g')

plt.legend()
plt.xlabel('bar number')
plt.ylabel('bar height')

plt.title('Epic Graph\nAnother Line! Whoa')

plt.show()
```


![png](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/matplotlib%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/output_7_0.png)



```python
import matplotlib.pyplot as plt

population_ages = [22,55,62,45,21,22,34,42,42,4,99,102,110,120,121,122,130,111,115,112,80,75,65,54,44,43,42,48]

bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130]

plt.hist(population_ages, bins, histtype='bar', rwidth=0.8)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()
```

    C:\Users\xxz\Anaconda3\lib\site-packages\matplotlib\axes\_axes.py:531: UserWarning: No labelled objects found. Use label='...' kwarg on individual plots.
      warnings.warn("No labelled objects found. "


![png](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/matplotlib%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/output_8_1.png)


## 4、Scatter Plots with Matplotlib


```python
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(9, 6))

n = 1024

#rand and randn  rand is uniform distribution  randn means  Gussian distribution
X = np.random.randn(1, n)
Y = np.random.randn(1, n)

T = np.arctan2(Y, X)

#alpha is transparent ability of dots
#c reprensents color of dots
plt.scatter(X, Y, s=50, c=T, alpha=.4, marker='o')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()
```

    C:\Users\xxz\Anaconda3\lib\site-packages\matplotlib\axes\_axes.py:531: UserWarning: No labelled objects found. Use label='...' kwarg on individual plots.
      warnings.warn("No labelled objects found. "


![png](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/matplotlib%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/output_10_1.png)


## 5、Stack Plots with Matplotlib


```python
import matplotlib.pyplot as plt

days = [1,2,3,4,5]

sleeping = [7,8,6,11,7]
eating =   [2,3,4,3,2]
working =  [7,8,7,2,2]
playing =  [8,5,7,8,13]

plt.stackplot(days, sleeping,eating,working,playing, colors=['m','c','r','k'])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.show()
```


![png](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/matplotlib%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/output_12_0.png)



```python
import matplotlib.pyplot as plt

days = [1,2,3,4,5]

sleeping = [7,8,6,11,7]
eating =   [2,3,4,3,2]
working =  [7,8,7,2,2]
playing =  [8,5,7,8,13]


plt.plot([],[],color='m', label='Sleeping', linewidth=5)
plt.plot([],[],color='c', label='Eating', linewidth=5)
plt.plot([],[],color='r', label='Working', linewidth=5)
plt.plot([],[],color='k', label='Playing', linewidth=5)

plt.stackplot(days, sleeping,eating,working,playing, colors=['m','c','r','k'])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()
```


![png](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/matplotlib%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/output_13_0.png)


## 6、Pie Charts with Matplotlib


```python
import matplotlib.pyplot as plt

slices = [7,2,2,13]
activities = ['sleeping','eating','working','playing']
cols = ['c','m','r','b']

plt.pie(slices,
        labels=activities,
        colors=cols,
        startangle=90,
        shadow= True,
        explode=(0,0.1,0,0),
        autopct='%1.1f%%')

plt.title('Interesting Graph\nCheck it out')
plt.show()
```


![png](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/matplotlib%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/output_15_0.png)


## 7、Loading Data from Files for Matplotlib


```python
import matplotlib.pyplot as plt
import csv

fig = plt.figure(figsize = (10,6)) 

x = []
y = []

with open('example.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(int(row[0]))
        print(x)
        y.append(int(row[1]))
        print(y)
        
plt.plot(x,y, label='Loaded from file!')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize = (10,6))

x, y = np.loadtxt('example.txt', delimiter=',', unpack=True)
plt.plot(x,y, label='Loaded from file!')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()
```

    [1]
    [5]
    [1, 2]
    [5, 2]
    [1, 2, 3]
    [5, 2, 4]
    [1, 2, 3, 4]
    [5, 2, 4, 8]
    [1, 2, 3, 4, 5]
    [5, 2, 4, 8, 3]
    [1, 2, 3, 4, 5, 6]
    [5, 2, 4, 8, 3, 6]
    [1, 2, 3, 4, 5, 6, 7]
    [5, 2, 4, 8, 3, 6, 8]
    [1, 2, 3, 4, 5, 6, 7, 8]
    [5, 2, 4, 8, 3, 6, 8, 3]
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    [5, 2, 4, 8, 3, 6, 8, 3, 2]
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    [5, 2, 4, 8, 3, 6, 8, 3, 2, 6]


![png](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/matplotlib%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/output_17_1.png)



![png](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/matplotlib%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/output_17_2.png)


## 8、3D graphs with Matplotlib


```python
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style

fig = plt.figure(figsize = (10,6))

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

x = [1,2,3,4,5,6,7,8,9,10]
y = [5,6,7,8,2,5,6,3,7,2]
z = [1,2,6,3,2,7,3,3,7,2]

ax1.plot_wireframe(x,y,z)

ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')

plt.show()
```


    <matplotlib.figure.Figure at 0x1ca32b7d9e8>


![png](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/matplotlib%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/output_19_1.png)


## 9、3D Scatter Plot with Matplotlib


```python
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style

fig = plt.figure(figsize = (10,6))

style.use('ggplot')

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

x = [1,2,3,4,5,6,7,8,9,10]
y = [5,6,7,8,2,5,6,3,7,2]
z = [1,2,6,3,2,7,3,3,7,2]

x2 = [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]
y2 = [-5,-6,-7,-8,-2,-5,-6,-3,-7,-2]
z2 = [1,2,6,3,2,7,3,3,7,2]

ax1.scatter(x, y, z, c='g', marker='o')
ax1.scatter(x2, y2, z2, c ='r', marker='o')

ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')

plt.show()
```


    <matplotlib.figure.Figure at 0x1ca30e80f60>


![png](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/matplotlib%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/output_21_1.png)


## 10、3D Bar Chart with Matplotlib


```python
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use('ggplot')

fig = plt.figure(figsize = (10,6))

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

x3 = [1,2,3,4,5,6,7,8,9,10]
y3 = [5,6,7,8,2,5,6,3,7,2]
z3 = np.zeros(10)

dx = np.ones(10)
dy = np.ones(10)
dz = [1,2,3,4,5,6,7,8,9,10]

ax1.bar3d(x3, y3, z3, dx, dy, dz)


ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')

plt.show()
```


    <matplotlib.figure.Figure at 0x1ca31048e48>


![png](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/matplotlib%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/output_23_1.png)


## 11、Conclusion with Matplotlib


```python
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
#style.use('fivethirtyeight')
style.use('bmh')

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

x, y, z = axes3d.get_test_data()

print(axes3d.__file__)
ax1.plot_wireframe(x,y,z, rstride = 5, cstride = 5)

ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')

plt.show()
fig.savefig("conclusion.png", dpi=300)  #save graph
```

    C:\Users\xxz\Anaconda3\lib\site-packages\mpl_toolkits\mplot3d\axes3d.py


![png](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/matplotlib%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/output_25_1.png)


## 12、 Text Annotation


```python
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot(111)

xx = np.arange(0,15,0.1)
ax.plot(xx, xx**2, xx, xx**3)

ax.text(13, 300, r"$y=x^2$", fontsize=20, color="blue");
ax.text(13, 3000, r"$y=x^3$", fontsize=20, color="green");

plt.show()
```


![png](https://raw.githubusercontent.com/xiezhongzhao/blog/gh-pages/_posts/matplotlib%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/output_27_0.png)


## END
