---
layout:     post                    # 使用的布局（不需要改）
title:      Ubuntu下设置Shadowsocks的非全局代理（PAC自动代理） # 标题 
subtitle:   Shadowsocks, PAC
date:       2017-12-18              # 时间
author:     Brian                      # 作者
header-img: img/post-bg-universe.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:       Tool                       #标签
---


## 1、Ubuntu安装shadowsocks

```
sudo add-apt-repository ppa:hzwhuang/ss-qt5
sudo apt-get update
sudo apt-get install shadowsocks-qt5
```
安装好shadowsocks-qt5之后，[配置搭建好的服务器的相关信息](http://xiezhongzhao.top/2017/12/20/GoogleCloudPlatform%E6%90%AD%E5%BB%BAvps/), 如图：![](http://ww1.sinaimg.cn/large/006zLtEmgy1fq6nvddbc8j30pz0evq5k.jpg)

## 2、Ubuntu下设置Shadowsocks的非全局代理（PAC自动代理）

ubuntu下shadowsockes设置完后,chrome必须通过插件才能进行翻墙，而firefox下木有合适的插件,有时候在终端安装依赖时也需要翻墙，会造成依赖无法下载问题,以下方法可以解决这些问题,通过以下方法，可以实现和windows下的shadowsockes的功能

### 2.1、安装pip（如果系统未安装,需安装）

```
sudo apt-get install python-pip python-dev build-essential 
sudo pip install --upgrade pip 
sudo pip install --upgrade virtualenv
```

### 2.2、安装Genepac

它可以自动生成的PAC文件

```
sudo pip install genpac
```

### 2.3、新建shadowsocks文件夹存放pac文件

```
mkdir ~/shadowsocks
cd shadowsocks
```

### 2.4、生成pac文件

```
genpac --proxy="SOCKS5 127.0.0.1:1080" --gfwlist-proxy="SOCKS5 127.0.0.1:1080" -o autoproxy.pac --gfwlist-url="https://raw.githubusercontent.com/gfwlist/gfwlist/master/gfwlist.txt"
```

### 2.5、设置系统网络代理

url为shadowsocks里的pac文件

![img](https://leanote.com/api/file/getImage?fileId=587305cfab6441236e01829c)


### 2.6、自定义网站代理

如果打开国外网站慢，可以将网址手动添加到pac文件中,如果添加到末尾,记得要加上逗号;以`atom.io`为列 


![img](https://leanote.com/api/file/getImage?fileId=58730a88ab6441236e01831d)

### 2.7、打开浏览器上网

![](http://ww1.sinaimg.cn/large/006zLtEmgy1fq6ob7l9drj311a0oo11g.jpg)


