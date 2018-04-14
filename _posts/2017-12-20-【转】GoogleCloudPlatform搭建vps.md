---
layout:     post                    # 使用的布局（不需要改）
title:      【转】Google Cloud Platform搭建VPS   # 标题 
subtitle:   VPS, Google Cloud Platform  #副标题
date:       2017-12-20              # 时间
author:     Brian                      # 作者
header-img: img/post-bg-2015.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:       - 转载             #标签
---

## GoogleCloudPlatform搭建vps

1：前提条件，拥有一张信用卡，Visa或者MasterCard或者JCB.

2：有一个可以临时代理的环境.

3：工具软件，从以下网址下载对应平台的shadowsocks.

[http://shadowsocks.org/en/download/clients.html](https://shadowsocks.org/en/download/clients.html)

4：拥有谷歌账户，这里注册谷歌账户的步骤就不在赘述了。谷歌的全家桶都用这一个账户。另外推荐使用chrome浏览器因为谷歌云是英文的，用这个浏览器可以翻译网页

![](http://ww1.sinaimg.cn/large/006zLtEmgy1fq6mfqzpb7j30ib09saaa.jpg)

## 一、创建vps

1、 <https://cloud.google.com/>打开网址进行注册

2、开始创建VM实例

![](http://ww1.sinaimg.cn/large/006zLtEmgy1fq6mmixfdxj30bw05uaa8.jpg)

3、进入后点击创建实例

![](http://ww1.sinaimg.cn/large/006zLtEmgy1fq6mnwqhoxj30dq0p075p.jpg)

4、创建成功就会出现一个VM实例的列表，如图

![](http://ww1.sinaimg.cn/large/006zLtEmgy1fq6mq5wy2tj30fe06dmx6.jpg)

5、验证速度

拷贝这个ip地址在浏览器输入<https://www.ipip.net/traceroute.php>. 验证速度均在100内，说明速度没问题

## 二、配置服务器

1、连接ssh服务器，输入的命令如下

```
Step 1 : sudo su
Step 2 : wget --no-check-certificate https://raw.githubusercontent.com/ligl0702/ligl0702.github.io/master/run.sh && chmod +x run.sh && ./run.sh
```

2、执行上述2个步骤，得到如下结果：

![](http://ww1.sinaimg.cn/large/006zLtEmgy1fq6mucorzjj30e404kjrc.jpg)

3、设置密码和端口

![](http://ww1.sinaimg.cn/large/006zLtEmgy1fq6mv1hysyj30dj0da0sy.jpg)

4、一路回车后，大概需要10分钟左右的配置时间，等待就好。

![](http://ww1.sinaimg.cn/large/006zLtEmgy1fq6mw7ik8uj30c705kglm.jpg)

5、恭喜你 看到这个画面证明你成功了。谷歌云代码部分已经配置完毕。

6、配置防火墙规则：

网络-----VPC网络----防火墙规则

![](http://ww1.sinaimg.cn/large/006zLtEmgy1fq6mxwekuoj30bz03sjrc.jpg)

把端口号设置为之前我们自己规定的789

![](http://ww1.sinaimg.cn/large/006zLtEmgy1fq6myoqkr0j30fe02bq2u.jpg)

如图，这里我之前设置了456，点进去都可以修改的。你之前设置的是多少就写多少

注意[default-allow-http](https://l.facebook.com/l.php?u=https%3A%2F%2Fconsole.cloud.google.com%2Fnetworking%2Ffirewalls%2Fdetails%2Fdefault-allow-http%3Fproject%3Dspeech-172101&h=ATO__1z2ydkNVaflhGXcygV4Om3ViGhVpCDIy4AE9N-WNqQquGFXMffy7EPvO5HDn4IMAqrvuEcTw5IuKHAd8jRDqIbklODCoUSYA8cQy_YHOA)和[default-allow-http](https://console.cloud.google.com/networking/firewalls/details/default-allow-http?project=speech-172101)s 都需要点进去分别设置的 不要漏掉。

![](http://ww1.sinaimg.cn/large/006zLtEmgy1fq6mzu7qwuj30c705kglm.jpg)

点进去修改这里，输入tcp:789; udp:789，点击保存。

![](http://ww1.sinaimg.cn/large/006zLtEmgy1fq6n0jwrgbj305x03zmx2.jpg)

OK，谷歌云vps搭建完毕。

## 三、配置客户端

1、Shadowsocks安卓客户端

https://github.com/shadowsocks/shadowsocks-android

2、Wingy苹果客户端

3、Windows电脑客户端

https://github.com/shadowsocks/shadowsocks-windows/releases/download/4.0.9/Shadowsocks-4.0.9.zip















