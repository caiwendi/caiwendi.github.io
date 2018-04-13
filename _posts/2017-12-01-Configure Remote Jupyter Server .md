---
layout:     post                    # 使用的布局（不需要改）
title:      Configure the Jupyter server   # 标题 
subtitle:   Jupyter Notebook #副标题
date:       2017-12-20             # 时间
author:     Brian                      # 作者
header-img: img/post-bg-os-metro.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:       - Tool                  #标签
---

## Configure the Jupyter server

### 1. Create an SSL certificate.

  `$ cd`
  `$ mkdir ssl`
  `$ cd ssl`
  `$ sudo openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout cert.key" -out "cert.pem" -batch`

### 2. Create a password. You use this password to log in to the Jupyter notebook server from your client so you can access notebooks on the server.
  (a) Open the iPython terminal.

  `$ ipython`
  At the iPython prompt, run the `passwd()` command to set the password. 

  `iPythonPrompt> from IPython.lib import passwd `
  `iPythonPrompt> passwd()`
  You get the password hash (For example, `sha1:examplefc216:3a35a98ed...`)
  (b) Record the password hash.
  (c) Exit the iPython terminal.
  `$ exit`

### 3. Create a Jupyter configuration file. 
`$ jupyter notebook --generate-config `
The command creates a configuration file (`jupyter_notebook_config.py`) in the `~/.jupyter` directory. 

### 4. Update the configuration file to store your password and SSL certificate information. 
  (a) Open the .config file.


  `vi ~/.jupyter/jupyter_notebook_config.py`


  (b) Paste the following text at the end of the file. You will need to provide your password hash. 

  ```
  c = get_config()  # Get the config object.
  c.NotebookApp.certfile = u'/home/ubuntu/ssl/cert.pem' # path to the certificate we generated
  c.NotebookApp.keyfile = u'/home/ubuntu/ssl/cert.key' # path to the certificate key we generated
  c.IPKernelApp.pylab = 'inline'  # in-line figure when using Matplotlib
  c.NotebookApp.ip = '*'  # Serve notebooks locally.
  c.NotebookApp.open_browser = False  # Do not open a browser window by default when using notebooks.
  c.NotebookApp.password = 'sha1:fc216:3a35a98ed980b9...
  ```

  This completes Jupyter server configuration.

### 5. You can  access the Jupyter notebook server at  IP address  in Browser.

If you log in to the jupyter notebook server, just input the password you set above .        
