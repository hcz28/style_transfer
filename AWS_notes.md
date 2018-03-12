1. 创建 spot instance
    1. 申请1个spot instance(比如p2.xlarge), 创建一个满足环境最小(土豪可以无视)需求的EBS根卷, 设置为实例终止后不删除， 即取消那个勾, 比如10G, 选取一个amazon ami, 比如ubuntu 16.04, 启动
    2. 连接成功后, 配置你自己的环境，比如安装软件和设置参数, 比如cuda, anaconda,tensorflow...之类, 测试无误后终止实例(注意，对spot instance进行poweroff,  被视为终止行为)
    3. 对实例留下来的EBS卷做snapshot
    4. 基于该snapshot创建AMI
    5. 再次申请spot instance时，使用自己创建的这个ami, 就不用弄装环境了, 如果有变动，你可以基于该ami启动的EBS，修改后，重新snapshot来创建AMI
2. 数据保留
    1. 创建一个用于保存数据的EBS卷，attach到实例，并且
        ```
        lsblk
        sudo mkfs -t ext4 device_name
        sudo mkdir mount_point
        sudo mount device_name mount_point
        ```
    2. 跑程序，download数据，或者保存checkpoint
    3. 停止instance
    4. 注意该EBS卷会保留
    5. 再次创建或申请instance，attach上面的数据EBS卷到实例，lsblk, mount就可以了, 注意不能mkfs了
    6. 使用之前的数据
3. 使用ssh链接以及数据传输
    ```
    ssh -i /path/my-key-pair.pem user_name@public_dns_name
    scp -i /path/my-key-pair.pem /path/SampleFile.txt user_name@public_dns_name:destination_path
    ```
4. 环境配置
    1. Anaconda
        ```
        wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
        bash Anaconda-latest-Linux-x86_64.sh
        export PATH=~/anaconda3/bin:$PATH
    2. CUDA 9.0
        
        1. install make gcc first
        
        2. check GPU
        ```
        lspci | grep -i nvidia
        ```
        3. install
        ```
        wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
        sudo sh cuda_9.0.176_384.81_linux.run --tmpdir=<path>
        ```
        4. environment setup
        ```
        export PATH=/usr/local/cuda-9.1/bin${PATH:+:${PATH}} 
        export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
        ```
    3. cuDNN

        first download the installation file into local disk and then use scp to transfer
        data
        ```
        sudo dpkg -i libcudnn7_7.0.3.11-1+cuda9.0_amd64.deb
        export CUDA_HOEM=/usr/local/cuda
        ```
    4. Install libcupti-dev library
        ```
        sudo apt-get install cuda-command-line-tools
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
        ```
    5. Tensorflow
        ```
        sudo apt-get install python3-pip python3-dev
        pip3 install tensorflow-gpu
        ```
    6. Validate
        ```python
        import tensorflow as tf
        hello = tf.constant('Hello, TensorFlow!')
        with tf.device('/gpu:0'), tf.Session() as sess:
            print(sess.run(hello))
        ```
    7. Download code
        ```
        git clone https://github.com/hcz28/style_transfer.git
        ```
    8. Others
        1. security group should enable ssh in the inbound rules

References
- [请问一个使用Amazon EC2 P2竞价实例做计算数据保存的问题? - Ray Wang的回答 - 知乎](https://www.zhihu.com/question/62458408/answer/199345173)
- [使用ssh链接以及数据传输](https://docs.aws.amazon.com/zh_cn/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html)
- [Amason EBS Guide](https://docs.aws.amazon.com/zh_cn/AWSEC2/latest/UserGuide/ebs-using-volumes.html)
- [CUDA Installation
  Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4VZnqTJ2A)
- [cuDNN Installation
  Guide](http://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.5/prod/Doc/cuDNN-Installation-Guide.pdf?uUC-ZDHRpDmlrNq_7GTYkv87I6DMyvaoxYPW7GmQs3Hd8I738fu2u9NDNsXZDu21SglpQCxd4Y4IBhHp5iuXFsD43i54dybJchanofnRidbVVmk8v8ujlkEFYhiARRkgqzBDUsQklP3aE2UmIOrDKjRu6qbUP8q5Fh6HuZPr3wQiiX8XBXI353R3emrZxiT9Mg)
- [Tensorflow Installation Guide](https://www.tensorflow.org/install/install_linux)
