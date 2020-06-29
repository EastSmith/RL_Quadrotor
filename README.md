# 用强化学习DDPG算法控制四轴飞行器（velocity_control）

## 仿真环境：RLSchool

此奖励的理论极限为0，即飞行速度与控制要求的速度完全一致时为0，但不可避免的它总是一个负数。

## "velocity_control" task

黄色箭头是预期的速度矢量；橙色箭头是实际的速度矢量。


 
## 安装

`pip install paddlepaddle==1.6.3          #如果使用gpu可选安装paddlepaddle-gpu==1.6.3.post97`\
`pip install parl==1.3.1`\
`pip install rlschool==0.3.1`

## 如何开始

 修改train.py文件，然后 python train.py  开始测试或者训练你自己的模型。
 

## tips：

* ActorModel的隐藏层和输出层进行`param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1)`初始化操作后，似乎可以在更少的训练轮次后收敛。

* 参考Homework_lesson5_ddpg里面的答案ActorModel的代码是含2个隐藏层，再加上一层16个维度输入层，一层4个维度的输出层，故ActorModel的网络结构为 16x64x64x4，
 其参数个数为16x64+64x64+64x4 = 5376
 
* 此次上传到github上的ActorModel的代码是含1个隐藏层，再加上一层16个维度输入层，一层5个维度的输出层，一层4个维度的转化层，故ActorModel的网络结构相当于 16x100x5x4，
 其参数个数为16x100+100x5+5x4= 2120
 
* 模型拥有更少的参数，意味则可以在实际部署的时候硬件资源消耗更少。

* 深度神经网络通常是一个黑箱，但最后一个5维神经元转4维输出的操作，就相当于一个不参与反向传播的单独的一层神经元（可以通过人为设置参数）；
  5维神经元的输出，其中一个维度代表基本值，其他4个维度代表浮动值。

* 更少的参数，搜索空间更小，运气好的话应该可以快速收敛。
      
       
