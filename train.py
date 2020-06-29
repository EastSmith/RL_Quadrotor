#!/usr/bin/env python
# coding: utf-8


# # Step1 安装依赖





#pip uninstall -y parl  # 说明：AIStudio预装的parl版本太老，容易跟其他库产生兼容性冲突，建议先卸载
#pip uninstall -y pandas scikit-learn # 提示：在AIStudio中卸载这两个库再import parl可避免warning提示，不卸载也不影响parl的使用

#'pip install paddlepaddle==1.6.3  -i https://mirror.baidu.com/pypi/simple           #可选安装paddlepaddle-gpu==1.6.3.post97
#pip install parl==1.3.1
#pip install rlschool==0.3.1

# 说明：安装日志中出现两条红色的关于 paddlehub 和 visualdl 的 ERROR 与parl无关，可以忽略，不影响使用







# # Step2 导入依赖




import os
import numpy as np

import parl
from parl import layers
from paddle import fluid
from parl.utils import logger
from parl.utils import action_mapping # 将神经网络输出映射到对应的 实际动作取值范围 内
from parl.utils import ReplayMemory # 经验回放

from rlschool import make_env  # 使用 RLSchool 创建飞行器环境


# # Step3 设置超参数

# In[3]:


######################################################################
######################################################################
#
# 1. 请设定 learning rate，尝试增减查看效果
#
######################################################################
######################################################################
ACTOR_LR =5* 0.0002   # Actor网络更新的 learning rate                开始直接5倍学习率，后期模型相对稳定后再调低
CRITIC_LR =5* 0.001   # Critic网络更新的 learning rate                开始直接5倍学习率，后期模型相对稳定后再调低

GAMMA = 0.99        # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.001         # target_model 跟 model 同步参数 的 软更新参数
MEMORY_SIZE = 36e4   # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1e4      # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn            
REWARD_SCALE = 0.01       # reward 的缩放因子
BATCH_SIZE = 2*256          # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来                 2倍的batch_size 
TRAIN_TOTAL_STEPS = 36e4   # 总训练步数
TEST_EVERY_STEPS = 1e4    # 每个N步评估一下算法效果，每次评估5个episode求平均reward
GM  = 0.2                 # 变电压的浮动参数


# # Step4 搭建Model、Algorithm、Agent架构
# * `Agent`把产生的数据传给`algorithm`，`algorithm`根据`model`的模型结构计算出`Loss`，使用`SGD`或者其他优化器不断的优化，`PARL`这种架构可以很方便的应用在各类深度强化学习问题中。
# 
# ## （1）Model
# * 分别搭建`Actor`、`Critic`的`Model`结构，构建`QuadrotorModel`。

# In[4]:


class ActorModel(parl.Model):
    def __init__(self, act_dim):
        ######################################################################
        ######################################################################
        #
        # 2. 请配置model结构
        #
        ######################################################################
        ######################################################################
        hid_size = 100
        hd2 = act_dim

        self.fc1 = layers.fc(size=hid_size, act='relu',param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
        self.fc2 = layers.fc(size=hd2  , act='tanh',param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))

    def policy(self, obs):
        ######################################################################
        ######################################################################
        #
        # 3. 请组装policy网络
        #
        ######################################################################
        ######################################################################
        hid = self.fc1(obs)
        logits = self.fc2(hid)
        
        return logits


# In[5]:


class CriticModel(parl.Model):
    def __init__(self):
        ######################################################################
        ######################################################################
        #
        # 4. 请配置model结构
        #
        ######################################################################               
        ######################################################################                
        hid_size = 100

        self.fc1 = layers.fc(size=hid_size, act='relu',param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
        self.fc2 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        # 输入 state, action, 输出对应的Q(s,a)

        ######################################################################
        ######################################################################
        #
        # 5. 请组装Q网络
        #
        ######################################################################
        ######################################################################
        concat = layers.concat([obs, act], axis=1)
        hid = self.fc1(concat)
        Q = self.fc2(hid)
        Q = layers.squeeze(Q, axes=[1])
        return Q


# In[36]:


class QuadrotorModel(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()


# ## （2）Algorithm
# * 可以采用下面的方式从`parl`库中快速引入`DDPG`算法，无需自己重新写算法

# In[38]:


from parl.algorithms import DDPG


# ## （3）Agent

# In[39]:


class QuadrotorAgent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):                       
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(QuadrotorAgent, self).__init__(algorithm)

        # 注意，在最开始的时候，先完全同步target_model和model的参数
        self.alg.sync_target(decay=0)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.pred_act = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(
                name='act', shape=[self.act_dim], dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            _, self.critic_cost = self.alg.learn(obs, act, reward, next_obs,
                                                 terminal)

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act = self.fluid_executor.run(
            self.pred_program, feed={'obs': obs},
            fetch_list=[self.pred_act])[0]
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal
        }
        critic_cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.critic_cost])[0]
        self.alg.sync_target()
        return critic_cost


# 
# # Step4 Training && Test（训练&&测试）

# In[40]:


def run_episode(env, agent, rpm):
    obs = env.reset()
    total_reward, steps = 0, 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action0 = agent.predict(batch_obs.astype('float32'))            
        #action  =    action.mean(axis=1)                #加的一行代码，使输出一致，效果你懂的，值得一试O(∩_∩)O哈哈~
        
        
        action = np.squeeze(action0)                         
        mean_a= action[4]                              #加的三行代码，还原输出，目的使输出稳定，相当于加了先验，4轴飞行器的电压的保持相对的稳定，更有利于收敛。       
        action = action[0:4]                           #其中一个维度是作为基本值，其他4个维度作为浮动值。
        action = GM*action + mean_a                   #此处我取了一个GM = 0.15的系数，为什么有效？可能神经网络训练的时候输出的值是差不多的，强行加一个系数相当于人为的先验。
                                             

        # 给输出动作增加探索扰动，输出限制在 [-1.0, 1.0] 范围内
        action = np.clip(np.random.normal(action, 1.0), -1.0, 1.0)            ## action = np.clip(action, -1.0, 1.0)   ，变成这个样子就是直接用网络输出不加扰动存入经验池，
         
                                                                              ##大家也可以加大或者降低normal值，来增加或者减小探索的幅度
        # 动作映射到对应的 实际动作取值范围 内, action_mapping是从parl.utils那里import进来的函数            
        action = action_mapping(action, env.action_space.low[0],
                                env.action_space.high[0])
       
        ##测试print(action)                          #之前测试action用的            

        next_obs, reward, done, info = env.step(action)
        rpm.append(obs, action0, REWARD_SCALE * reward, next_obs, done)       #注意变量名 action0，rpm需要原始输出，而env需要处理后的输出

        if rpm.size() > MEMORY_WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs,                     batch_terminal = rpm.sample_batch(BATCH_SIZE)
            critic_cost = agent.learn(batch_obs, batch_action, batch_reward,
                                      batch_next_obs, batch_terminal)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward, steps

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward, steps = 0, 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype('float32'))
            

            action = np.squeeze(action)                      
            mean_a= action[4]                                     #加的代码，还原输出，目的使输出稳定，原因同上。
            action = action[0:4]
            action = GM*action + mean_a                           #此处我取了一个GM = 0.2的系数,在全局变量里面设置，用于变电压浮动的控制

            action = np.clip(action, -1.0, 1.0)         #加的一行代码，防止报错
            action = action_mapping(action, env.action_space.low[0], 
                                    env.action_space.high[0])

            next_obs, reward, done, info = env.step(action)

            obs = next_obs
            total_reward += reward
            steps += 1

            if done:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)


# # Step 5 创建环境和Agent，创建经验池，启动训练，定期保存模型




# 创建飞行器环境
env = make_env("Quadrotor", task="velocity_control", seed=0)              
env.reset()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]  +1            #输出加一个维度，评估时再还原


# 根据parl框架构建agent
######################################################################
######################################################################
#
# 6. 请构建agent:  QuadrotorModel, DDPG, QuadrotorAgent三者嵌套
#
######################################################################
######################################################################
model = QuadrotorModel(act_dim)
algorithm = DDPG(
    model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
agent = QuadrotorAgent(algorithm, obs_dim, act_dim)


# parl库也为DDPG算法内置了ReplayMemory，可直接从 parl.utils 引入使用
rpm = ReplayMemory(int(MEMORY_SIZE), obs_dim, act_dim)


# In[ ]:


# 启动训练
test_flag = 0
total_steps = 0
while total_steps < TRAIN_TOTAL_STEPS:
    train_reward, steps = run_episode(env, agent, rpm)
    total_steps += steps
    #logger.info('Steps: {} Reward: {}'.format(total_steps, train_reward)) # 打印训练reward

    if total_steps // TEST_EVERY_STEPS >= test_flag: # 每隔一定step数，评估一次模型
        while total_steps // TEST_EVERY_STEPS >= test_flag:
            test_flag += 1
 
        evaluate_reward = evaluate(env, agent)
        logger.info('Steps {}, Test reward: {}'.format(
            total_steps, evaluate_reward)) # 打印评估的reward

        # 每评估一次，就保存一次模型，以训练的step数命名
        ckpt = 'model_dir0/s2[{}]_{}.ckpt'.format(int(evaluate_reward),total_steps)                   #想存不同版本的ckpt文件，可以在此处改目录，一个版本一个目录肯定不会混。
        agent.save(ckpt)


# # 验收测评

#  **我的理解是既然是速度控制，那么越接近规定的速度越好，最好的情况就是与规定的速度相同也就是0误差，这可能就是reward要定为很小的负值的意义**






######################################################################
######################################################################
#
# 7. 请选择你训练的最好的一次模型文件做评估
#
######################################################################
######################################################################
ckpt = 'model_dir0/s2[-20]_980000.ckpt'  # 请设置ckpt为你训练中效果最好的一次评估保存的模型文件名称

agent.restore(ckpt)
evaluate_reward = evaluate(env, agent)
logger.info('Evaluate reward: {}'.format(evaluate_reward)) # 打印评估的reward


# **一个有趣的地方：**
# * 训练的时候action的GM值取的全局变量，GM= 0.2，但测试的时候我改写了评估程序，令 action = gm * action +(1-gm)* mean_a 。
# * 这个操作只会对测试产生影响，而不会对rpm产生影响，因为存入rpm的是神经网络的原始输出值。
# * 评估时每循环一次都改变了gm的值，gm最小取0，最大取1。取gm = 0时，action 失效;取 gm = 1时,mean_a 失效。
# * 我循环测试了21次，每次gm 值增加0.05 ,即使是gm为0 或者 gm 为1的时候，飞行器都能得到高的reward ，这说明无论是action (具有4个输出维度)，还是 mean_a(只有1个输出维度) 都能独立完成任务。
# * 但经过测试发现，gm值越高时，越容易收到不理想的reward。在gm值设置为0.65的时候，已经显露出不稳定的迹象了，明显reward的方差变大了。
# * 这可能就是设置基本值mean_a和浮动值action ，并按一定比例叠加送入到env之后能够提高收敛速度的原因，确实提高了输出的相对稳定性。




ckpt = 'model_dir0/s2[-20]_980000.ckpt'  # 请设置ckpt为你训练中效果最好的一次评估保存的模型文件名称

agent.restore(ckpt)
def evaluate1(env, agent ,gm):
    
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward, steps = 0, 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype('float32'))
            

            action = np.squeeze(action)                      
            mean_a= action[4]                                     #加的代码，还原输出，目的使输出稳定，原因同上。
            action = action[0:4]
            action = gm*action +(1-gm) * mean_a                           #注意此处的gm，用于变电压浮动的控制
            

            action = np.clip(action, -1.0, 1.0)         #加的一行代码，防止报错
            action = action_mapping(action, env.action_space.low[0], 
                                    env.action_space.high[0])

            next_obs, reward, done, info = env.step(action)

            obs = next_obs
            total_reward += reward
            steps += 1

            if done:
                break
        eval_reward.append(total_reward)
        print("一次评估完成，此时的gm值",gm,"此次的total_reward",total_reward)
    return np.mean(eval_reward)
for gm in range(21):
    gm   = 0.05*float(gm)
    print("此轮的gm值:",gm)
    evaluate_reward = evaluate1(env, agent,gm)
    logger.info('Evaluate reward: {}'.format(evaluate_reward)) # 打印评估的reward

