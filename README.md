标准答案的ActorModel的代码：
class ActorModel(parl.Model):
    def __init__(self, act_dim):
        hidden_dim_1, hidden_dim_2 = 64, 64
        self.fc1 = layers.fc(size=hidden_dim_1, act='tanh')
        self.fc2 = layers.fc(size=hidden_dim_2, act='tanh')
        self.fc3 = layers.fc(size=act_dim, act='tanh')

    def policy(self, obs):
        x = self.fc1(obs)
        x = self.fc2(x)
        return self.fc3(x)
 可见是含2个隐藏层，再加上一层16个维度输入层，一层4个维度的输出层，故ActorModel的网络结构为 16*64*64*4，
 其参数个数为16*64+64*64+64*4 = 5376
 
 此次上传到github上的ActorModel的代码：
 class ActorModel(parl.Model):
    def __init__(self, act_dim):
       
        hid_size = 100
        
        self.fc1 = layers.fc(size=hid_size, act='relu',param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
        self.fc2 = layers.fc(size=out  , act='tanh',param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
        

    def policy(self, obs):
        hid = self.fc1(obs)
        logits = self.fc2(hid)
        
        return logits
可见是含1个隐藏层，再加上一层16个维度输入层，一层5个维度的输出层，一层4个维度的转化层，故ActorModel的网络结构为 16*100*5*4，
 其参数个数为16*100+100*5+5*4= 2120
      
       
