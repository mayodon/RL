import gym
import numpy as np
import random
from collections import deque
import matplotlib
matplotlib.use('TKAgg') 

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
   
class DQN(nn.Module):
  
    def __init__(self, state_size, action_size, hidden_size1=64, hidden_size2=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, action_size)
        
    def forward(self, x):
       #print(f"输入形状: {x.shape}")
        
        x = self.fc1(x)
        #print(f"第一层线性变换后形状: {x.shape}")
        
        x = torch.relu(x)
        #print(f"第一层ReLU激活后形状: {x.shape}")
        
        x = self.fc2(x)
        #print(f"第二层线性变换后形状: {x.shape}")
        
        x = torch.relu(x)
        #print(f"第二层ReLU激活后形状: {x.shape}")
        
        x = self.fc3(x)
        #print(f"输出层形状: {x.shape}")
        #breakpoint()
        return x
        

class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        
        self.gamma = 0.9304013121805538  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.044539208446220474
        self.epsilon_decay = 0.993093099783833
        self.learning_rate = 0.00017153121233810543
        self.batch_size = 128
        
        # 经验回放缓冲区 - 增大缓冲区大小
        self.memory = deque(maxlen=10000)
        
        # 创建主网络和目标网络 - 使用优化后的网络结构
        self.model = DQN(state_size, action_size, hidden_size1=64, hidden_size2=128)
        self.target_model = DQN(state_size, action_size, hidden_size1=64, hidden_size2=128)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # 初始化目标网络权重
        self.update_target_model()
        
    def update_target_model(self):
    
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
     
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
    
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item( )
    
    def replay(self):
    
        if len(self.memory) < self.batch_size:
            return
        # 从记忆中随机采样批次
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([i[0] for i in minibatch])
        actions = torch.LongTensor([[i[1]] for i in minibatch])
        rewards = torch.FloatTensor([[i[2]] for i in minibatch])
        next_states = torch.FloatTensor([i[3] for i in minibatch])
        dones = torch.FloatTensor([[i[4]] for i in minibatch])
        # Q(s,a)
        current_q = self.model(states).gather(1, actions)# 输入状态，输出动作价值
        # Q(s',a')
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
        # 目标Q值
        
        target_q = rewards + (self.gamma * next_q * (1 - dones))
        # 计算损失
        loss = self.criterion(current_q, target_q)
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dqn(episodes=500, render_interval=100):
 
    # 创建CartPole环境
    env = gym.make('CartPole-v1')
    
    # 获取状态和动作空间大小
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # 创建智能体
    agent = DQNAgent(state_size, action_size)
    
    # 记录每个回合的得分
    scores = []
    
    for episode in range(episodes):
        # 重置环境
        state = env.reset()
        if isinstance(state, tuple):  # 处理gym新版本的返回值
            state = state[0]
        
        state = np.reshape(state, [1, state_size])[0]
        done = False
        score = 0
        
        while not done:
            # 渲染环境（定期）
            if episode % render_interval == 0:
                env.render()
            
            # 选择动作
            action = agent.act(state)
            
            # 执行动作
            try:
                # 尝试新版本Gym的返回格式
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except ValueError:
                # 兼容旧版本Gym
                next_state, reward, done, info = env.step(action)
                
            next_state = np.reshape(next_state, [1, state_size])[0]
            
            # 修改奖励以加快学习
            reward = reward if not done or score >= 499 else -10
            
            # 记住经验
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            score += 1
            
            # 从经验中学习
            agent.replay()
            
            if done:
                # 每11个回合更新一次目标网络 
                if episode % 11 == 0:
                    agent.update_target_model()
                
                scores.append(score)
                print(f"回合: {episode}/{episodes}, 得分: {score}, 探索率: {agent.epsilon:.2f}")
                break
    
    env.close()
    
    # 绘制学习曲线
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.xlabel('episodes')
    plt.ylabel('score')
    plt.title('DQN learning curve')
    plt.grid(True)
    plt.show()
    
    # 返回训练好的智能体
    return agent

def test_dqn(episodes=10):
    """测试训练好的DQN智能体"""
    
    # 创建环境
    env = gym.make('CartPole-v1', render_mode='human')
    
    # 获取状态和动作空间大小
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # 创建智能体
    agent = DQNAgent(state_size, action_size)
    agent.epsilon = 0.01  # 测试时使用较小的探索率
    
    # 加载预训练模型（如果有）
    try:
        agent.model.load_state_dict(torch.load('dqn_cartpole.pth'))
        print("加载预训练模型成功")
    except:
        print("没有找到预训练模型，使用随机初始化的模型")
    
    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        state = np.reshape(state, [1, state_size])[0]
        done = False
        score = 0
        
        while not done:
            # 渲染环境
            env.render()
            
            # 选择动作
            action = agent.act(state)
            
            # 执行动作
            try:
                # 尝试新版本Gym的返回格式
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except ValueError:
                # 兼容旧版本Gym
                next_state, reward, done, info = env.step(action)
                
            next_state = np.reshape(next_state, [1, state_size])[0]
            
            state = next_state
            score += 1
            
            if done:
                print(f"测试回合 {episode}, 得分: {score}")
                break
    
    env.close()

if __name__ == "__main__":
    print("开始训练DQN智能体...")
    trained_agent = train_dqn(episodes=100, render_interval=50)
    
    # 保存模型
    if trained_agent is not None:
        torch.save(trained_agent.model.state_dict(), 'dqn_cartpole.pth')
        print("模型已保存")
    
    print("\n测试训练好的DQN智能体:")
    test_dqn(episodes=3) 