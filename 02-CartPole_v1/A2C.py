import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib
matplotlib.use('TKAgg') 
import matplotlib.pyplot as plt

class Actor(nn.Module):
    """
    Actor网络：学习策略π(a|s)
    输入状态，输出动作概率分布
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # 使用softmax输出动作概率分布
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs

class Critic(nn.Module):
    """
    Critic网络：学习状态价值函数V(s)
    输入状态，输出状态价值
    """
    def __init__(self, state_size, hidden_size=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # 输出单个价值
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class ActorCriticAgent:
   
    def __init__(self, state_size, action_size, lr_actor=0.001, lr_critic=0.005, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # 折扣因子
        
        # 创建Actor和Critic网络
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 存储轨迹数据
        self.reset_trajectory()
        
    def reset_trajectory(self):

        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        
    def select_action(self, state):
      
        state = torch.FloatTensor(state).unsqueeze(0)
        
        # 获取动作概率分布
        action_probs = self.actor(state)
        
        # 获取状态价值
        value = self.critic(state)
        
        # 创建分类分布并采样动作
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # 计算log概率 目的是转换了选择动作的概率为对数概率
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value
    
    def store_transition(self, state, action, reward, log_prob, value):
    
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def compute_returns(self, next_value=0):
      
        returns = []
        R = next_value
        
        # 从后往前计算折扣回报
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
            
        return returns
    
    def update(self, next_value=0):
  
        if len(self.rewards) == 0:
            return
            
        # 计算折扣回报
        returns = self.compute_returns(next_value)
        returns = torch.FloatTensor(returns)
        
        # 转换为张量
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values).squeeze()
        
        # 计算优势函数 A(s,a) = R - V(s)
        advantages = returns - values
        
        # Actor损失：策略梯度损失 梯度上升找最大值 不过这里取负号借用torch的梯度下降方法
        # L_actor = -log π(a|s) * A(s,a)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic损失：均方误差损失 用蒙特卡洛回报作为V(s)的目标
        # L_critic = (R - V(s))^2
        critic_loss = F.mse_loss(values, returns)
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 重置轨迹
        self.reset_trajectory()
        
        return actor_loss.item(), critic_loss.item()

def train_actor_critic(episodes=1000, max_steps=500, render_interval=100):

    # 创建环境
    env = gym.make('CartPole-v1')
    
    # 获取状态和动作空间大小
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # 创建智能体
    agent = ActorCriticAgent(state_size, action_size)
    
    # 记录训练数据
    episode_rewards = []
    actor_losses = []
    critic_losses = []
    
    for episode in range(episodes):
        # 重置环境
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
            
        episode_reward = 0
        
        for step in range(max_steps):
            # 渲染环境（定期）
            if episode % render_interval == 0:
                env.render()
            
            # 选择动作
            action, log_prob, value = agent.select_action(state)
            
            # 执行动作
            try:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            except ValueError:
                next_state, reward, done, _ = env.step(action)
            
            # 存储转移
            agent.store_transition(state, action, reward, log_prob, value)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # 计算下一个状态的价值（如果episode没有结束）
        if not done:
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            next_value = agent.critic(next_state).item()
        else:
            next_value = 0
        
        # 更新网络
        actor_loss, critic_loss = agent.update(next_value)
        
        # 记录数据
        episode_rewards.append(episode_reward)
        if actor_loss is not None:
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
        
        # 打印进度
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
    
    env.close()
    
    # 绘制学习曲线
    plt.figure(figsize=(15, 5))
    
    # 奖励曲线
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # Actor损失
    plt.subplot(1, 3, 2)
    plt.plot(actor_losses)
    plt.title('Actor Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Critic损失
    plt.subplot(1, 3, 3)
    plt.plot(critic_losses)
    plt.title('Critic Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return agent

def test_actor_critic(agent=None, episodes=5):
    
    env = gym.make('CartPole-v1', render_mode='human')
    
    if agent is None:
        # 如果没有提供智能体，尝试加载保存的模型
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = ActorCriticAgent(state_size, action_size)
        
        try:
            agent.actor.load_state_dict(torch.load('actor_cartpole.pth'))
            agent.critic.load_state_dict(torch.load('critic_cartpole.pth'))
            print("加载预训练模型成功")
        except:
            print("没有找到预训练模型，使用随机初始化的模型")
    
    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
            
        episode_reward = 0
        
        while True:
            env.render()
            
            # 选择动作（不需要log_prob和value）
            action, _, _ = agent.select_action(state)
            
            try:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            except ValueError:
                next_state, reward, done, _ = env.step(action)
            
            state = next_state
            episode_reward += reward
            
            if done:
                print(f"Test Episode {episode}, Reward: {episode_reward}")
                break
    
    env.close()

if __name__ == "__main__":
    print("开始训练Actor-Critic智能体...")
    trained_agent = train_actor_critic(episodes=500, render_interval=100)
    
    # 保存模型
    torch.save(trained_agent.actor.state_dict(), 'actor_cartpole.pth')
    torch.save(trained_agent.critic.state_dict(), 'critic_cartpole.pth')
    print("模型已保存")
    
    print("\n测试训练好的Actor-Critic智能体:")
    test_actor_critic(trained_agent, episodes=3)
