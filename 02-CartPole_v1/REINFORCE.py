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

class PolicyNetwork(nn.Module):
    """
    策略网络：π(a|s, θ)
    输入状态，输出动作概率分布
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # 使用softmax输出动作概率分布
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs

class REINFORCEAgent:

    def __init__(self, state_size, action_size, lr=0.00001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # 折扣因子
        
        # 创建策略网络
        self.policy = PolicyNetwork(state_size, action_size)
        
        # 优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 存储episode数据
        self.reset_episode()
        
    def reset_episode(self):
    
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        
    def select_action(self, state):
    
        state = torch.FloatTensor(state).unsqueeze(0)
        
        # 获取动作概率分布
        action_probs = self.policy(state)
        
        # 创建分类分布并采样动作
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # 计算log概率
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob
    
    def store_transition(self, state, action, reward, log_prob):

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
    
    def compute_returns(self):
     
        returns = []
        G = 0
        
        # 从后往前计算折扣回报
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
            
        return returns
    
    def update(self):
     
        if len(self.rewards) == 0:
            return 0
            
        # 计算蒙特卡洛回报
        returns = self.compute_returns()
        returns = torch.FloatTensor(returns)
        
        # 标准化回报
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 转换log概率为张量
        log_probs = torch.stack(self.log_probs)
        
        # 计算策略梯度损失
        # L = -Σ_t log π(a_t|s_t, θ) * G_t
        # 负号是因为我们要最大化目标函数，但优化器执行梯度下降
        policy_loss = -(log_probs * returns).mean()
        
        # 更新策略网络
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # 重置episode数据
        self.reset_episode()
        
        return policy_loss.item()

def train_reinforce(episodes=1000, max_steps=500, render_interval=100):

    # 创建环境
    env = gym.make('CartPole-v1')
    
    # 获取状态和动作空间大小
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # 创建智能体
    agent = REINFORCEAgent(state_size, action_size)
    
    # 记录训练数据
    episode_rewards = []
    policy_losses = []
    
    for episode in range(episodes):
        # 重置环境
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
            
        episode_reward = 0
        
        # 生成完整episode
        for _ in range(max_steps):
            # 渲染环境（定期）
            if episode % render_interval == 0:
                env.render()
            
            # 选择动作
            action, log_prob = agent.select_action(state)
            
            # 执行动作
            try:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            except ValueError:
                next_state, reward, done, _ = env.step(action)
            
            # 存储转移
            agent.store_transition(state, action, reward, log_prob)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Episode结束后更新策略
        policy_loss = agent.update()
        
        # 记录数据
        episode_rewards.append(episode_reward)
        policy_losses.append(policy_loss)
        
        # 打印进度
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
    
    env.close()
    
    # 绘制学习曲线
    plt.figure(figsize=(12, 4))
    
    # 奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # 策略损失
    plt.subplot(1, 2, 2)
    plt.plot(policy_losses)
    plt.title('Policy Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return agent

def test_reinforce(agent=None, episodes=5):
    
    env = gym.make('CartPole-v1', render_mode='human')
    
    if agent is None:
        # 如果没有提供智能体，尝试加载保存的模型
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = REINFORCEAgent(state_size, action_size)
        
        try:
            agent.policy.load_state_dict(torch.load('reinforce_policy_cartpole.pth'))
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
            
            # 选择动作
            action, _ = agent.select_action(state)
            
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
    print("开始训练REINFORCE智能体...")
    trained_agent = train_reinforce(episodes=1000, render_interval=100)
    
    # 保存模型
    torch.save(trained_agent.policy.state_dict(), 'reinforce_policy_cartpole.pth')
    print("模型已保存")
    
    print("\n测试训练好的REINFORCE智能体:")
    test_reinforce(trained_agent, episodes=3)
