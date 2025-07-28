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
    Critic网络：学习动作价值函数Q(s,a)
    输入状态和动作，输出Q值
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # 输出Q值

    def forward(self, state, action):
        # 将状态和动作拼接
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

class QActorCriticAgent:

    def __init__(self, state_size, action_size, lr_actor=0.001, lr_critic=0.005, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # 折扣因子

        # 创建Actor和Critic网络
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_action(self, state):

        state = torch.FloatTensor(state).unsqueeze(0)

        # 获取动作概率分布
        action_probs = self.actor(state)

        # 创建分类分布并采样动作
        dist = Categorical(action_probs)
        action = dist.sample()

        # 计算log概率
        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def get_q_value(self, state, action):

        state = torch.FloatTensor(state).unsqueeze(0)
        # 将动作转换为one-hot编码
        action_onehot = torch.zeros(1, self.action_size)
        action_onehot[0, action] = 1.0

        q_value = self.critic(state, action_onehot)
        return q_value

    def update(self, state, action, action_log_prob, reward, next_state, done):

        # 转换为张量
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)

        # 将动作转换为one-hot编码
        action_onehot = torch.zeros(1, self.action_size)
        action_onehot[0, action] = 1.0

        # 获取当前Q(s,a)
        current_q = self.critic(state, action_onehot)

        # 获取下一状态的动作和Q值
        if not done:
            # 根据当前策略选择下一个动作
            next_action_probs = self.actor(next_state)
            next_dist = Categorical(next_action_probs)
            next_action = next_dist.sample()

            # 获取Q(s',a')
            next_action_onehot = torch.zeros(1, self.action_size)
            next_action_onehot[0, next_action] = 1.0
            next_q = self.critic(next_state, next_action_onehot)
        else:
            next_q = torch.tensor([[0.0]])

        # 计算TD目标：r + γQ(s',a')
        td_target = reward + self.gamma * next_q

        # Critic更新：w = w + α_w[r + γq(s',a',w) - q(s,a,w)]∇_w q(s,a,w)
        critic_loss = F.mse_loss(current_q, td_target.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor更新：θ = θ + α_θ ∇_θ ln π(a|s,θ) q(s,a,w)
        # 重新计算Q值 因为critic已经更新 
        with torch.no_grad():
            updated_q = self.critic(state, action_onehot)

        actor_loss = -action_log_prob * updated_q

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

def train_qac(episodes=1000, max_steps=500, render_interval=100):

    # 创建环境
    env = gym.make('CartPole-v1')

    # 获取状态和动作空间大小
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 创建智能体
    agent = QActorCriticAgent(state_size, action_size)

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
        episode_actor_losses = []
        episode_critic_losses = []

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

            # QAC单步更新
            actor_loss, critic_loss = agent.update(state, action, log_prob, reward, next_state, done)

            episode_actor_losses.append(actor_loss)
            episode_critic_losses.append(critic_loss)

            state = next_state
            episode_reward += reward

            if done:
                break
        
        # 记录数据
        episode_rewards.append(episode_reward)
        actor_losses.append(np.mean(episode_actor_losses))
        critic_losses.append(np.mean(episode_critic_losses))
        
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

def test_qac(agent=None, episodes=5):

    env = gym.make('CartPole-v1', render_mode='human')

    if agent is None:
        # 如果没有提供智能体，尝试加载保存的模型
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = QActorCriticAgent(state_size, action_size)

        try:
            agent.actor.load_state_dict(torch.load('qac_actor_cartpole.pth'))
            agent.critic.load_state_dict(torch.load('qac_critic_cartpole.pth'))
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
    print("开始训练Q-Actor-Critic智能体...")
    trained_agent = train_qac(episodes=500, render_interval=100)

    # 保存模型
    torch.save(trained_agent.actor.state_dict(), 'qac_actor_cartpole.pth')
    torch.save(trained_agent.critic.state_dict(), 'qac_critic_cartpole.pth')
    print("模型已保存")

    print("\n测试训练好的Q-Actor-Critic智能体:")
    test_qac(trained_agent, episodes=3)
