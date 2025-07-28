# 强化学习算法实现与对比

本项目实现了四种经典的强化学习算法，用于解决CartPole-v1环境问题。其中DQN参数可能比较奇怪，原因是参数是经过我有优化算法优化过的参数，算是比较好的参数，虽然奇怪但是性能不错。

## 文件结构

- `DQN.py` - Deep Q-Network算法（价值函数方法）
- `REINFORCE.py` - 经典REINFORCE算法（策略梯度蒙特卡洛方法）
- `QAC.py` - Q-Actor-Critic算法（QAC）
- `A2C.py` - Advantage Actor-Critic算法（A2C）

## 环境配置

### 系统要求
- Python 3.7+

### 依赖安装
```bash

# 安装依赖
pip install torch torchvision torchaudio
pip install gymnasium[classic_control]  # 新版本
# 或
pip install gym[classic_control]        # 旧版本
pip install numpy matplotlib

# 如果遇到渲染问题，额外安装
pip install pygame
```

### 验证安装
```python
import gym
import torch
print(f"Gym version: {gym.__version__}")
print(f"PyTorch version: {torch.__version__}")

# 测试环境
env = gym.make('CartPole-v1')
print(f"观察空间: {env.observation_space}")
print(f"动作空间: {env.action_space}")
```

## 算法原理

### 1. Deep Q-Network

**核心思想**: 使用深度神经网络近似Q值函数，通过经验回放和目标网络稳定训练。

**网络结构**:
- Q网络：Q(s, a; θ) - 估计动作价值函数
- 目标网络：Q(s, a; θ⁻) - 提供稳定的目标值

**更新公式**:
```
# Bellman方程
Q(s, a) = r + γ max_a' Q(s', a')

# 损失函数
L = (r + γ max_a' Q(s', a'; θ⁻) - Q(s, a; θ))²
```

**关键技术**:
- **经验回放**: 打破数据相关性，提高样本效率
- **目标网络**: 每隔一定步数更新，稳定训练
- **ε-贪婪策略**: 平衡探索与利用

**特点**:
- 样本效率高，可以重复使用经验
- 适用于离散动作空间
- 只能处理离散动作
- 容易过估计Q值

### 2. REINFORCE算法

**核心思想**: 纯策略梯度方法，使用蒙特卡洛回报直接优化策略。

**网络结构**:
- 只有策略网络：π(a|s, θ)

**更新公式**:
```
θ ← θ + α∇_θ ln π(a_t|s_t, θ) × G_t
```
其中 G_t = Σ_{k=t+1}^T γ^{k-t-1} r_k（蒙特卡洛回报）

**特点**:
- 简单直接
- 高方差，学习不稳定
- 需要完整episode才能更新

### 2. Q-Actor-Critic 

**核心思想**: 引入Critic网络估计Q值，减少REINFORCE的方差。

**网络结构**:
- Actor网络：π(a|s, θ) - 策略网络
- Critic网络：Q(s, a, w) - 动作价值网络

**更新公式**:
```
# Critic更新
w ← w + α_w[r + γQ(s', a', w) - Q(s, a, w)]∇_w Q(s, a, w)

# Actor更新  
θ ← θ + α_θ ∇_θ ln π(a|s, θ) × Q(s, a, w)
```

**特点**:
- 可以单步更新，提高样本效率
- 使用Q值替代蒙特卡洛回报，减少方差
- 需要同时训练两个网络，增加复杂性

### 3. Advantage Actor-Critic

**核心思想**: 使用优势函数A(s,a)进一步减少方差，提高学习稳定性。

**网络结构**:
- Actor网络：π(a|s, θ) - 策略网络  
- Critic网络：V(s, w) - 状态价值网络

**更新公式**:
```
# 优势函数
A(s, a) = R - V(s)  # 或者 A(s, a) = r + γV(s') - V(s)

# Critic更新
L_critic = (R - V(s))²

# Actor更新
L_actor = -ln π(a|s, θ) × A(s, a)
```

**特点**:
- 优势函数减少方差，学习更稳定
- 可以批量更新，提高效率

## 算法演进与融合

### 演进路径

```
价值函数方法: Q-Learning → DQN → Double DQN → Dueling DQN...
策略梯度方法: REINFORCE → Actor-Critic → A2C → A3C/PPO/...
融合方法: Actor-Critic结合了两种思路
```

### 两大流派对比

| 特征 | 价值函数方法  | 策略梯度方法 |
|------|-------------------|---------------------------|
| **核心思想** | 学习Q(s,a)，间接得到策略 | 直接学习策略π(a\|s) |
| **动作空间** | 离散动作 | 连续/离散动作 |
| **探索方式** | ε-贪婪 | 随机策略 |
| **收敛性** | 确定性策略 | 随机策略 |
| **样本效率** | 高（经验回放） | 低（在线学习） |
| **稳定性** | 较稳定 | 高方差 |

### 1. 从REINFORCE到Actor-Critic

**问题**: REINFORCE使用蒙特卡洛回报G_t，方差很大
```python
# REINFORCE
policy_loss = -log_prob * G_t  # G_t方差大
```

**解决**: 引入Critic网络估计价值函数
```python
# Actor-Critic  
policy_loss = -log_prob * Q(s, a)  # Q(s,a)方差更小
```

### 2. 从QAC到A2C

**问题**: Q(s,a)仍然有偏差，且需要为每个动作估计Q值
```python
# QAC需要Q(s,a)
q_value = critic(state, action)
```

**解决**: 使用优势函数A(s,a) = R - V(s)
```python
# A2C使用优势函数
advantage = returns - values  # 减少方差
policy_loss = -log_prob * advantage
```

### 3. 核心融合思想

#### 价值函数方法的核心
所有价值函数方法都基于**Bellman方程**:
```
Q(s, a) = E[r + γ max_a' Q(s', a')]
```

#### 策略梯度方法的核心
所有策略梯度方法都遵循**策略梯度定理**:
```
∇_θ J(θ) = E[∇_θ ln π(a|s, θ) × Ψ_t]
```

区别在于Ψ_t的选择:
- **REINFORCE**: Ψ_t = G_t (蒙特卡洛回报)
- **AC**: Ψ_t = Q(s, a) (动作价值)
- **A2C**: Ψ_t = A(s, a) (优势函数)

#### Actor-Critic的融合
Actor-Critic巧妙地结合了两种方法:
- **Critic**: 使用价值函数方法学习V(s)或Q(s,a)
- **Actor**: 使用策略梯度方法学习π(a|s)
- **优势**: 既有价值函数的稳定性，又有策略梯度的灵活性

## 性能对比

| 算法 | 方差 | 偏差 | 样本效率 | 稳定性 | 复杂度 | 动作空间 |
|------|------|------|----------|--------|--------|----------|
| DQN | 低 | 有 | 高 | 好 | 中 | 离散 |
| REINFORCE | 高 | 无 | 低 | 差 | 低 | 连续/离散 |
| QAC | 中 | 有 | 中 | 中 | 中 | 连续/离散 |
| A2C | 低 | 有 | 高 | 好 | 中 | 连续/离散 |


## 运行方法

### 快速开始
```bash
# 运行DQN
python DQN.py

# 运行REINFORCE
python REINFORCE.py

# 运行Actor-Critic
python QAC.py

# 运行A2C
python A2C.py
```

### 训练参数调整
每个算法都支持参数调整，在主函数中修改：
```python
# 例如在REINFORCE.py中
trained_agent = train_reinforce(
    episodes=1000,      # 训练轮数
    render_interval=100 # 渲染间隔
)
```

### 模型保存与加载
训练完成后，模型会自动保存：
- DQN: `dqn_cartpole.pth`
- REINFORCE: `reinforce_policy_cartpole.pth`
- AC: `qac_actor_cartpole.pth`, `qac_critic_cartpole.pth`
- A2C: `actor_cartpole.pth`, `critic_cartpole.pth`

## 超参数说明

### 共同参数
- `gamma`: 折扣因子 (0.99)
- `hidden_size`: 隐藏层大小 (128)
- `episodes`: 训练轮数 (500-1000)
- `max_steps`: 每轮最大步数 (500)

### DQN特定参数
- `learning_rate`: 学习率 (0.0002)
- `epsilon`: 探索率 (1.0 → 0.04)
- `epsilon_decay`: 探索率衰减 (0.993)
- `batch_size`: 批次大小 (128)
- `memory_size`: 经验回放缓冲区大小 (10000)
- `target_update`: 目标网络更新频率 (每100步)

### 策略梯度算法参数
- `lr_actor`: Actor学习率 (0.001)
- `lr_critic`: Critic学习率 (0.005)

### 算法特定参数
- **DQN**: 重点调节探索率和经验回放参数
- **REINFORCE**: 只需要策略网络学习率
- **QAC**: 需要同时调节Actor和Critic学习率
- **A2C**: 可以使用不同的优势函数计算方式



