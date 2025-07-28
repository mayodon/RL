
import numpy as np

class GridWorld:
    """
    网格世界环境
    """
    def __init__(self, size=5):
        self.size = size
        self.grid = np.zeros((size, size))
        
        # 设置目标位置
        self.goal = (size-1, size-1)
        self.grid[self.goal] = 2
        
        # 设置陷阱
        self.traps = [(1, 1), (2, 3), (3, 1)]
        for trap in self.traps:
            self.grid[trap] = -1
            
        # 设置起始位置
        self.agent_pos = (0, 0)
        
        # 动作空间: 上(0), 右(1), 下(2), 左(3)
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.action_names = ["上", "右", "下", "左"]
        
    def reset(self):
        """重置环境，返回初始状态"""
        self.agent_pos = (0, 0)
        return self.agent_pos
    
    def step(self, action):
        """执行动作，返回新状态、奖励、是否结束"""
        # 计算新位置
        new_pos = (self.agent_pos[0] + self.actions[action][0], 
                  self.agent_pos[1] + self.actions[action][1])
        
        # 检查是否超出边界
        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
            self.agent_pos = new_pos
            
        # 计算奖励和是否结束
        if self.agent_pos == self.goal:
            reward = 10
            done = True
        elif self.agent_pos in self.traps:
            reward = -10
            done = True
        else:
            reward = -1  # 每走一步的小惩罚，鼓励找到最短路径
            done = False
            
        return self.agent_pos, reward, done
    
    def render_text(self):
        grid_copy = self.grid.copy()
        grid_copy[self.agent_pos] = 1  # 标记智能体位置
        
        print("\n" + "-" * (self.size * 4 + 1))
        for i in range(self.size):
            row = "|"
            for j in range(self.size):
                if (i, j) == self.goal:
                    row += " G |"
                elif (i, j) in self.traps:
                    row += " T |"
                elif (i, j) == self.agent_pos:
                    row += " A |"
                else:
                    row += "   |"
            print(row)
            print("-" * (self.size * 4 + 1))
        print("")

def q_learning(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1, render_interval=50):

    # 初始化Q表
    q_table = {}
    for i in range(env.size):
        for j in range(env.size):
            q_table[(i, j)] = np.zeros(4)
    
    # 记录每个回合的总奖励
    rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # 选择动作（ε-贪婪策略）越大
            if np.random.random() < epsilon:
                action = np.random.randint(4)  # 随机探索
            else:
                action = np.argmax(q_table[state])  # 选择最佳动作
            
            # 执行动作
            next_state, reward, done = env.step(action)
            
            # 更新Q值
            q_table[state][action] = q_table[state][action] +  alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
            )
            
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
        
        # 定期可视化
        if episode % render_interval == 0 or episode == episodes - 1:
            print(f"回合 {episode}, 总奖励: {total_reward}")
            # 打印Q表中的一些值
            print("Q表示例值:")
            for i in range(min(3, env.size)):
                for j in range(min(3, env.size)):
                    print(f"状态({i},{j}): {q_table[(i,j)]}")
            print("")
    
    print("平均奖励:", sum(rewards) / len(rewards))
    print("最大奖励:", max(rewards))
    print("最小奖励:", min(rewards))
    
    return q_table

def print_policy(env, q_table):
    
    policy = np.zeros((env.size, env.size), dtype=str)
    
    for i in range(env.size):
        for j in range(env.size):
            state = (i, j)
            if state == env.goal:
                policy[i, j] = "G"  # 目标
            elif state in env.traps:
                policy[i, j] = "T"  # 陷阱
            else:
                # 获取最佳动作
                best_action = np.argmax(q_table[state])
                if best_action == 0:
                    policy[i, j] = "↑"  # 上
                elif best_action == 1:
                    policy[i, j] = "→"  # 右
                elif best_action == 2:
                    policy[i, j] = "↓"  # 下
                elif best_action == 3:
                    policy[i, j] = "←"  # 左
    
    print("\n学习到的策略:")
    print("-" * (env.size * 4 + 1))
    for i in range(env.size):
        row = "|"
        for j in range(env.size):
            row += f" {policy[i, j]} |"
        print(row)
        print("-" * (env.size * 4 + 1))
    print("")


if __name__ == "__main__":
    # 创建环境
    env = GridWorld(size=5)
    
    # 训练智能体
    print("开始训练...")
    q_table = q_learning(env, episodes=200, render_interval=50)
    
    # 打印学习到的策略
    print_policy(env, q_table)
    
    
