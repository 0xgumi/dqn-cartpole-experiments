import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd
import os
from collections import deque
# 현재 실행 중인 파일이 위치한 디렉토리 가져오기
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 해당 디렉토리에 파일 저장되도록 경로 설정
CSV_FILE = os.path.join(BASE_DIR, "dqn_cartpole_v1_results.csv")
TXT_FILE = os.path.join(BASE_DIR, "dqn_cartpole_v1_summary.txt")

# 환경 설정
def create_env():
    return gym.make("CartPole-v1")

# DQN 모델 정의
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 하이퍼파라미터 설정
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995  # 탐색 감소율
learning_rate = 0.001
batch_size = 32
memory_size = 5000  # 메모리 크기 증가
train_start = 1000
N = 5  # 타겟 네트워크 업데이트 간격

def train_dqn(q_network, target_network, memory, optimizer, loss_fn, device):
    if len(memory) < train_start:
        return
    
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    q_values = q_network(states).gather(1, actions)
    next_q_values = target_network(next_states).max(1)[0].detach()
    target_q_values = rewards + (gamma * next_q_values * (1 - dones))

    loss = loss_fn(q_values.squeeze(), target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# DQN 실험 실행
def run_experiment(episodes=300, trials=50):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    results = []

    for trial in range(trials):
        env = create_env()
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        q_network = DQN(state_size, action_size).to(device)
        target_network = DQN(state_size, action_size).to(device)
        target_network.load_state_dict(q_network.state_dict())
        target_network.eval()
        optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        memory = deque(maxlen=memory_size)

        last_50_scores = []
        epsilon = 1.0
        
        for episode in range(episodes):
            state, _ = env.reset()
            state = np.array(state)
            total_reward = 0
            
            for t in range(500):
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                        action = torch.argmax(q_network(state_tensor)).item()
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = np.array(next_state)
                done = terminated or truncated
                total_reward += reward
                
                memory.append((state, action, reward, next_state, done))
                state = next_state
                train_dqn(q_network, target_network, memory, optimizer, loss_fn, device)

                if done:
                    break
            
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            if episode % N == 0:
                target_network.load_state_dict(q_network.state_dict())

            if episode >= 250:
                last_50_scores.append(total_reward)
        
        min_score = min(last_50_scores)
        max_score = max(last_50_scores)
        avg_score = sum(last_50_scores) / len(last_50_scores)
        std_dev = np.std(last_50_scores)

        results.append([trial+1, min_score, max_score, avg_score, std_dev])
        env.close()
        print(f"Trial {trial+1}/{trials} 완료: 평균 {avg_score:.2f}, 표준편차 {std_dev:.2f}")
    
    # 결과 CSV 파일 저장
    df = pd.DataFrame(results, columns=["실험번호", "최저점수", "최고점수", "평균점수", "표준편차"])
    df.to_csv(CSV_FILE, index=False)
    
    # 요약본 TXT 저장
    with open(TXT_FILE, "w") as f:
        f.write("🔥 DQN 실험 결과 (50번 반복 테스트)\n")
        f.write("------------------------------------\n")
        for row in results:
            f.write(f"[실험 {row[0]}] 최저: {row[1]}, 최고: {row[2]}, 평균: {row[3]:.2f}, 표준편차: {row[4]:.2f}\n")
        f.write("------------------------------------\n")
        f.write(f"📊 전체 평균:\n")
        f.write(f"- 최저점 평균: {np.mean([r[1] for r in results]):.2f}\n")
        f.write(f"- 최고점 평균: {np.mean([r[2] for r in results]):.2f}\n")
        f.write(f"- 평균 점수 평균: {np.mean([r[3] for r in results]):.2f}\n")
        f.write(f"- 표준편차 평균: {np.mean([r[4] for r in results]):.2f}\n")

if __name__ == "__main__":
    run_experiment()
