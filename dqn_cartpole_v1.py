import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# 1️⃣ 환경 설정 (렌더링 없이 빠른 실행)
env = gym.make("CartPole-v1")  
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# 2️⃣ DQN 모델 정의
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

# 3️⃣ 하이퍼파라미터 설정
gamma = 0.99  
epsilon = 1.0  
epsilon_min = 0.01  
epsilon_decay = 0.995
learning_rate = 0.001  
batch_size = 32  
memory_size = 2000  
train_start = 1000  

# 4️⃣ DQN 모델 & 최적화 함수 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  
q_network = DQN(state_size, action_size).to(device)
target_network = DQN(state_size, action_size).to(device)  # 타겟 네트워크 추가
target_network.load_state_dict(q_network.state_dict())  # 초기 가중치 동기화
target_network.eval()  # 타겟 네트워크는 학습하지 않음

optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# 5️⃣ 경험 리플레이 메모리
memory = deque(maxlen=memory_size)

# 6️⃣ DQN 학습 함수
def train_dqn():
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
    next_q_values = target_network(next_states).max(1)[0].detach()  # 타겟 네트워크 사용
    target_q_values = rewards + (gamma * next_q_values * (1 - dones))

    loss = loss_fn(q_values.squeeze(), target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 7️⃣ 학습 실행 (총 300번 & 마지막 50번 기록 저장)
episodes = 300  
last_50_scores = []  # 최근 50개 점수 저장

for episode in range(episodes):
    state, _ = env.reset()
    state = np.array(state)
    total_reward = 0

    for t in range(500):
        # 행동 선택 (탐험 vs 활용)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action = torch.argmax(q_network(state_tensor)).item()

        # 환경 진행
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.array(next_state)
        done = terminated or truncated
        total_reward += reward

        # 경험 저장
        memory.append((state, action, reward, next_state, done))

        # 상태 업데이트
        state = next_state

        # DQN 학습
        train_dqn()

        if done:
            break

    # 탐험 비율 감소
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # 🔥 5 에피소드마다 타겟 네트워크 업데이트
    if episode % 5 == 0:
        target_network.load_state_dict(q_network.state_dict())

    # 마지막 50개 점수 저장
    if episode >= 250:
        last_50_scores.append(total_reward)

    print(f"Episode {episode + 1}/{episodes}, Score: {total_reward}, Epsilon: {epsilon:.4f}")

# 8️⃣ 마지막 50개 점수 분석
if len(last_50_scores) > 0:
    min_score = min(last_50_scores)
    max_score = max(last_50_scores)
    avg_score = sum(last_50_scores) / len(last_50_scores)
    std_dev = np.std(last_50_scores)  # 표준편차 계산

    print("\n🔥 마지막 50개 에피소드 성능:")
    print(f"✅ 최저 점수: {min_score}")
    print(f"✅ 최고 점수: {max_score}")
    print(f"✅ 평균 점수: {avg_score:.2f}")
    print(f"✅ 표준편차: {std_dev:.2f}")

print("\n훈련 완료!")
env.close()  # 🎥 모든 학습이 끝나면 환경 닫기
