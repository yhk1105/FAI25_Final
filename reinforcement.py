import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from multiprocessing import get_context
from game.game import setup_config, start_poker
from agents.Encoder import Encoderfor2P as Encoder
from tqdm import tqdm
from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai
from baseline4 import setup_ai as baseline4_ai
from baseline5 import setup_ai as baseline5_ai
from baseline6 import setup_ai as baseline6_ai
from baseline7 import setup_ai as baseline7_ai
from game.players import BasePokerPlayer
import copy
import time

# --- 定義神經網路架構 ---


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent():
    def __init__(self, input_dim, output_dim):
        self.q_net = DQN(input_dim, output_dim)
        self.target_q_net = DQN(input_dim, output_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.buffer = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 0.3
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.output_dim = output_dim
        self.loss_history = []

    def train_from_episode(self, episode):
        for transition in episode:
            self.buffer.append(transition)
        self.train()
        # 避免把模型權重丟在根目錄/agents，統一收在 models/
        self.save("models/dqn_poker_best.pt")

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.output_dim - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(state)
            return torch.argmax(q_values, dim=1).item()

    def train(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        state_dim = len(states[0])
        next_states = [s if s is not None else np.zeros(
            state_dim, dtype=np.float32) for s in next_states]
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.q_net(states).gather(1, actions)
        next_q_values = self.target_q_net(next_states).max(1, keepdim=True)[0]
        expected_q = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
        self.loss_history.append(loss.item())

    def update_target(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_q_net.load_state_dict(self.q_net.state_dict())


class RLAgent(BasePokerPlayer):
    def __init__(self, dqn_model, encoder):
        super().__init__()
        self.dqn = dqn_model
        self.encoder = encoder
        self.episode_memory = []
        self.last_state = None
        self.last_action = None
        self.my_seat = None
        self.total_seats = 0
        self.win_history = []
        self.my_initial_stack = 1000
        self.episode_done = False

    def declare_action(self, valid_actions, hole_card, round_state):
        is_big_blind = 1 if round_state['big_blind_pos'] == self.my_seat else 0
        my_stack = round_state['seats'][self.my_seat]['stack']
        opponent_stack = round_state['seats'][1 - self.my_seat]['stack']
        last_oppo_action = round_state['action_histories']['preflop'][-1]['action'] if 'preflop' in round_state[
            'action_histories'] and round_state['action_histories']['preflop'] else "CALL"
        call_amount = [va for va in valid_actions if va['action']
                       == "call"][0]['amount']
        min_raise = [va for va in valid_actions if va['action']
                     == "raise"][0]['amount']['min']
        max_raise = [va for va in valid_actions if va['action']
                     == "raise"][0]['amount']['max']
        state = self.encoder.encode_game_state(
            hole_cards=hole_card,
            community_cards=round_state['community_card'],
            round_count=round_state['round_count'],
            my_stack=my_stack,
            opponent_stack=opponent_stack,
            last_opponent_action=last_oppo_action,
            is_big_blind=is_big_blind,
            pot_size=round_state['pot']['main']['amount'],
            call_amount=call_amount,
            min_raise=min_raise,
            max_raise=max_raise
        )
        action_idx = self.dqn.select_action(state)
        action_name = ["fold", "call", "raise_min",
                       "raise_mid", "raise_max"][action_idx]
        action = None
        amount = None
        for va in valid_actions:
            if action_name == "fold" and va['action'] == "fold":
                action, amount = "fold", va['amount']
                break
            elif (action_name == "call" and va['action'] == "call") or (action_name == "raise" and len(valid_actions) == 2):
                action, amount = "call", va['amount']
                break
            elif action_name.startswith("raise") and va['action'] == "raise":
                if va['amount']['min'] == -1:
                    fallback = next(
                        a for a in valid_actions if a['action'] == "call")
                    action, amount = fallback['action'], fallback['amount']
                else:
                    if action_name == "raise_min":
                        action, amount = "raise", va['amount']['min']
                    elif action_name == "raise_mid":
                        action, amount = "raise", int(
                            (va['amount']['max'] + va['amount']['min']) / 2)
                    elif action_name == "raise_max":
                        action, amount = "raise", va['amount']['max']
                break
        if action is None:
            fallback = valid_actions[0]
            action, amount = fallback['action'], fallback['amount']

        if self.last_state is not None:
            self.episode_memory.append(
                (self.last_state, self.last_action, 0, state, False))
        self.last_state = state
        self.last_action = action_idx
        return action, amount

    def receive_round_result_message(self, winners, hand_info, round_state):
        my_final_stack = [p['stack']
                          for p in round_state['seats'] if p['uuid'] == self.uuid][0]
        reward = my_final_stack - self.my_initial_stack
        self.my_initial_stack = my_final_stack
        if self.last_state is not None:
            self.episode_memory.append(
                (self.last_state, self.last_action, reward, None, True))
        self.episode_done = True

    def receive_game_start_message(self, game_info):
        self.my_initial_stack = 1000
        self.total_seats = len(game_info['seats'])
        for i in range(len(game_info['seats'])):
            if game_info['seats'][i]['uuid'] == self.uuid:
                self.my_seat = i
                break

    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, new_action, round_state): pass

# --- 多進程資料蒐集 function ---


def collect_experiences(n_episodes, agent_weights, encoder_weights, queue):
    import torch
    import random
    import traceback
    try:
        torch.manual_seed(random.randint(0, 1000000))
        encoder = Encoder()
        dqn_agent = DQNAgent(input_dim=135, output_dim=5)
        # 使用 deepcopy 防止共享記憶體
        dqn_agent.q_net.load_state_dict(copy.deepcopy(agent_weights))
        # encoder通常是stateless，若有狀態也可載入
        experiences = []
        baselines = [
            baseline1_ai, baseline2_ai, baseline3_ai,
            baseline4_ai, baseline5_ai, baseline6_ai, baseline7_ai
        ]
        for _ in range(n_episodes):
            rl_player = RLAgent(dqn_agent, encoder)
            opponent_num = random.randint(0, 6)
            opponent_name = f"baseline{opponent_num + 1}"
            opponent_ai = baselines[opponent_num]()
            config = setup_config(
                max_round=20, initial_stack=1000, small_blind_amount=5)
            config.register_player(name="RLAgent", algorithm=rl_player)
            config.register_player(
                name=f"{opponent_name}", algorithm=opponent_ai)
            start_poker(config, verbose=0)
            # 回合結束後拉出 episode_memory
            if hasattr(rl_player, 'episode_memory'):
                experiences.extend(rl_player.episode_memory)
        queue.put(experiences)
    except Exception as e:
        print("[Child Process Error]", e)
        traceback.print_exc()
        # 放空回傳，避免主進程卡住
        queue.put([])

# --- Main 多進程 RL 訓練迴圈 ---


def RLtrain_parallel(
    total_episodes=5000, episodes_per_process=20, n_processes=4, checkpoint_path=None
):
    ctx = get_context("spawn")
    queue = ctx.Queue(maxsize=1000)  # 明確設定 queue 大小
    encoder = Encoder()
    dqn_agent = DQNAgent(input_dim=135, output_dim=5)
    if checkpoint_path is not None:
        print(f"Loading weights from {checkpoint_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(checkpoint_path, map_location=device)
        dqn_agent.q_net.load_state_dict(state_dict)
        dqn_agent.target_q_net.load_state_dict(state_dict)

    baselines = [baseline1_ai, baseline2_ai, baseline3_ai,
                 baseline4_ai, baseline5_ai, baseline6_ai, baseline7_ai]

    num_batches = total_episodes // (episodes_per_process * n_processes)
    print(
        f"Parallel RL Training: {n_processes} processes, {episodes_per_process} episodes each, {num_batches} batches.")
    update_target_every = 10
    log_every = 100
    print(dqn_agent.epsilon)
    for batch_idx in tqdm(range(num_batches)):
        # 多進程同時產生資料
        processes = []
        agent_weights = copy.deepcopy(dqn_agent.q_net.state_dict())
        encoder_weights = None  # encoder 若有狀態才需要丟
        for p in range(n_processes):
            proc = ctx.Process(
                target=collect_experiences,
                args=(episodes_per_process, agent_weights,
                      encoder_weights, queue)
            )
            proc.start()
            processes.append(proc)
        # 蒐集所有資料（有 timeout 防止卡死）
        batch_experiences = []
        for _ in range(n_processes):
            try:
                data = queue.get(timeout=120)
                batch_experiences.extend(data)
            except Exception as e:
                print("[Main] Timeout when waiting for subprocess queue.", e)
        # 等待所有子進程結束
        for proc in processes:
            proc.join(timeout=180)
            if proc.exitcode != 0:
                print(
                    f"[Warning] Process {proc.pid} exited abnormally with code {proc.exitcode}")
        # 填入 buffer
        for exp in batch_experiences:
            dqn_agent.buffer.append(exp)
        # 訓練
        dqn_agent.train()
        # 更新 target net
        if (batch_idx + 1) % update_target_every == 0:
            dqn_agent.update_target()
        # log
        if (batch_idx + 1) % log_every == 0:
            last_100_losses = dqn_agent.loss_history[-100:]
            print(
                f"[Batch {batch_idx+1}] Loss avg: {np.mean(last_100_losses) if last_100_losses else 0:.4f}")
        # 可加 win rate 評估...
        if batch_idx % 500 == 0:
            print(dqn_agent.epsilon)
            # 訓練過程 checkpoint 放到 results/，避免污染 repo 根目錄
            dqn_agent.save(f"results/rl/nn_poker_best_{batch_idx}.pt")
            print(f"completed {batch_idx} batches.")
    dqn_agent.save(f"results/rl/nn_poker_best_final.pt")
    np.savetxt(f"results/rl/loss_history.txt", np.array(dqn_agent.loss_history))


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    RLtrain_parallel(total_episodes=10000, episodes_per_process=1,
                     n_processes=10, checkpoint_path="models/policy_nn.pt")
