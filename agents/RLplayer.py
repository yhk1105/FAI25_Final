from __future__ import annotations

import pathlib
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game.players import BasePokerPlayer
from agents.Encoder import Encoderfor2P as Encoder2p


class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, input_dim: int, output_dim: int):
        self.q_net = DQN(input_dim, output_dim)
        self.target_q_net = DQN(input_dim, output_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.buffer = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 0.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.output_dim = output_dim

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.output_dim - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_net(state_t)
        return torch.argmax(q_values, dim=1).item()

    def update_target(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        self.q_net.load_state_dict(torch.load(path, map_location="cpu"))
        self.target_q_net.load_state_dict(self.q_net.state_dict())


class RLAgent(BasePokerPlayer):
    """
    供 `start_game_2.py`/`start_game_plain.py` 使用的 RL player。
    會載入 `agents/policy_nn.pt`（監督式 policy）作為行為網路權重。
    """

    def __init__(self, model_path: str | None = None):
        super().__init__()
        self.dqn = DQNAgent(input_dim=135, output_dim=5)

        default_path = pathlib.Path(__file__).resolve().parent / "policy_nn.pt"
        self.model_path = str(default_path) if model_path is None else model_path

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(self.model_path, map_location=device)
        self.dqn.q_net.load_state_dict(state_dict)
        self.dqn.target_q_net.load_state_dict(state_dict)

        self.encoder = Encoder2p()
        self.episode_memory = []
        self.last_state = None
        self.last_action = None
        self.my_seat = None
        self.total_seats = 0
        self.win_history = []
        self.my_initial_stack = 1000

    def declare_action(self, valid_actions, hole_card, round_state):
        is_big_blind = 1 if round_state["big_blind_pos"] == self.my_seat else 0
        my_stack = round_state["seats"][self.my_seat]["stack"]
        opponent_stack = round_state["seats"][1 - self.my_seat]["stack"]
        last_oppo_action = (
            round_state["action_histories"]["preflop"][-1]["action"]
            if "preflop" in round_state["action_histories"] and round_state["action_histories"]["preflop"]
            else "CALL"
        )

        call_amount = [va for va in valid_actions if va["action"] == "call"][0]["amount"]
        min_raise = [va for va in valid_actions if va["action"] == "raise"][0]["amount"]["min"]
        max_raise = [va for va in valid_actions if va["action"] == "raise"][0]["amount"]["max"]

        state = self.encoder.encode_game_state(
            hole_cards=hole_card,
            community_cards=round_state["community_card"],
            round_count=round_state["round_count"],
            my_stack=my_stack,
            opponent_stack=opponent_stack,
            last_opponent_action=last_oppo_action,
            is_big_blind=is_big_blind,
            pot_size=round_state["pot"]["main"]["amount"],
            call_amount=call_amount,
            min_raise=min_raise,
            max_raise=max_raise,
        )

        action_idx = self.dqn.select_action(state)
        action_name = ["fold", "call", "raise_min", "raise_mid", "raise_max"][action_idx]

        action = None
        amount = None
        for va in valid_actions:
            if action_name == "fold" and va["action"] == "fold":
                action, amount = "fold", va["amount"]
                break
            if (action_name == "call" and va["action"] == "call") or (
                action_name.startswith("raise") and len(valid_actions) == 2
            ):
                action, amount = "call", va["amount"]
                break
            if action_name.startswith("raise") and va["action"] == "raise":
                if va["amount"]["min"] == -1:
                    fallback = next(a for a in valid_actions if a["action"] == "call")
                    action, amount = fallback["action"], fallback["amount"]
                else:
                    if action_name == "raise_min":
                        action, amount = "raise", va["amount"]["min"]
                    elif action_name == "raise_mid":
                        action, amount = "raise", int((va["amount"]["max"] + va["amount"]["min"]) / 2)
                    elif action_name == "raise_max":
                        action, amount = "raise", va["amount"]["max"]
                break

        if action is None:
            fallback = valid_actions[0]
            action, amount = fallback["action"], fallback["amount"]

        if self.last_state is not None:
            self.episode_memory.append((self.last_state, self.last_action, 0, state, False))
        self.last_state = state
        self.last_action = action_idx
        return action, amount

    def receive_round_result_message(self, winners, hand_info, round_state):
        my_final_stack = [p["stack"] for p in round_state["seats"] if p["uuid"] == self.uuid][0]
        reward = my_final_stack - self.my_initial_stack
        self.my_initial_stack = my_final_stack
        if self.last_state is not None:
            self.episode_memory.append((self.last_state, self.last_action, reward, None, True))

        self.episode_memory = []
        self.last_state = None
        self.last_action = None

        is_win = any(winner["uuid"] == self.uuid for winner in winners)
        self.win_history.append(1 if is_win else 0)
        if len(self.win_history) % 100 == 0:
            print(f"[Eval] {len(self.win_history)} games - Win rate (last 100): {np.mean(self.win_history[-100:]):.3f}")

    def receive_game_start_message(self, game_info):
        self.my_initial_stack = 1000
        self.total_seats = len(game_info["seats"])
        for i in range(len(game_info["seats"])):
            if game_info["seats"][i]["uuid"] == self.uuid:
                self.my_seat = i
                break

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass


def setup_ai():
    return RLAgent()


