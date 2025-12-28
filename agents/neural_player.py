import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from game.players import BasePokerPlayer
import json
import os


class HandEncoder:
    """手牌和遊戲狀態編碼器"""

    def __init__(self):
        self.suits = ['C', 'D', 'H', 'S']  # 梅花、方塊、紅心、黑桃
        self.ranks = ['2', '3', '4', '5', '6',
                      '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        self.rank_to_idx = {rank: idx for idx, rank in enumerate(self.ranks)}
        self.suit_to_idx = {suit: idx for idx, suit in enumerate(self.suits)}

    def encode_card(self, card):
        """將單張牌編碼為one-hot向量 (17維：13個rank + 4個suit)"""
        if not card or len(card) != 2:
            return np.zeros(17)

        suit, rank = card[0], card[1]  # 格式：'H7' -> suit='H', rank='7'

        encoding = np.zeros(17)
        if rank in self.rank_to_idx:
            encoding[self.rank_to_idx[rank]] = 1  # rank encoding (0-12)
        if suit in self.suit_to_idx:
            encoding[13 + self.suit_to_idx[suit]] = 1  # suit encoding (13-16)

        return encoding

    def encode_hand(self, hole_cards):
        """編碼手牌 (2張牌 = 34維)"""
        if len(hole_cards) < 2:
            return np.zeros(34)

        card1_encoding = self.encode_card(hole_cards[0])
        card2_encoding = self.encode_card(hole_cards[1])
        return np.concatenate([card1_encoding, card2_encoding])

    def encode_community(self, community_cards):
        """編碼社區牌 (最多5張牌 = 85維)"""
        encoding = np.zeros(85)  # 5 * 17 = 85

        for i, card in enumerate(community_cards[:5]):
            card_encoding = self.encode_card(card)
            encoding[i*17:(i+1)*17] = card_encoding

        return encoding

    def encode_game_state(self, hole_cards, community_cards, round_count, my_stack, opponent_stack, pot_size=0):
        """編碼完整遊戲狀態"""
        hand_encoding = self.encode_hand(hole_cards)  # 34維
        community_encoding = self.encode_community(community_cards)  # 85維

        # 正規化數值特徵
        normalized_round = round_count / 20.0  # 假設最大20回合
        normalized_my_stack = my_stack / 2000.0  # 假設最大stack 2000
        normalized_opp_stack = opponent_stack / 2000.0
        normalized_pot = pot_size / 500.0  # 假設最大pot 500

        # 街道資訊 (4維 one-hot: preflop, flop, turn, river)
        street_encoding = np.zeros(4)
        if len(community_cards) == 0:
            street_encoding[0] = 1  # preflop
        elif len(community_cards) == 3:
            street_encoding[1] = 1  # flop
        elif len(community_cards) == 4:
            street_encoding[2] = 1  # turn
        elif len(community_cards) == 5:
            street_encoding[3] = 1  # river

        # 組合所有特徵
        features = np.concatenate([
            hand_encoding,  # 34維
            community_encoding,  # 85維
            [normalized_round, normalized_my_stack,
                normalized_opp_stack, normalized_pot],  # 4維
            street_encoding  # 4維
        ])

        return features  # 總計 127維


class PokerMLP(nn.Module):
    """德州撲克MLP模型"""

    def __init__(self, input_size=127, hidden_sizes=[256, 128, 64], output_size=3):
        super(PokerMLP, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Softmax(dim=-1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class NeuralPlayer(BasePokerPlayer):
    """基於神經網路的德州撲克玩家"""

    def __init__(self, model_path=None):
        super().__init__()
        self.encoder = HandEncoder()
        self.model = PokerMLP()
        self.training_data = []  # 用於收集訓練資料
        self.model_path = model_path or "agents/poker_model.pth"

        # 嘗試載入預訓練模型
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(
                    self.model_path, map_location='cpu'))
                print(f"模型已從 {self.model_path} 載入")
            except:
                print(f"無法載入模型 {self.model_path}，使用隨機初始化")

        self.model.eval()

    def declare_action(self, valid_actions, hole_card, round_state):
        # 提取遊戲狀態
        community_card = round_state['community_card']
        round_count = round_state.get('round_count', 1)

        # 獲取籌碼資訊
        my_stack = self._get_my_stack(round_state)
        opponent_stack = self._get_opponent_stack(round_state)
        pot_size = round_state.get('pot', {}).get('main', {}).get('amount', 0)

        # 編碼特徵
        features = self.encoder.encode_game_state(
            hole_card, community_card, round_count,
            my_stack, opponent_stack, pot_size
        )

        # 神經網路預測
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            action_probs = self.model(features_tensor).squeeze()

        # 將機率轉換為動作
        action_names = ['fold', 'call', 'raise']
        action_idx = torch.multinomial(action_probs, 1).item()
        chosen_action = action_names[action_idx]

        print(
            f"Neural Network決策 - Fold: {action_probs[0]:.3f}, Call: {action_probs[1]:.3f}, Raise: {action_probs[2]:.3f} -> {chosen_action}")

        # 執行動作
        if chosen_action == 'fold':
            return valid_actions[0]['action'], valid_actions[0]['amount']
        elif chosen_action == 'call':
            return valid_actions[1]['action'], valid_actions[1]['amount']
        else:  # raise
            if valid_actions[2]['amount']['min'] != -1:  # 可以加注
                raise_amount = min(
                    valid_actions[2]['amount']['min'] + 20,
                    valid_actions[2]['amount']['max']
                )
                return valid_actions[2]['action'], raise_amount
            else:  # 不能加注，改為跟注
                return valid_actions[1]['action'], valid_actions[1]['amount']

    def _get_my_stack(self, round_state):
        """獲取自己的籌碼數"""
        for seat in round_state['seats']:
            if seat['uuid'] == self.uuid:
                return seat['stack']
        return 1000

    def _get_opponent_stack(self, round_state):
        """獲取對手的籌碼數"""
        for seat in round_state['seats']:
            if seat['uuid'] != self.uuid:
                return seat['stack']
        return 1000

    def collect_training_data(self, hole_card, round_state, action, amount):
        """收集訓練資料"""
        community_card = round_state['community_card']
        round_count = round_state.get('round_count', 1)
        my_stack = self._get_my_stack(round_state)
        opponent_stack = self._get_opponent_stack(round_state)
        pot_size = round_state.get('pot', {}).get('main', {}).get('amount', 0)

        features = self.encoder.encode_game_state(
            hole_card, community_card, round_count,
            my_stack, opponent_stack, pot_size
        )

        # 動作標籤
        action_label = 0 if action == 'fold' else (
            1 if action == 'call' else 2)

        self.training_data.append({
            'features': features.tolist(),
            'action': action_label
        })

    def save_training_data(self, filename="training_data.json"):
        """儲存訓練資料"""
        with open(filename, 'w') as f:
            json.dump(self.training_data, f)
        print(f"訓練資料已儲存到 {filename}")

    def save_model(self):
        """儲存模型"""
        torch.save(self.model.state_dict(), self.model_path)
        print(f"模型已儲存到 {self.model_path}")

    # 必要的方法實作
    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return NeuralPlayer()
