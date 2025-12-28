import numpy as np


class Encoderfor2P:
    """
    兩人對戰（2-Player）版本的狀態編碼器。

    輸出維度：135
      - 手牌：34（2 張牌 * 17）
      - 公共牌：85（5 張牌 * 17，不足補空）
      - 數值特徵：4（round / my_stack / opp_stack / pot）
      - 數值特徵：4（call / min_raise / max_raise / effective_stack）
      - street one-hot：4
      - 對手上一動作 one-hot：3（raise/call/fold）
      - 是否 big blind：1
    """

    def __init__(self):
        self.suits = ["C", "D", "H", "S"]
        self.ranks = ["2", "3", "4", "5", "6", "7", "8",
                      "9", "T", "J", "Q", "K", "A"]
        self.rank_to_idx = {rank: idx for idx, rank in enumerate(self.ranks)}
        self.suit_to_idx = {suit: idx for idx, suit in enumerate(self.suits)}

    def encode_card(self, card: str):
        if not card or len(card) != 2:
            return np.zeros(17)
        suit, rank = card[0], card[1]
        encoding = np.zeros(17)
        if rank in self.rank_to_idx:
            encoding[self.rank_to_idx[rank]] = 1
        if suit in self.suit_to_idx:
            encoding[13 + self.suit_to_idx[suit]] = 1
        return encoding

    def encode_hand(self, hole_cards):
        if len(hole_cards) < 2:
            return np.zeros(34)
        return np.concatenate([self.encode_card(hole_cards[0]), self.encode_card(hole_cards[1])])

    def encode_community(self, community_cards):
        encoding = np.zeros(85)
        if len(community_cards) < 5:
            community_cards = community_cards + [""] * (5 - len(community_cards))
        for i, card in enumerate(community_cards[:5]):
            encoding[i * 17:(i + 1) * 17] = self.encode_card(card)
        return encoding

    def encode_game_state(
        self,
        hole_cards,
        community_cards,
        round_count,
        my_stack,
        opponent_stack,
        last_opponent_action,
        is_big_blind,
        pot_size=0,
        call_amount=0,
        min_raise=0,
        max_raise=0,
    ):
        hand_encoding = self.encode_hand(hole_cards)
        community_encoding = self.encode_community(community_cards)

        normalized_round = round_count / 20.0
        normalized_my_stack = my_stack / 2000.0
        normalized_opp_stack = opponent_stack / 2000.0
        normalized_pot = pot_size / 2000.0

        normalized_call = call_amount / 2000.0
        normalized_min_raise = min_raise / 2000.0
        normalized_max_raise = max_raise / 2000.0
        effective_stack = min(my_stack, opponent_stack) / 2000.0

        street_encoding = np.zeros(4)
        if len(community_cards) == 0:
            street_encoding[0] = 1  # preflop
        elif len(community_cards) == 3:
            street_encoding[1] = 1  # flop
        elif len(community_cards) == 4:
            street_encoding[2] = 1  # turn
        elif len(community_cards) == 5:
            street_encoding[3] = 1  # river

        last_opponent_action_encoding = np.zeros(3)
        if last_opponent_action == "raise":
            last_opponent_action_encoding[0] = 1
        elif last_opponent_action == "call":
            last_opponent_action_encoding[1] = 1
        elif last_opponent_action == "fold":
            last_opponent_action_encoding[2] = 1

        features = np.concatenate(
            [
                hand_encoding,  # 34
                community_encoding,  # 85
                [normalized_round, normalized_my_stack, normalized_opp_stack, normalized_pot],  # 4
                [normalized_call, normalized_min_raise, normalized_max_raise, effective_stack],  # 4
                street_encoding,  # 4
                last_opponent_action_encoding,  # 3
                [is_big_blind],  # 1
            ]
        )
        return features


