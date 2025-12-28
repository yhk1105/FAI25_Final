from game.players import BasePokerPlayer
import random

class MyAgent(BasePokerPlayer):
    def __init__(self):
        self.round_count = 0
        self.hole_card = None
        self.total_players = 0
        self.my_seat = 0

    def declare_action(self, valid_actions, hole_card, round_state):
        street = round_state['street']
        community_card = round_state['community_card']
        my_stack = self._my_stack(round_state)
        pot = round_state['pot']['main']['amount']
        rank_order = '23456789TJQKA'
        c1_rank, c2_rank = hole_card[0][1], hole_card[1][1]
        c1_value, c2_value = rank_order.index(c1_rank), rank_order.index(c2_rank)
        is_pair = c1_rank == c2_rank
        suited = hole_card[0][0] == hole_card[1][0]
        connected = abs(c1_value - c2_value) == 1
        min_raise = valid_actions[2]['amount']['min']
        max_raise = valid_actions[2]['amount']['max']
        call_amt = valid_actions[1]['amount']

        def test_val(action, amount):
            # 只在能raise的時候raise，否則call
            if action == 'raise' and min_raise == -1:
                return 'call', call_amt
            elif action == 'raise':
                return 'raise', max(min(amount, max_raise), min_raise)
            if 'raise' not in valid_actions:
                return 'call', call_amt
            return action, amount

        # ==========【短碼超保守策略】==========
        # 極短碼防守，優先判斷
        if my_stack < 150:
            # 很強牌才推 all-in
            if is_pair or c1_value >= 10 or c2_value >= 10 or suited or connected:
                return test_val('raise', my_stack)
            # 只 call 最小注，不進攻
            if call_amt <= 10:
                return 'call', call_amt
            return 'fold', 0

        # ==========【大額 all-in call 防守】==========
        # 只用 trips 以上才 call/all-in
        if call_amt > 0.6 * my_stack:
            if self.should_call_allin(hole_card, community_card, call_amt, my_stack):
                return 'call', call_amt
            else:
                return 'fold', 0

        # ==========【200 以下進階保守偷雞】==========
        if my_stack < 200:
            strength = self._hand_strength(hole_card, community_card)
            # trips 以上才梭哈
            if strength in ['trips', 'straight', 'flush', 'fullhouse', 'quads', 'straight_flush']:
                return test_val('raise', my_stack)
            # flop/turn 沒人下注時 20% 偷 pot 20%
            if call_amt == 0 and street in ['flop', 'turn'] and random.random() < 0.2:
                bluff_amt = min(int(pot * 0.2), max_raise)
                if min_raise <= bluff_amt <= max_raise:
                    return test_val('raise', bluff_amt)
            if call_amt <= 10:
                return 'call', call_amt
            return 'fold', 0

        # ==========【Preflop 常規邏輯】==========
        if street == 'preflop':
            if call_amt > 60:
                if (is_pair and c1_value >= 9) or (c1_value >= 11 and c2_value >= 10):
                    return 'call', call_amt
                else:
                    return 'fold', 0
            if is_pair and c1_value >= 8 or (c1_value >= 10 and c2_value >= 10):
                raise_amt = min(22, max_raise)
                if min_raise <= raise_amt:
                    return test_val('raise', raise_amt)
            if suited and connected and min(c1_value, c2_value) >= 5:
                raise_amt = min(15, max_raise)
                if min_raise <= raise_amt:
                    return test_val('raise', raise_amt)
            return 'call', call_amt

        # ==========【Flop/Turn/River 常規邏輯】==========
        has_pair = self._has_pair(hole_card, community_card)
        two_pair_or_better = self._has_two_pair_or_better(hole_card, community_card)
        strong_draw = self._has_flush_or_straight_draw(hole_card, community_card)
        is_nut = self._is_nut(hole_card, community_card)

        if street in ['flop', 'turn', 'river']:
            # 對手大加注沒兩對/堅果不跟
            if call_amt > pot * 0.5:
                if two_pair_or_better or is_nut:
                    return 'call', call_amt
                else:
                    return 'fold', 0

            # 河牌，只做 value bet 或 check
            if street == 'river':
                strength = self._hand_strength(hole_card, community_card)
                if strength in ['two_pair', 'trips', 'straight', 'flush', 'fullhouse', 'quads', 'straight_flush']:
                    return test_val('raise', my_stack)
                elif strength == 'pair':
                    if random.random() < 0.5:
                        return test_val('raise', int(pot * 0.1))
                    else:
                        return 'call', call_amt
                else:
                    if call_amt == 0:
                        return 'call', 0
                    return 'fold', 0

            # flop/turn有兩對以上主動下注
            if two_pair_or_better:
                bet_amt = min(int(pot * 0.7), max_raise)
                if bet_amt >= min_raise:
                    return test_val('raise', bet_amt)
            # flop有pair或強draw下注
            if has_pair or strong_draw:
                bet_amt = min(int(pot * 0.4), max_raise)
                if bet_amt >= min_raise:
                    return test_val('raise', bet_amt)
            # flop沒人攻擊，20%偷池
            if street == 'flop' and call_amt == 0 and random.random() < 0.2:
                if min_raise <= int(pot * 0.3) <= max_raise:
                    return test_val('raise', int(pot * 0.3))
            if call_amt == 0:
                return 'call', 0
            else:
                return 'fold', 0

        return 'fold', 0

    def should_call_allin(self, hole_card, community_card, bet_amount, my_stack):
        # 只有 trips 以上才 call all-in
        strength = self._hand_strength(hole_card, community_card)
        return strength in ['trips', 'straight', 'flush', 'fullhouse', 'quads', 'straight_flush']

    def _hand_strength(self, hole_card, community_card):
        all_cards = hole_card + community_card
        ranks = [c[1] for c in all_cards]
        suits = [c[0] for c in all_cards]
        for r in set(ranks):
            if ranks.count(r) == 4:
                return 'quads'
        if any(ranks.count(r) == 3 for r in set(ranks)) and any(ranks.count(r) == 2 for r in set(ranks)):
            return 'fullhouse'
        if max([suits.count(s) for s in set(suits)]) >= 5:
            return 'flush'
        rank_order = '23456789TJQKA'
        values = sorted(set([rank_order.index(r) for r in ranks]))
        for i in range(len(values) - 4):
            if values[i + 4] - values[i] == 4:
                return 'straight'
        if any(ranks.count(r) == 3 for r in set(ranks)):
            return 'trips'
        pairs = [r for r in set(ranks) if ranks.count(r) == 2]
        if len(pairs) >= 2:
            return 'two_pair'
        if len(pairs) == 1:
            return 'pair'
        return 'nothing'

    def _my_stack(self, round_state):
        for seat in round_state['seats']:
            if seat['uuid'] == self.uuid:
                return seat['stack']
        return 0

    def _has_pair(self, hole_card, community_card):
        all_cards = hole_card + community_card
        ranks = [c[1] for c in all_cards]
        return len(set(ranks)) < len(ranks)

    def _has_two_pair_or_better(self, hole_card, community_card):
        all_cards = hole_card + community_card
        ranks = [c[1] for c in all_cards]
        counts = [ranks.count(r) for r in set(ranks)]
        return counts.count(2) >= 2 or max(counts) >= 3

    def _has_flush_or_straight_draw(self, hole_card, community_card):
        all_cards = hole_card + community_card
        suits = [c[0] for c in all_cards]
        flush_draw = max([suits.count(s) for s in set(suits)]) >= 4
        rank_order = '23456789TJQKA'
        values = sorted([rank_order.index(c[1]) for c in all_cards])
        for i in range(len(values) - 3):
            window = values[i:i + 4]
            if window[-1] - window[0] <= 4:
                return True
        return flush_draw

    def _is_nut(self, hole_card, community_card):
        rank_order = '23456789TJQKA'
        all_cards = hole_card + community_card
        if len(community_card) < 5:
            return False
        suits = [c[0] for c in all_cards]
        ranks = [c[1] for c in all_cards]
        for suit in set(suits):
            suit_cards = [c for c in all_cards if c[0] == suit]
            suit_ranks = set([c[1] for c in suit_cards])
            if set('TJQKA').issubset(suit_ranks):
                return True
        for suit in set(suits):
            suit_cards = [c for c in all_cards if c[0] == suit]
            if len(suit_cards) >= 5:
                values = sorted([rank_order.index(c[1]) for c in suit_cards], reverse=True)
                for i in range(len(values) - 4):
                    window = values[i:i + 5]
                    if window[0] - window[4] == 4 and window[0] == 12:
                        return True
        if ranks.count('A') == 4:
            return True
        for suit in set(suits):
            suit_cards = [c for c in all_cards if c[0] == suit]
            if len(suit_cards) >= 5:
                suit_values = [rank_order.index(c[1]) for c in suit_cards]
                if max(suit_values) == 12 and ranks.count('A') >= 1:
                    return True
        values = sorted(set([rank_order.index(r) for r in ranks]), reverse=True)
        for i in range(len(values) - 4):
            window = values[i:i + 5]
            if window[0] - window[4] == 4 and window[0] == 12:
                return True
        if ranks.count('A') == 3:
            for r in set(ranks):
                if r != 'A' and ranks.count(r) >= 2:
                    return True
        return False

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats):
        self.round_count = round_count
        self.hole_card = hole_card
        self.total_players = len(seats)
        self.my_seat = 0
        for i in range(len(seats)):
            if seats[i]['uuid'] == self.uuid:
                self.my_seat = i
                break
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, new_action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

def setup_ai():
    return MyAgent()
