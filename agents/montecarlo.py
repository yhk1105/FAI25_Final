from game.players import BasePokerPlayer
import random
import itertools
from agents.hand_evaluator import HandEvaluator   # 這裡請換成你的實際檔案名稱或模組路徑
from game.engine.card import Card
import time
rank_order = '23456789TJQKA'

def estimate_win_rate(nb_simulation, hole_card, community_card):
    suits = "CDHS"
    used = set(hole_card + community_card)
    base_deck = [s + r
                 for r in rank_order
                 for s in suits
                 if s + r not in used]
    need_comm = 5 - len(community_card)
    need_total = need_comm + 2
    start_time = time.time()
    win = tie = 0
    for _ in range(nb_simulation):
        draw = random.sample(base_deck, need_total)
        sim_comm = community_card + draw[:need_comm]
        opp_hands = draw[need_comm:]
        com = [Card.from_str(c) for c in sim_comm]
        hole = [Card.from_str(c) for c in hole_card]
        opp = [Card.from_str(c) for c in opp_hands]
        my_score = HandEvaluator.eval_hand(hole, com)
        opp_score = HandEvaluator.eval_hand(opp, com)
        if my_score > opp_score:
            win += 1
        elif my_score == opp_score:
            tie += 1/2
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    return (win + tie) / nb_simulation

class MonteCarloAgent(BasePokerPlayer):
    def __init__(self):
        self.call_amount = 0
        self.opponent_keep_raise = True
    def declare_action(self, valid_actions, hole_card, round_state):
        community_card = round_state['community_card']
        street = round_state['street']
        win_rate = estimate_win_rate(1000, hole_card, community_card)
        print("win rate: ", win_rate)
        pot = round_state['pot']['main']['amount']
        call_amt = [va['amount'] for va in valid_actions if va['action'] == "call"][0]
        raise_min = [va['amount']['min'] for va in valid_actions if va['action'] == "raise"][0]
        raise_max = [va['amount']['max'] for va in valid_actions if va['action'] == "raise"][0]
        my_stack = self._my_stack(round_state)
        action_histories = round_state['action_histories']
        street_order = ['preflop', 'flop', 'turn', 'river']
        current_index = street_order.index(street)

        if current_index > 0:
            previous_street = street_order[current_index - 1]
            previous_actions = action_histories.get(previous_street, [])
            if previous_actions:
                last_action = previous_actions[-1]['action']
                if last_action not in ['raise', 'BIGBLIND', 'SMALLBLIND']:
                    self.opponent_keep_raise = False
            else:
                self.opponent_keep_raise = False
        else:
            current_actions = action_histories.get('preflop', [])
            meaningful_actions = [a for a in current_actions if a['action'] not in ['SMALLBLIND', 'BIGBLIND']]
            if not meaningful_actions or meaningful_actions[-1]['action'] not in ['raise']:
                self.opponent_keep_raise = False

        pot_odds = call_amt / (pot + call_amt) if (pot + call_amt) > 0 else 0

        if my_stack < 10 * raise_min:
            raise_thres = 0.7
        elif my_stack > 100 * raise_min:
            raise_thres = 0.85
        else:
            raise_thres = 0.75

        def legal_raise(amt):
            return min(max(amt, raise_min), raise_max)

        if win_rate > 0.9:
            if raise_max == -1 or 'raise' not in [va['action'] for va in valid_actions]:
                return 'call', call_amt
            return 'raise', raise_max

        elif win_rate > raise_thres:
            amt = legal_raise((raise_min + raise_max) // 3)
            if raise_max == -1 or 'raise' not in [va['action'] for va in valid_actions]:
                return 'call', call_amt
            return 'raise', legal_raise(amt)

        elif win_rate > 0.6:
            amt = legal_raise(raise_min + 10)
            if raise_max == -1 or 'raise' not in [va['action'] for va in valid_actions]:
                return 'call', call_amt
            return 'raise', legal_raise(amt)
        elif self.opponent_keep_raise and street in ['turn', 'river']:
            return 'fold', 0
        elif win_rate < 0.3 and street in ['turn', 'river']:
            return 'fold', 0
        elif win_rate > pot_odds:
            return 'call', call_amt
        elif random.random() < 0.1 and street != 'preflop':
            if raise_max == -1 or 'raise' not in [va['action'] for va in valid_actions]:
                return 'call', call_amt
            return 'raise', legal_raise(raise_min)
        else:
            return 'fold', 0

    def _my_stack(self, round_state):
        for seat in round_state['seats']:
            if seat['uuid'] == self.uuid:
                return seat['stack']
        return 0
    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): 
        self.opponent_keep_raise = True
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, new_action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

def setup_ai():
    return MonteCarloAgent()
