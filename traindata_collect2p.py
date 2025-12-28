import numpy as np
import random
from multiprocessing import Process, Manager, cpu_count
from game.game import setup_config, start_poker
from agents.Encoder import Encoderfor2P as Encoder
from agents.rule_base import setup_ai as rule_ai
from agents.montecarlo import setup_ai as monte_ai
from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai
from baseline4 import setup_ai as baseline4_ai
from baseline5 import setup_ai as baseline5_ai
from baseline6 import setup_ai as baseline6_ai
from baseline7 import setup_ai as baseline7_ai
from tqdm import tqdm
ACTION_MAP = {
    'fold': 0,
    'call': 1,
    'raise_min': 2,
    'raise_mid': 3,
    'raise_max': 4
}


def map_action(action, amount, valid_actions):
    if action == 'fold':
        return 0
    elif action == 'call':
        return 1
    elif action == 'raise':
        min_raise = [va for va in valid_actions if va['action']
                     == "raise"][0]['amount']['min']
        max_raise = [va for va in valid_actions if va['action']
                     == "raise"][0]['amount']['max']
        mid_raise = int((min_raise + max_raise) / 2)
        if min_raise == -1:
            return 1
        if amount == min_raise:
            return 2
        elif amount == max_raise:
            return 4
        else:
            return 3
    else:
        return 0


class Recorder:
    def __init__(self):
        self.memory = []

    def record(self, state, action, valid_actions):
        self.memory.append((state, action, valid_actions))


def play_and_record_one_hand(agent, opponent, encoder, recorder):
    config = setup_config(
        max_round=20, initial_stack=1000, small_blind_amount=5)
    config.register_player(name="A", algorithm=agent)
    config.register_player(name="B", algorithm=opponent)
    orig_declare_action = agent.declare_action

    def hook_declare_action(valid_actions, hole_card, round_state):
        my_stack = [p['stack']
                    for p in round_state['seats'] if p['uuid'] == agent.uuid][0]
        opponent_stack = [p['stack']
                          for p in round_state['seats'] if p['uuid'] != agent.uuid][0]
        is_big_blind = 1 if round_state['big_blind_pos'] == round_state['seats'].index(
            [p for p in round_state['seats'] if p['uuid'] == agent.uuid][0]) else 0
        last_oppo_action = round_state['action_histories']['preflop'][-1]['action'] if 'preflop' in round_state[
            'action_histories'] and round_state['action_histories']['preflop'] else "CALL"
        call_amount = [va for va in valid_actions if va['action']
                       == "call"][0]['amount']
        min_raise = [va for va in valid_actions if va['action']
                     == "raise"][0]['amount']['min']
        max_raise = [va for va in valid_actions if va['action']
                     == "raise"][0]['amount']['max']
        state = encoder.encode_game_state(
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
        action, amount = orig_declare_action(
            valid_actions, hole_card, round_state)
        action_id = map_action(action, amount, valid_actions)
        recorder.record(state, action_id, valid_actions)
        return action, amount
    agent.declare_action = hook_declare_action
    start_poker(config, verbose=0)
    agent.declare_action = orig_declare_action


baselines = [
    baseline1_ai, baseline2_ai, baseline3_ai,
    baseline4_ai, baseline5_ai, baseline6_ai, baseline7_ai
]


def worker(num_hands, result_list):
    encoder = Encoder()
    local_memory = []
    for i in tqdm(range(num_hands)):
        opponent_ai = random.choice(baselines)()
        get_ai = monte_ai if random.random() < 0.5 else rule_ai
        agent = get_ai()
        recorder = Recorder()
        play_and_record_one_hand(agent, opponent_ai, encoder, recorder)
        for m in recorder.memory:
            local_memory.append(m)
    result_list += local_memory


if __name__ == "__main__":
    import math
    total_hands = 5000
    num_proc = min(cpu_count(), 11)
    hands_per_proc = math.ceil(total_hands / num_proc)

    manager = Manager()
    result_list = manager.list()
    processes = []

    for _ in range(num_proc):
        p = Process(target=worker, args=(hands_per_proc, result_list))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # 整理結果
    all_states = [x[0] for x in result_list]
    all_actions = [x[1] for x in result_list]
    np.savez(f"poker_data_with_baseline.npz", states=np.array(
        all_states), actions=np.array(all_actions))
    print("Data collection done.")
