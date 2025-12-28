import json
from game.game import setup_config, start_poker
from agents.rule_base import setup_ai as rule_base_ai
from agents.montecarlo import setup_ai as setup_ai_1
from agents.random_player import setup_ai as setup_ai_2
from agents.RLplayer import setup_ai as setup_ai_3
from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai
from baseline4 import setup_ai as baseline4_ai
from baseline5 import setup_ai as baseline5_ai
from baseline6 import setup_ai as baseline6_ai
from baseline7 import setup_ai as baseline7_ai
import time
# from agents.call_player import setup_ai as call_ai   ← 若你需要可加回來
# from agents.console_player import setup_ai as console_ai ← 若你要手動玩
game_results = []
# 設定遊戲參數
for baseline in range(1, 8):
    for i in range(5):
        config = setup_config(
            max_round=20, initial_stack=1000, small_blind_amount=5)
        config.register_player(name="nn_10000", algorithm=setup_ai_3())
        if baseline == 1:
            config.register_player(name="baseline1", algorithm=baseline1_ai())
        elif baseline == 2:
            config.register_player(name="baseline2", algorithm=baseline2_ai())
        elif baseline == 3:
            config.register_player(name="baseline3", algorithm=baseline3_ai())
        elif baseline == 4:
            config.register_player(name="baseline4", algorithm=baseline4_ai())
        elif baseline == 5:
            config.register_player(name="baseline5", algorithm=baseline5_ai())
        elif baseline == 6:
            config.register_player(name="baseline6", algorithm=baseline6_ai())
        elif baseline == 7:
            config.register_player(name="baseline7", algorithm=baseline7_ai())
        time_start = time.time()
        game_result = start_poker(config, verbose=1)
        time_take = time.time() - time_start
        game_result['baseline'] = baseline
        game_result['i'] = i
        game_result['time_take'] = time_take
        game_result['stack'] = game_result['players'][0]['stack']

        game_results.append(game_result)

filename = f"result_nn_10000.json"

with open(filename, "w") as f:
    json.dump(game_results, f, indent=4)
