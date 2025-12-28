import json
from game.game import setup_config, start_poker
from agents.neural_player import setup_ai as setup_neural
from agents.random_player import setup_ai as setup_random

# 設定遊戲參數
config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)

# 註冊兩位玩家：神經網路AI 與 隨機玩家
config.register_player(name="Neural_AI", algorithm=setup_neural())
config.register_player(name="Random_AI", algorithm=setup_random())

# 開始對戰，verbose=1 會輸出每一步
game_result = start_poker(config, verbose=1)

# 印出最終結果（以 JSON 結構印出）
print(json.dumps(game_result, indent=4))
