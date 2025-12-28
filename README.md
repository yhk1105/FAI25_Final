# FAI 人工智慧導論期末專案：Heads-up 德州撲克 AI（Rule / Monte Carlo / Supervised / DQN）

本專案在**兩人德州撲克（Heads-up Texas Hold’em）**環境中，實作多種 AI 玩家並與課程提供的 `baseline1~baseline7` 對戰比較。重點不是做 UI，而是把「**狀態表示 → 決策策略/模型 → 對戰評估**」完整串起來。

更完整的動機、方法推導與實驗結果整理在 `docs/report.pdf`。

---

## 專案主旨

- **問題**：在每個決策點（preflop/flop/turn/river）根據手牌、公共牌、底池、籌碼、對手行為等資訊，選擇合理動作（fold/call/raise 或更細的 raise bucket）。
- **方法**：
  - **Rule-based**：手寫啟發式規則（依街道、籌碼量、下注大小、牌力/聽牌等）
  - **Monte Carlo**：用模擬估 win rate，再用 pot odds / 閾值做決策
  - **Supervised NN**：蒐集 expert 行為資料，訓練 policy network 模仿決策
  - **DQN（RL）**：以 DQN 對 baseline 產生 experience、更新 Q network（可用 supervised policy 做初始化）
- **評估方式**：重複對戰多場，統計勝率/平均籌碼變化，並輸出 JSON 供後續分析。

---

## 專案重點貢獻（你做了哪些工程/研究工作）

- **撲克環境對接**：使用 `game/` 引擎（dealer、round manager、hand evaluator…）跑完整 heads-up 對局。
- **多種玩家（agents）**：在 `agents/` 中實作可插拔的 `BasePokerPlayer` 子類別。
- **狀態特徵設計（encoding）**：
  - `agents/neural_player.py`：127 維特徵（手牌/公共牌 one-hot + 正規化數值 + street）
  - `agents/Encoder.py`：135 維特徵（額外加入 call/min/max raise、effective stack、對手上一動作、是否 BB 等）
- **訓練與產出物**：
  - 監督式 policy 權重：`models/policy_nn.pt`
  - RL/DQN 訓練 checkpoint：`models/dqn_poker_best.pt`（其餘訓練中 checkpoint 會輸出到 `results/rl/`）

---

## 主要檔案導覽（看程式碼從哪裡開始）

- **撲克遊戲引擎（助教提供，這份專案較少著墨）**
  - `game/`：對局流程、發牌/下注、合法行動、round state 編碼等（你主要是「接 API 寫玩家」）

- **對戰入口**
  - `start_game_plain.py`：跑單場對戰（可快速切換 agent vs agent）
  - `start_game_2.py`：批次對 baseline1~7 多次對戰並輸出 `result_*.json`
- **AI 玩家**
  - `agents/rule_base.py`：規則式玩家
  - `agents/montecarlo.py`：Monte Carlo win-rate 玩家
  - `agents/neural_player.py`：監督式 NN 玩家（fold/call/raise）
  - `agents/RLplayer.py`：RL 玩家（載入 `models/policy_nn.pt`，輸出 5 actions：fold/call/raise_min/raise_mid/raise_max）
- **資料產生 / 訓練**
  - `traindata_collect2p.py`：**自動化蒐集監督式資料**（多進程跑對局、hook agent 決策，輸出 `.npz`）
  - `train_nn.py`：監督式 policy（5 actions）訓練（讀 `.npz`）
  - `reinforcement.py`：DQN 訓練（多進程蒐集 experience）
- **baseline**
  - `baseline*.so`：課程提供的 baseline agent（二進位）

---

## 如何重現（最小可重現）

### 安裝依賴

```bash
pip install -r requirements-min.txt
```

> 若要跑神經網路/RL 訓練（`train_nn.py`、`reinforcement.py`、`agents/RLplayer.py`），改用 `requirement.txt`。

### 跑一場對戰

```bash
python start_game_plain.py
```

### 批次評測 baseline（輸出結果 JSON）

```bash
python start_game_2.py
```

---

## 怎麼自動化生資料

這份專案主要有兩種資料來源：

- **(A) 監督式資料（state → action）**：用「expert（Rule-based / Monte Carlo）」在真實對局中的決策當標籤，做 imitation learning。
- **(B) 強化學習 experience（s, a, r, s'）**：RL 訓練時直接從對局回饋產生 transition。

### (A) 監督式資料：`traindata_collect2p.py`

`traindata_collect2p.py` 的做法是：

- **對局自動化**：用多進程反覆開局（expert vs baseline），不需要人工介入。
- **Hook/攔截決策點**：在每次 expert 的 `declare_action(...)` 被呼叫時：
  - 用 `agents/Encoder.py` 把當下 `round_state` 編碼成 **135 維 state 向量**
  - 把 expert 真正做出的動作（fold/call/raise）再離散成 **5 類 action label**（fold/call/raise_min/raise_mid/raise_max）
- **輸出格式**：最後輸出 `.npz`，包含：
  - `states`: shape = `(N, 135)`
  - `actions`: shape = `(N,)`，整數 0~4

你可以用下面指令產生一份 dataset（預設輸出到 `data/`）：

```bash
python traindata_collect2p.py
```
### (B) 強化學習資料：`reinforcement.py`

`reinforcement.py` 會用多進程同時跑多場對局，在每場對局中由 RL player 紀錄 transition（`episode_memory`），最後把 experience 丟回主進程的 replay buffer，進行 DQN 更新並定期存 checkpoint。

---

## 參考

- 報告與細節：`docs/report.pdf`


